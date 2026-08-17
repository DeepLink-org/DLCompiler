#include "triton/Dialect/Gluon/metax/Analysis/LayoutPropagation.h"

#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "gluon-layout-propagation"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[propagate] " << X << "\n")

using namespace mlir;
using namespace mlir::dataflow;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::gluon {
namespace {

bool isMemDesc(Value value) { return isa<ttg::MemDescType>(value.getType()); }
bool isTensor(Value value) { return isa<RankedTensorType>(value.getType()); }

ttg::MemDescType withEncoding(ttg::MemDescType type, Attribute encoding) {
  return ttg::MemDescType::get(type.getShape(), type.getElementType(), encoding,
                               type.getMemorySpace(), type.getMutableMemory(),
                               type.getAllocShape());
}

FailureOr<Attribute> inferTransEncoding(Attribute encoding,
                                        ArrayRef<int64_t> shape,
                                        ArrayRef<int32_t> order,
                                        Location loc) {
  auto *interface =
      encoding.getDialect()
          .getRegisteredInterface<triton::DialectInferLayoutInterface>();
  if (!interface)
    return failure();
  Attribute inferred;
  if (failed(interface->inferTransOpEncoding(encoding, shape, order, inferred,
                                             loc)))
    return failure();
  return inferred;
}

SmallVector<int32_t> invertPermutation(ArrayRef<int32_t> order) {
  SmallVector<int32_t> inverse(order.size());
  for (auto [index, dim] : llvm::enumerate(order))
    inverse[dim] = index;
  return inverse;
}

template <typename AnalysisT, typename LatticeT, typename StateT>
void meetAndPropagate(AnalysisT *analysis, LatticeT *lattice,
                      const StateT &state) {
  ChangeResult changed = lattice->meet(state);
  analysis->propagateIfChanged(lattice, changed);
}

} // namespace

void LayoutEncoding::print(raw_ostream &os) const {
  if (isUninitialized())
    os << "<UNINITIALIZED>";
  else if (isUnknown())
    os << "<UNKNOWN>";
  else
    getLayoutEncoding().print(os);
}

LayoutEncoding LayoutEncoding::meet(const LayoutEncoding &lhs,
                                    const LayoutEncoding &rhs) {
  if (lhs.isUnknown() || rhs.isUnknown())
    return getUnknownLayout();
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  return lhs == rhs ? lhs : getUnknownLayout();
}

LayoutEncoding LayoutEncoding::join(const LayoutEncoding &lhs,
                                    const LayoutEncoding &rhs) {
  return meet(lhs, rhs);
}

LogicalResult LayoutBackwardPropagation::visitRegionInReverse(Operation *op) {
  for (Region &region : llvm::reverse(op->getRegions())) {
    for (Block &block : llvm::reverse(region)) {
      for (Operation &nested : llvm::reverse(block)) {
        SmallVector<LayoutEncodingLattice *> operands;
        for (Value operand : nested.getOperands())
          operands.push_back(getLatticeElement(operand));
        SmallVector<const LayoutEncodingLattice *> results;
        for (Value result : nested.getResults())
          results.push_back(getLatticeElement(result));
        if (failed(visitOperation(&nested, operands, results)))
          return failure();
      }
    }
  }
  return success();
}

void LayoutBackwardPropagation::visitWarpSpecRegionArgs(
    Operation *op, Value operand, const LayoutEncoding &encoding) {
  auto argument = dyn_cast<BlockArgument>(operand);
  if (!argument)
    return;
  auto partitions = op->getParentOfType<ttg::WarpSpecializePartitionsOp>();
  if (!partitions)
    return;

  auto warpSpecialize = partitions.getParentOp();
  unsigned index = argument.getArgNumber();
  meetAndPropagate(this,
                   getLatticeElement(warpSpecialize.getExplicitCaptures()[index]),
                   encoding);
  for (Region *partition : warpSpecialize.getPartitionRegions())
    meetAndPropagate(this, getLatticeElement(partition->getArgument(index)),
                     encoding);
}

LogicalResult LayoutBackwardPropagation::visitOperation(
    Operation *op, ArrayRef<LayoutEncodingLattice *> operands,
    ArrayRef<const LayoutEncodingLattice *> results) {
  if (isa<LocalAliasOp>(op))
    return success();

  if (isa<RegionBranchOpInterface, ttg::WarpSpecializePartitionsOp>(op))
    return visitRegionInReverse(op);

  if (auto trans = dyn_cast<ttg::MemDescTransOp>(op)) {
    LayoutEncoding state = results.front()->getValue();
    if (state.isUninitialized() || state.isUnknown())
      return success();
    auto resultType = cast<ttg::MemDescType>(trans.getType());
    FailureOr<Attribute> source = inferTransEncoding(
        state.getLayoutEncoding(), resultType.getShape(),
        invertPermutation(trans.getOrder()), op->getLoc());
    if (failed(source))
      return failure();
    LayoutEncoding sourceState(*source);
    meetAndPropagate(this, operands.front(), sourceState);
    visitWarpSpecRegionArgs(op, trans.getSrc(), sourceState);
    return success();
  }

  if (auto reshape = dyn_cast<ttg::MemDescReshapeOp>(op)) {
    LayoutEncoding state = results.front()->getValue();
    if (state.isUninitialized() || state.isUnknown())
      return success();
    auto resultType = cast<ttg::MemDescType>(reshape.getType());
    auto constrainedResult = withEncoding(resultType, state.getLayoutEncoding());
    auto sourceType = cast<ttg::MemDescType>(reshape.getSrc().getType());
    ttg::MemDescType inferredSource;
    if (failed(ttg::MemDescReshapeOp::inferReturnTypes(
            op->getContext(), op->getLoc(), constrainedResult,
            sourceType.getShape(), inferredSource)))
      return failure();
    LayoutEncoding sourceState(inferredSource.getEncoding());
    meetAndPropagate(this, operands.front(), sourceState);
    visitWarpSpecRegionArgs(op, reshape.getSrc(), sourceState);
    return success();
  }

  if (auto require = dyn_cast<RequireLayoutOp>(op)) {
    if (isa<RankedTensorType>(require.getType()))
      return success();
    LayoutEncoding state(cast<ttg::MemDescType>(require.getType()).getEncoding());
    for (auto [lattice, operand] : llvm::zip_equal(operands, op->getOperands())) {
      meetAndPropagate(this, lattice, state);
      visitWarpSpecRegionArgs(op, operand, state);
    }
    return success();
  }

  // Match TLX: backward propagation is generic for memdesc operands. Forward
  // propagation remains restricted to view operations with defined semantics.
  for (const LayoutEncodingLattice *result : results) {
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (!isMemDesc(operand))
        continue;
      meetAndPropagate(this, operands[index], result->getValue());
      visitWarpSpecRegionArgs(op, operand, result->getValue());
    }
  }
  return success();
}

void LayoutBackwardPropagation::visitBranchOperand(OpOperand &operand) {
  Operation *branch = operand.getOwner();
  if (!isa<ttg::WarpSpecializeOp, ttg::WarpSpecializePartitionsOp>(branch))
    return;
  Operation *regionOwner = isa<ttg::WarpSpecializePartitionsOp>(branch)
                               ? branch->getParentOp()
                               : branch;
  (void)visitRegionInReverse(regionOwner);
}

void LayoutBackwardPropagation::visitCallOperand(OpOperand &) {
  llvm_unreachable("calls must be eliminated before layout propagation");
}

void LayoutBackwardPropagation::setToExitState(LayoutEncodingLattice *) {}

LogicalResult LayoutForwardPropagation::visitRegion(Operation *op) {
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nested : block) {
        SmallVector<const LayoutEncodingLattice *> operands;
        for (Value operand : nested.getOperands())
          operands.push_back(getLatticeElement(operand));
        SmallVector<LayoutEncodingLattice *> results;
        for (Value result : nested.getResults())
          results.push_back(getLatticeElement(result));
        if (failed(visitOperation(&nested, operands, results)))
          return failure();
      }
    }
  }
  return success();
}

LogicalResult LayoutForwardPropagation::visitWarpSpecRegionArgs(
    Value result, const LayoutEncoding &encoding) {
  for (OpOperand &use : result.getUses()) {
    auto warpSpecialize = dyn_cast<ttg::WarpSpecializeOp>(use.getOwner());
    if (!warpSpecialize)
      continue;
    unsigned index = use.getOperandNumber();
    if (index >= warpSpecialize.getExplicitCaptures().size())
      continue;
    for (Region *partition : warpSpecialize.getPartitionRegions())
      meetAndPropagate(this, getLatticeElement(partition->getArgument(index)),
                       encoding);
    if (failed(visitRegion(warpSpecialize)))
      return failure();
  }
  return success();
}

LogicalResult LayoutForwardPropagation::visitOperation(
    Operation *op, ArrayRef<const LayoutEncodingLattice *> operands,
    ArrayRef<LayoutEncodingLattice *> results) {
  if (isa<LocalAliasOp>(op))
    return success();
  if (isa<RegionBranchOpInterface, ttg::WarpSpecializePartitionsOp>(op))
    return visitRegion(op);
  if (!isa<ttg::MemDescIndexOp, ttg::MemDescReinterpretOp,
           ttg::MemDescSubsliceOp, ttg::MemDescTransOp,
           ttg::MemDescReshapeOp, ttg::LocalAllocOp>(op))
    return success();

  for (auto [index, operandLattice] : llvm::enumerate(operands)) {
    if (!isMemDesc(op->getOperand(index)))
      continue;
    LayoutEncoding state = operandLattice->getValue();
    if (state.isUninitialized())
      continue;

    if (auto trans = dyn_cast<ttg::MemDescTransOp>(op)) {
      if (!state.isUnknown()) {
        auto sourceType = cast<ttg::MemDescType>(trans.getSrc().getType());
        FailureOr<Attribute> inferred = inferTransEncoding(
            state.getLayoutEncoding(), sourceType.getShape(), trans.getOrder(),
            op->getLoc());
        if (failed(inferred))
          return failure();
        state = LayoutEncoding(*inferred);
      }
    } else if (auto reshape = dyn_cast<ttg::MemDescReshapeOp>(op)) {
      if (!state.isUnknown()) {
        auto sourceType = cast<ttg::MemDescType>(reshape.getSrc().getType());
        auto constrainedSource = withEncoding(sourceType, state.getLayoutEncoding());
        auto resultType = cast<ttg::MemDescType>(reshape.getType());
        ttg::MemDescType inferredResult;
        if (failed(ttg::MemDescReshapeOp::inferReturnTypes(
                op->getContext(), op->getLoc(), constrainedSource,
                resultType.getShape(), inferredResult)))
          return failure();
        state = LayoutEncoding(inferredResult.getEncoding());
      }
    }

    for (LayoutEncodingLattice *result : results) {
      // Match replaceUsesAndPropagateType's MemDescReshape handling: preserve
      // an already-constrained consumer-facing result encoding when reshape
      // inference canonicalizes the same physical mapping to SharedLinear.
      if (auto reshape = dyn_cast<ttg::MemDescReshapeOp>(op)) {
        LayoutEncoding current = result->getValue();
        if (!state.isUninitialized() && !state.isUnknown() &&
            !current.isUninitialized() && !current.isUnknown()) {
          auto resultType = cast<ttg::MemDescType>(reshape.getType());
          if (ttg::areLayoutsEquivalent(
                  resultType.getShape(),
                  cast<ttg::LayoutEncodingTrait>(state.getLayoutEncoding()),
                  cast<ttg::LayoutEncodingTrait>(
                      current.getLayoutEncoding())))
            continue;
        }
      }
      meetAndPropagate(this, result, state);
    }
  }

  for (auto [index, result] : llvm::enumerate(op->getResults()))
    if (failed(visitWarpSpecRegionArgs(result, results[index]->getValue())))
      return failure();
  return success();
}

void LayoutForwardPropagation::setToEntryState(LayoutEncodingLattice *) {}

void TensorLayout::print(raw_ostream &os) const {
  if (isUninitialized())
    os << "<UNINITIALIZED>";
  else if (isUnknown())
    os << "<UNKNOWN>";
  else
    getLayoutEncoding().print(os);
}

TensorLayout TensorLayout::meet(const TensorLayout &lhs,
                                const TensorLayout &rhs) {
  if (lhs.isUnknown() || rhs.isUnknown())
    return getUnknownLayout();
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  return lhs == rhs ? lhs : getUnknownLayout();
}

TensorLayout TensorLayout::join(const TensorLayout &lhs,
                                const TensorLayout &rhs) {
  return meet(lhs, rhs);
}

namespace {

bool isAllowedTensorLayoutUser(Operation *op, unsigned operandIndex) {
  if (auto require = dyn_cast<RequireLayoutOp>(op)) {
    auto type = dyn_cast<RankedTensorType>(require.getType());
    return operandIndex == 0 && type &&
           isSupportedDotConstraintEncoding(type.getEncoding());
  }
  return isa<ttg::ConvertLayoutOp>(op) || isTransparentLayoutCarrierOp(op);
}

bool canRewriteTensorResult(Operation *op) {
  return isa<ttg::LocalLoadOp, ttg::BsmPermOp, ExtractSliceOp,
             InsertSliceOp, RegionBranchOpInterface>(op);
}

// These three operations are layout-transparent register glue.  They do not
// choose a layout and never materialize a conversion: every tensor edge in
// the op participates in one meet, so a concrete requirement simply passes
// through and a disagreement becomes the ordinary Unknown/Conflict state.
void propagateSameTensorEncoding(
    TensorBackwardPropagation *analysis, ArrayRef<TensorLayoutLattice *> operands,
    ArrayRef<const TensorLayoutLattice *> results) {
  TensorLayout consensus;
  for (TensorLayoutLattice *lattice : operands)
    consensus = TensorLayout::meet(consensus, lattice->getValue());
  for (const TensorLayoutLattice *lattice : results)
    consensus = TensorLayout::meet(consensus, lattice->getValue());

  for (TensorLayoutLattice *lattice : operands)
    meetAndPropagate(analysis, lattice, consensus);
  for (const TensorLayoutLattice *lattice : results)
    meetAndPropagate(analysis, const_cast<TensorLayoutLattice *>(lattice),
                     consensus);
}

} // namespace

LogicalResult TensorBackwardPropagation::visitOperation(
    Operation *op, ArrayRef<TensorLayoutLattice *> operands,
    ArrayRef<const TensorLayoutLattice *> results) {
  if (isa<ExtractSliceOp, InsertSliceOp, ttg::BsmPermOp>(op)) {
    propagateSameTensorEncoding(this, operands, results);
    return success();
  }

  if (auto require = dyn_cast<RequireLayoutOp>(op)) {
    auto type = dyn_cast<RankedTensorType>(require.getType());
    if (!type || !isSupportedDotConstraintEncoding(type.getEncoding()))
      return success();
    TensorLayout state(type.getEncoding());
    for (auto [lattice, operand] : llvm::zip_equal(operands, op->getOperands()))
      if (isTensor(operand))
        meetAndPropagate(this, lattice, state);
    return success();
  }

  if (auto convert = dyn_cast<ttg::ConvertLayoutOp>(op)) {
    if (!results.empty() && isTensor(convert.getSrc())) {
      TensorLayout state = results.front()->getValue();
      if (!state.isUnknown())
        meetAndPropagate(this, operands.front(), state);
    }
  }

  for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
    if (!isTensor(operand) || isAllowedTensorLayoutUser(op, index))
      continue;
    TensorLayout state = operands[index]->getValue();
    if (!state.isUninitialized())
      meetAndPropagate(this, operands[index], TensorLayout::getUnknownLayout());
  }

  if (!canRewriteTensorResult(op)) {
    for (Value result : op->getResults()) {
      if (!isTensor(result))
        continue;
      TensorLayoutLattice *lattice = getLatticeElement(result);
      if (!lattice->getValue().isUninitialized())
        meetAndPropagate(this, lattice, TensorLayout::getUnknownLayout());
    }
  }
  return success();
}

void TensorBackwardPropagation::visitBranchOperand(OpOperand &operand) {
  if (!isTensor(operand.get()) ||
      isTransparentLayoutCarrierOp(operand.getOwner()))
    return;
  TensorLayoutLattice *lattice = getLatticeElement(operand.get());
  if (!lattice->getValue().isUninitialized())
    meetAndPropagate(this, lattice, TensorLayout::getUnknownLayout());
}

void TensorBackwardPropagation::visitCallOperand(OpOperand &operand) {
  if (!isTensor(operand.get()))
    return;
  TensorLayoutLattice *lattice = getLatticeElement(operand.get());
  if (!lattice->getValue().isUninitialized())
    meetAndPropagate(this, lattice, TensorLayout::getUnknownLayout());
}

void TensorBackwardPropagation::setToExitState(TensorLayoutLattice *) {}

} // namespace mlir::triton::gluon

#undef LDBG
#undef DEBUG_TYPE
