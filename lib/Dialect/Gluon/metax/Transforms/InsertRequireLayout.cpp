/*
 * 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
 * Reserved.
 */
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "triton/Dialect/Gluon/metax/Analysis/LayoutPropagation.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "gluon-insert-require-layout"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[insert] " << X << "\n")

using namespace mlir;
using namespace mlir::dataflow;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace gd = mlir::triton::gluon;

namespace {

struct SharedRequirement {
  ttg::LocalLoadOp load;
  ttg::SwizzledSharedEncodingAttr encoding;
};

struct RequirementPlan {
  SmallVector<SharedRequirement> shared;
};

// Return the root of a descriptor chain only when every view preserves the
// encoding order. Transpose and reshape require dialect layout inference and
// are deliberately left to PropagateLayout rather than approximated here.
Value findOrderPreservingRoot(Value value) {
  while (Operation *definition = value.getDefiningOp()) {
    if (isa<gd::RequireLayoutOp, ttg::MemDescIndexOp,
            ttg::MemDescSubsliceOp, ttg::MemDescReinterpretOp>(definition)) {
      value = definition->getOperand(0);
      continue;
    }
    break;
  }
  return value;
}

class ProducerOrderState {
public:
  void meet(ArrayRef<unsigned> candidate) {
    if (conflict)
      return;
    if (!order) {
      order = llvm::to_vector(candidate);
      return;
    }
    conflict = ArrayRef<unsigned>(*order) != candidate;
  }

  bool isConflict() const { return conflict; }
  ArrayRef<unsigned> getOrder() const {
    assert(order && !conflict && "expected a concrete producer order");
    return *order;
  }
  explicit operator bool() const { return order.has_value() && !conflict; }

private:
  std::optional<SmallVector<unsigned>> order;
  bool conflict = false;
};

bool isTrackedDotValue(Value value) {
  return isa<RankedTensorType>(value.getType());
}

// Insert only consumes the concrete M0 edges produced by the existing C500
// Accelerate pass. The standard converter's B0 DotOperand conversions remain
// part of the initial dot contract and must not become Gluon requirements.
bool isMacaDotOperandEncoding(Attribute encoding) {
  return gd::isSupportedDotConstraintEncoding(encoding);
}

bool isTransparentDotUserBeforeConstraintMaterialization(Operation *op,
                                                          unsigned operandIndex) {
  if (auto dot = dyn_cast<tt::DotOp>(op))
    return operandIndex < 2 && operandIndex < dot->getNumOperands();
  return isa<ttg::ConvertLayoutOp, gd::ExtractSliceOp, gd::InsertSliceOp,
             ttg::BsmPermOp>(op) ||
         gd::isTransparentLayoutCarrierOp(op);
}

// Directly ported from TLX InsertRequireLayout. The state records both the
// selected DotOperand encoding and whether a local_load-to-dot rewrite remains
// legal while the graph is still untouched.
class DotRewriteState {
public:
  enum class Kind {
    Uninitialized,
    Required,
    Conflict,
    Illegal,
  };

  DotRewriteState() = default;
  explicit DotRewriteState(Attribute encoding)
      : kind(Kind::Required), encoding(encoding) {}

  static DotRewriteState getConflict() {
    DotRewriteState state;
    state.kind = Kind::Conflict;
    return state;
  }

  static DotRewriteState getIllegal() {
    DotRewriteState state;
    state.kind = Kind::Illegal;
    return state;
  }

  bool operator==(const DotRewriteState &rhs) const {
    return kind == rhs.kind && encoding == rhs.encoding;
  }

  bool isUninitialized() const { return kind == Kind::Uninitialized; }
  bool isRequired() const { return kind == Kind::Required; }
  bool isConflict() const { return kind == Kind::Conflict; }
  bool isIllegal() const { return kind == Kind::Illegal; }

  Attribute getEncoding() const {
    assert(isRequired() && "expected a required dot encoding");
    return *encoding;
  }

  void print(raw_ostream &os) const {
    if (isUninitialized()) {
      os << "<uninitialized>";
      return;
    }
    if (isConflict()) {
      os << "<conflict>";
      return;
    }
    if (isIllegal()) {
      os << "<illegal>";
      return;
    }
    getEncoding().print(os);
  }

  friend raw_ostream &operator<<(raw_ostream &os,
                                 const DotRewriteState &state) {
    state.print(os);
    return os;
  }

  static DotRewriteState meet(const DotRewriteState &lhs,
                              const DotRewriteState &rhs) {
    if (lhs.isIllegal() || rhs.isIllegal())
      return getIllegal();
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    if (lhs == rhs)
      return lhs;
    return getConflict();
  }

  static DotRewriteState join(const DotRewriteState &lhs,
                              const DotRewriteState &rhs) {
    return meet(lhs, rhs);
  }

private:
  Kind kind = Kind::Uninitialized;
  std::optional<Attribute> encoding;
};

class DotRewriteLattice : public Lattice<DotRewriteState> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DotRewriteLattice)
  using Lattice::Lattice;
};

class DotRewriteBackward
    : public SparseBackwardDataFlowAnalysis<DotRewriteLattice> {
public:
  using SparseBackwardDataFlowAnalysis<DotRewriteLattice>::
      SparseBackwardDataFlowAnalysis;
  using SparseBackwardDataFlowAnalysis<DotRewriteLattice>::
      propagateIfChanged;

  void initializeEquivalentLatticeAnchor(Operation *top) override {
    top->walk([&](ttg::ConvertLayoutOp convert) {
      if (isTrackedDotValue(convert.getSrc()) &&
          isTrackedDotValue(convert.getResult()))
        unionLatticeAnchors<DotRewriteLattice>(convert.getSrc(),
                                               convert.getResult());
    });
  }

  LogicalResult
  visitOperation(Operation *op, ArrayRef<DotRewriteLattice *> operands,
                 ArrayRef<const DotRewriteLattice *> results) override {
    if (auto dot = dyn_cast<tt::DotOp>(op)) {
      for (unsigned operandIndex : {0u, 1u}) {
        auto type =
            cast<RankedTensorType>(dot.getOperand(operandIndex).getType());
        Attribute layout = type.getEncoding();
        if (!isMacaDotOperandEncoding(layout))
          continue;
        auto encoding = cast<ttg::DotOperandEncodingAttr>(layout);
        ChangeResult changed =
            operands[operandIndex]->meet(DotRewriteState(encoding));
        propagateIfChanged(operands[operandIndex], changed);
      }
      return success();
    }

    // Register slices and logical BSM are same-layout carriers.  They only
    // carry the requirement back to their source(s); Insert never anchors a
    // require_layout on either side of these operations.  The late C500
    // lowering owns BSM's i32 physical carrier ABI.
    if (isa<gd::ExtractSliceOp, gd::InsertSliceOp, ttg::BsmPermOp>(op)) {
      if (results.size() != 1)
        return failure();
      DotRewriteState state = results.front()->getValue();
      if (state.isUninitialized())
        return success();
      for (DotRewriteLattice *operand : operands) {
        ChangeResult changed = operand->meet(state);
        propagateIfChanged(operand, changed);
      }
      return success();
    }

    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (!isTrackedDotValue(operand) ||
          isTransparentDotUserBeforeConstraintMaterialization(op, index))
        continue;

      DotRewriteState state = operands[index]->getValue();
      if (state.isUninitialized())
        continue;
      LDBG("value=" << operand << " state=illegal user=" << op->getName());
      ChangeResult changed =
          operands[index]->meet(DotRewriteState::getIllegal());
      propagateIfChanged(operands[index], changed);
    }
    return success();
  }

  void visitBranchOperand(OpOperand &operand) override {
    if (!isTrackedDotValue(operand.get()) ||
        gd::isTransparentLayoutCarrierOp(operand.getOwner()))
      return;
    poisonUnhandledCase(operand);
  }

  void visitCallOperand(OpOperand &operand) override {
    poisonUnhandledCase(operand);
  }

  void visitNonControlFlowArguments(RegionSuccessor &,
                                    ArrayRef<BlockArgument>) {}

  void setToExitState(DotRewriteLattice *) override {}

private:
  void poisonUnhandledCase(OpOperand &operand) {
    if (!isTrackedDotValue(operand.get()))
      return;
    DotRewriteLattice *lattice = getLatticeElement(operand.get());
    if (lattice->getValue().isUninitialized())
      return;
    ChangeResult changed = lattice->meet(DotRewriteState::getIllegal());
    propagateIfChanged(lattice, changed);
  }
};

std::optional<ttg::SwizzledSharedEncodingAttr>
deriveMmaSharedEncoding(ttg::LocalLoadOp load,
                        ttg::DotOperandEncodingAttr dotOperand) {
  auto sharedType = cast<ttg::MemDescType>(load.getSrc().getType());
  auto mma = cast<ttg::MACAMmaEncodingAttr>(dotOperand.getParent());
  SmallVector<unsigned> sharedOrder = ttg::getOrder(sharedType);
  const unsigned sharedRank = sharedOrder.size();
  Value endpointRoot = findOrderPreservingRoot(load.getSrc());
  ProducerOrderState asyncOrder;
  ProducerOrderState registerOrder;

  load->getParentOfType<ModuleOp>().walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<ttg::AsyncCopyGlobalToLocalOp>([&](auto copy) {
          if (findOrderPreservingRoot(copy.getResult()) != endpointRoot)
            return;
          auto sourceType = cast<RankedTensorType>(copy.getSrc().getType());
          SmallVector<unsigned> order = ttg::getOrderForMemory(sourceType);
          asyncOrder.meet(order);
          LLVM_DEBUG({
            llvm::dbgs() << "[insert] root=" << endpointRoot
                         << " async producer=" << copy.getSrc() << " order=[";
            llvm::interleaveComma(order, llvm::dbgs());
            llvm::dbgs() << "]\n";
          });
        })
        .Case<ttg::LocalStoreOp>([&](auto store) {
          if (findOrderPreservingRoot(store.getDst()) != endpointRoot)
            return;
          auto sourceType = cast<RankedTensorType>(store.getSrc().getType());
          SmallVector<unsigned> order = ttg::getOrderForMemory(sourceType);
          registerOrder.meet(order);
          LLVM_DEBUG({
            llvm::dbgs() << "[insert] root=" << endpointRoot
                         << " local_store producer=" << store.getSrc()
                         << " order=[";
            llvm::interleaveComma(order, llvm::dbgs());
            llvm::dbgs() << "]\n";
          });
        })
        .Case<ttg::LocalAllocOp>([&](auto alloc) {
          if (alloc.getResult() != endpointRoot || !alloc.getSrc())
            return;
          auto sourceType = cast<RankedTensorType>(alloc.getSrc().getType());
          SmallVector<unsigned> order = ttg::getOrderForMemory(sourceType);
          registerOrder.meet(order);
          LLVM_DEBUG({
            llvm::dbgs() << "[insert] root=" << endpointRoot
                         << " initializer=" << alloc.getSrc() << " order=[";
            llvm::interleaveComma(order, llvm::dbgs());
            llvm::dbgs() << "]\n";
          });
        });
  });

  if (asyncOrder.isConflict()) {
    load.emitRemark()
        << "cannot fold dot layout into Shared because async-copy producers "
           "require conflicting memory orders";
    return std::nullopt;
  }
  if (asyncOrder) {
    sharedOrder.assign(asyncOrder.getOrder().begin(),
                       asyncOrder.getOrder().end());
  } else if (registerOrder.isConflict()) {
    load.emitRemark()
        << "cannot fold dot layout into Shared because register producers "
           "prefer conflicting memory orders";
    return std::nullopt;
  } else if (registerOrder) {
    sharedOrder.assign(registerOrder.getOrder().begin(),
                       registerOrder.getOrder().end());
  }

  if (sharedOrder.size() != sharedRank) {
    load.emitRemark() << "cannot fold dot layout into Shared because producer "
                         "order rank does not match the Shared encoding rank";
    return std::nullopt;
  }

  return mma.composeSharedLayoutForOperand(
      dotOperand, mma.getCTALayout(), sharedType.getShape(), sharedOrder,
      sharedType.getElementType().getIntOrFloatBitWidth(),
      /*needTrans=*/false);
}

LogicalResult collectMmaSharedRequirements(RequirementPlan &plan,
                                           ModuleOp module) {
  SymbolTableCollection symbolTable;
  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<SparseConstantPropagation>();
  solver.load<DotRewriteBackward>(symbolTable);
  if (failed(solver.initializeAndRun(module)))
    return failure();

  module.walk([&](ttg::LocalLoadOp load) {
    const DotRewriteLattice *lattice =
        solver.lookupState<DotRewriteLattice>(load.getResult());
    if (!lattice || lattice->getValue().isUninitialized())
      return;
    if (lattice->getValue().isIllegal() ||
        lattice->getValue().isConflict()) {
      LDBG("local_load=" << load.getLoc()
                           << " skipped state=" << lattice->getValue());
      load.emitRemark()
          << "dot operand layout constraint cannot be folded into local_load "
             "because the value has incompatible users or conflicting dot "
             "requirements";
      return;
    }
    auto dotOperand =
        dyn_cast<ttg::DotOperandEncodingAttr>(lattice->getValue().getEncoding());
    if (!dotOperand || !isa<ttg::MACAMmaEncodingAttr>(dotOperand.getParent()))
      return;
    std::optional<ttg::SwizzledSharedEncodingAttr> sharedEncoding =
        deriveMmaSharedEncoding(load, dotOperand);
    if (!sharedEncoding)
      return;
    plan.shared.push_back(SharedRequirement{load, *sharedEncoding});
    LDBG("local_load=" << load.getLoc() << " operand="
                        << (dotOperand.getOpIdx() == 0 ? "A" : "B")
                        << " endpoint=" << load.getSrc()
                        << " S0=" << *sharedEncoding);
  });

  return success();
}

// GluonAccelerateMatmul leaves a direct convert_layout on each M0 dot operand.
// Replace only that edge with the internal constraint;
// the earlier B0-to-DotOperand conversion remains part of the standard
// converter contract.
void materializeTensorRequireLayout(tt::DotOp dot, unsigned operandIndex) {
  Value operand = dot.getOperand(operandIndex);
  auto convert = operand.getDefiningOp<ttg::ConvertLayoutOp>();
  if (!convert)
    return;

  auto targetType = dyn_cast<RankedTensorType>(convert.getType());
  if (!targetType || !isMacaDotOperandEncoding(targetType.getEncoding()))
    return;

  OpBuilder builder(convert);
  auto require = builder.create<gd::RequireLayoutOp>(
      convert.getLoc(), convert.getType(), convert.getSrc());
  dot.setOperand(operandIndex, require.getResult());
  if (convert.getResult().use_empty())
    convert.erase();
}

void materializeDotUserTensorConstraints(ModuleOp module) {
  module.walk([&](tt::DotOp dot) {
    for (unsigned operandIndex : {0u, 1u})
      materializeTensorRequireLayout(dot, operandIndex);
  });
}

void materializeSharedRequirements(ArrayRef<SharedRequirement> requirements) {
  for (const SharedRequirement &requirement : requirements) {
    OpOperand &operand = requirement.load->getOpOperand(/*index=*/0);
    Value source = operand.get();
    if (source.getDefiningOp<gd::RequireLayoutOp>())
      continue;
    auto sourceType = cast<ttg::MemDescType>(source.getType());
    auto targetType = ttg::MemDescType::get(
        sourceType.getShape(), sourceType.getElementType(), requirement.encoding,
        sourceType.getMemorySpace(), sourceType.getMutableMemory(),
        sourceType.getAllocShape());
    OpBuilder builder(requirement.load);
    auto require = builder.create<gd::RequireLayoutOp>(source.getLoc(),
                                                        targetType, source);
    operand.set(require.getResult());
  }
}

} // namespace

namespace mlir::triton::gluon {

#define GEN_PASS_DEF_GLUONINSERTREQUIRELAYOUTPASS
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h.inc"

namespace {

class GluonInsertRequireLayoutPass final
    : public impl::GluonInsertRequireLayoutPassBase<
          GluonInsertRequireLayoutPass> {
public:
  void runOnOperation() override {
    RequirementPlan plan;
    if (failed(collectMmaSharedRequirements(plan, getOperation()))) {
      signalPassFailure();
      return;
    }
    materializeDotUserTensorConstraints(getOperation());
    materializeSharedRequirements(plan.shared);
  }
};

} // namespace

std::unique_ptr<Pass> createGluonInsertRequireLayoutPass() {
  return std::make_unique<GluonInsertRequireLayoutPass>();
}

} // namespace mlir::triton::gluon

#undef LDBG
#undef DEBUG_TYPE
