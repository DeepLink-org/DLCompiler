#include "triton/Dialect/Gluon/metax/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Gluon/metax/Analysis/LayoutPropagation.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gluon-propagate-layout"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[propagate] " << X << "\n")

using namespace mlir;
using namespace mlir::dataflow;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::gluon {

#define GEN_PASS_DEF_GLUONPROPAGATELAYOUTPASS
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h.inc"

namespace {

class RequireLayoutPattern : public OpRewritePattern<RequireLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RequireLayoutOp require,
                                PatternRewriter &rewriter) const override {
    if (!isa<RankedTensorType>(require.getSrc().getType()))
      return failure();
    if (require.getSrc().getType() == require.getType()) {
      rewriter.replaceOp(require, require.getSrc());
      return success();
    }
    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(
        require, require.getType(), require.getSrc());
    return success();
  }
};

RankedTensorType getNewTensorType(RankedTensorType type, Attribute encoding) {
  return RankedTensorType::get(type.getShape(), type.getElementType(), encoding);
}

ttg::MemDescType getNewMemDescType(ttg::MemDescType type,
                                   Attribute encoding) {
  return ttg::MemDescType::get(type.getShape(), type.getElementType(), encoding,
                               type.getMemorySpace(), type.getMutableMemory(),
                               type.getAllocShape());
}

bool isRetaggableTensorProducerValue(Value value) {
  if (!isa<RankedTensorType>(value.getType()))
    return false;
  return isa_and_nonnull<ttg::LocalLoadOp, ttg::BsmPermOp, ExtractSliceOp,
                         InsertSliceOp, RegionBranchOpInterface>(
      value.getDefiningOp());
}

Type getTensorCandidateType(Value value, DataFlowSolver &solver,
                            const DenseSet<Value> &blockedValues) {
  auto type = cast<RankedTensorType>(value.getType());
  if (blockedValues.contains(value))
    return type;
  const auto *lattice = solver.lookupState<TensorLayoutLattice>(value);
  if (!lattice || lattice->getValue().isUninitialized() ||
      lattice->getValue().isUnknown())
    return type;
  return getNewTensorType(type, lattice->getValue().getLayoutEncoding());
}

void rewriteTensorValueFromLattice(Value value, DataFlowSolver &solver,
                                   const DenseSet<Value> &blockedValues) {
  if (!isRetaggableTensorProducerValue(value))
    return;
  auto type = cast<RankedTensorType>(value.getType());
  auto newType = cast<RankedTensorType>(
      getTensorCandidateType(value, solver, blockedValues));
  if (type != newType) {
    LDBG("retag tensor value " << value << " to " << newType.getEncoding());
    value.setType(newType);
  }
}

FailureOr<const LayoutEncodingLattice *>
lookupMemDescLattice(Value value, DataFlowSolver &solver,
                     Operation *diagnosticOp) {
  if (auto *lattice = solver.lookupState<LayoutEncodingLattice>(value))
    return lattice;
  diagnosticOp->emitError() << "expected memdesc layout lattice for " << value;
  return failure();
}

FailureOr<LayoutEncoding>
getMemDescConsensusLayout(ArrayRef<Value> values, DataFlowSolver &solver,
                          Operation *diagnosticOp) {
  LayoutEncoding consensus;
  for (Value value : values) {
    FailureOr<const LayoutEncodingLattice *> lattice =
        lookupMemDescLattice(value, solver, diagnosticOp);
    if (failed(lattice))
      return failure();
    consensus = LayoutEncoding::join(consensus, (*lattice)->getValue());
  }
  return consensus;
}

LogicalResult rewriteMemDescValueFromLattice(Value value,
                                             DataFlowSolver &solver,
                                             Operation *diagnosticOp) {
  auto type = dyn_cast<ttg::MemDescType>(value.getType());
  if (!type)
    return success();
  FailureOr<const LayoutEncodingLattice *> lattice =
      lookupMemDescLattice(value, solver, diagnosticOp);
  if (failed(lattice))
    return failure();
  LayoutEncoding layout = (*lattice)->getValue();
  if (layout.isUninitialized() || layout.isUnknown())
    return success();
  auto newType = getNewMemDescType(type, layout.getLayoutEncoding());
  if (type != newType) {
    LDBG("retag memdesc value " << value << " to "
                                 << newType.getEncoding());
    value.setType(newType);
  }
  return success();
}

bool areSameRegionSuccessor(RegionSuccessor lhs, RegionSuccessor rhs) {
  if (lhs.getSuccessor() != rhs.getSuccessor())
    return false;
  ValueRange lhsInputs = lhs.getSuccessorInputs();
  ValueRange rhsInputs = rhs.getSuccessorInputs();
  return lhsInputs.size() == rhsInputs.size() &&
         llvm::equal(lhsInputs, rhsInputs);
}

// This LLVM revision predates RegionBranchOpInterface::getSuccessorInputs and
// getPredecessorValues. Derive the same generic interface facts without
// falling back to an scf-op list.
void collectRegionBranchSuccessors(
    RegionBranchOpInterface branchOp,
    SmallVectorImpl<RegionSuccessor> &successors) {
  auto appendUnique = [&](ArrayRef<RegionSuccessor> candidates) {
    for (RegionSuccessor candidate : candidates) {
      if (llvm::none_of(successors, [&](RegionSuccessor successor) {
            return areSameRegionSuccessor(successor, candidate);
          }))
        successors.push_back(candidate);
    }
  };

  SmallVector<RegionSuccessor> candidates;
  branchOp.getSuccessorRegions(RegionBranchPoint::parent(), candidates);
  appendUnique(candidates);
  for (Region &region : branchOp->getRegions()) {
    candidates.clear();
    branchOp.getSuccessorRegions(region, candidates);
    appendUnique(candidates);
  }
}

void getRegionBranchPredecessorValues(
    RegionBranchOpInterface branchOp, RegionSuccessor successor,
    unsigned index, SmallVectorImpl<Value> &predecessors) {
  ValueRange successorInputs = successor.getSuccessorInputs();
  assert(index < successorInputs.size());

  SmallVector<RegionSuccessor> candidates;
  branchOp.getSuccessorRegions(RegionBranchPoint::parent(), candidates);
  if (llvm::any_of(candidates, [&](RegionSuccessor candidate) {
        return areSameRegionSuccessor(candidate, successor);
      })) {
    OperandRange operands =
        branchOp.getEntrySuccessorOperands(RegionBranchPoint(successor));
    assert(operands.size() == successorInputs.size());
    predecessors.push_back(operands[index]);
  }

  for (Region &region : branchOp->getRegions()) {
    candidates.clear();
    branchOp.getSuccessorRegions(region, candidates);
    if (llvm::none_of(candidates, [&](RegionSuccessor candidate) {
          return areSameRegionSuccessor(candidate, successor);
        }))
      continue;

    for (Block &block : region) {
      auto terminator = dyn_cast<RegionBranchTerminatorOpInterface>(
          block.getTerminator());
      if (!terminator)
        continue;
      SmallVector<Attribute> constants(terminator->getNumOperands(),
                                       Attribute());
      SmallVector<RegionSuccessor> terminatorSuccessors;
      terminator.getSuccessorRegions(constants, terminatorSuccessors);
      if (llvm::none_of(terminatorSuccessors,
                        [&](RegionSuccessor candidate) {
                          return areSameRegionSuccessor(candidate, successor);
                        }))
        continue;
      OperandRange operands =
          terminator.getSuccessorOperands(RegionBranchPoint(successor));
      assert(operands.size() == successorInputs.size());
      predecessors.push_back(operands[index]);
    }
  }
}

std::optional<Type>
getTensorConsensusType(ValueRange values, DataFlowSolver &solver,
                       const DenseSet<Value> &blockedValues) {
  if (values.empty())
    return std::nullopt;
  std::optional<Type> consensus;
  for (Value value : values) {
    if (!isa<RankedTensorType>(value.getType()))
      return std::nullopt;
    Type candidate = getTensorCandidateType(value, solver, blockedValues);
    if (!consensus)
      consensus = candidate;
    else if (*consensus != candidate)
      return std::nullopt;
  }
  return consensus;
}

DenseSet<Value> computeBlockedTensorValues(triton::FuncOp func,
                                           DataFlowSolver &solver) {
  DenseSet<Value> blocked;
  bool changed = true;
  while (changed) {
    changed = false;
    func.walk([&](RegionBranchOpInterface branchOp) {
      SmallVector<RegionSuccessor> successors;
      collectRegionBranchSuccessors(branchOp, successors);
      for (RegionSuccessor successor : successors) {
        for (auto [index, successorInput] :
             llvm::enumerate(successor.getSuccessorInputs())) {
          if (!isa<RankedTensorType>(successorInput.getType()))
            continue;
          SmallVector<Value> predecessors;
          getRegionBranchPredecessorValues(branchOp, successor, index,
                                           predecessors);
          if (predecessors.empty() ||
              getTensorConsensusType(predecessors, solver, blocked))
            continue;
          changed |= blocked.insert(successorInput).second;
          for (Value predecessor : predecessors)
            if (isa<RankedTensorType>(predecessor.getType()))
              changed |= blocked.insert(predecessor).second;
        }
      }
      return WalkResult::advance();
    });
  }
  return blocked;
}

void updateTensorRegionBranchTypes(triton::FuncOp func, DataFlowSolver &solver,
                                   const DenseSet<Value> &blockedValues) {
  func.walk<WalkOrder::PostOrder>([&](RegionBranchOpInterface branchOp) {
    SmallVector<RegionSuccessor> successors;
    collectRegionBranchSuccessors(branchOp, successors);
    bool changed = true;
    while (changed) {
      changed = false;
      for (RegionSuccessor successor : successors) {
        for (auto [index, successorInput] :
             llvm::enumerate(successor.getSuccessorInputs())) {
          if (!isa<RankedTensorType>(successorInput.getType()))
            continue;
          SmallVector<Value> predecessors;
          getRegionBranchPredecessorValues(branchOp, successor, index,
                                           predecessors);
          std::optional<Type> consensus =
              getTensorConsensusType(predecessors, solver, blockedValues);
          if (!consensus || successorInput.getType() == *consensus)
            continue;
          successorInput.setType(*consensus);
          changed = true;
        }
      }
    }
  });
}

LogicalResult propagateLayouts(triton::FuncOp func) {
  WalkResult constrained = func.walk([&](RequireLayoutOp) {
    return WalkResult::interrupt();
  });
  if (!constrained.wasInterrupted())
    return success();

  SymbolTableCollection symbolTable;
  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<SparseConstantPropagation>();
  solver.load<LayoutBackwardPropagation>(symbolTable);
  solver.load<LayoutForwardPropagation>();
  solver.load<TensorBackwardPropagation>(symbolTable);
  if (failed(solver.initializeAndRun(func)))
    return failure();

  DenseSet<Value> blocked = computeBlockedTensorValues(func, solver);
  WalkResult rewrite = func.walk([&](Operation *op) {
    if (isa<RequireLayoutOp>(op))
      return WalkResult::advance();

    if (auto warpSpecialize = dyn_cast<ttg::WarpSpecializeOp>(op)) {
      for (auto [index, capture] :
           llvm::enumerate(warpSpecialize.getExplicitCaptures())) {
        auto captureType = dyn_cast<ttg::MemDescType>(capture.getType());
        if (!captureType)
          continue;
        SmallVector<Value> related{capture};
        for (Region *partition : warpSpecialize.getPartitionRegions())
          related.push_back(partition->getArgument(index));
        FailureOr<LayoutEncoding> consensus =
            getMemDescConsensusLayout(related, solver, warpSpecialize);
        if (failed(consensus))
          return WalkResult::interrupt();
        if (consensus->isUninitialized() || consensus->isUnknown())
          continue;
        auto newType =
            getNewMemDescType(captureType, consensus->getLayoutEncoding());
        capture.setType(newType);
        for (Region *partition : warpSpecialize.getPartitionRegions())
          partition->getArgument(index).setType(newType);
      }
      return WalkResult::advance();
    }

    for (Value result : op->getResults()) {
      if (isa<ttg::MemDescType>(result.getType())) {
        if (failed(rewriteMemDescValueFromLattice(result, solver, op)))
          return WalkResult::interrupt();
      } else {
        rewriteTensorValueFromLattice(result, solver, blocked);
      }
    }
    return WalkResult::advance();
  });
  if (rewrite.wasInterrupted())
    return failure();

  updateTensorRegionBranchTypes(func, solver, blocked);
  return success();
}

class GluonPropagateLayoutPass
    : public impl::GluonPropagateLayoutPassBase<GluonPropagateLayoutPass> {
public:
  void runOnOperation() override {
    bool failedPropagation = false;
    getOperation().walk([&](triton::FuncOp func) {
      failedPropagation |= failed(propagateLayouts(func));
    });
    if (failedPropagation) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<RequireLayoutPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton::gluon

#undef LDBG
#undef DEBUG_TYPE
