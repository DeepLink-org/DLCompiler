#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"
#include "dicp/TransformOps/DicpTransformOps.h"
#include "dicp/TransformOps/Transforms.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include <optional>

#define DEBUG_TYPE "npu-loop-fusion"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace dicp;
using namespace LinalgExt;

namespace mlir::dicp::LinalgExt {
#define GEN_PASS_DEF_FUSELOOP
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace mlir::dicp::LinalgExt

namespace {

static constexpr llvm::StringRef kFuseLoopTagAttr = "fuse_loop_tag";

/// Helper struct to hold information about a group of loops to be fused.
struct FusionGroup {
  int32_t stageId;
  SmallVector<StringAttr> loopTags;
};

//===----------------------------------------------------------------------===//
// Dependency Analysis Utilities
//===----------------------------------------------------------------------===//

/// Checks if `opB` can be safely moved UP to `opA`'s position.
/// This requires that all values used by `opB` (operands and captured values)
/// are defined by operations that properly dominate `opA`.
static bool canMoveUpTo(Operation *opA, Operation *opB,
                        DominanceInfo &domInfo) {
  auto isSafeOperand = [&](Value val) {
    Operation *defOp = val.getDefiningOp();
    if (!defOp)
      return true; // Block arguments inherently dominate everything in the
                   // block.
    bool dominates =
        domInfo.properlyDominates(defOp, opA, /*enclosingOpOk=*/false);

    if (!dominates) {
      LDBG("Cannot move Up: "
           << *defOp << " does NOT dominate target position " << *opA);
    }
    return dominates;
  };

  // Check explicit operands.
  if (!llvm::all_of(opB->getOperands(), isSafeOperand))
    return false;

  // Check values implicitly captured in regions.
  bool regionsSafe = true;
  visitUsedValuesDefinedAbove(opB->getRegions(), [&](OpOperand *operand) {
    if (regionsSafe && !isSafeOperand(operand->get()))
      regionsSafe = false;
  });

  return regionsSafe;
}

/// Checks if `opA` can be safely moved DOWN to `opB`'s position.
/// This requires that all users of `opA`'s results are strictly after `opB`.
static bool canMoveDownTo(Operation *opA, Operation *opB,
                          DominanceInfo &domInfo) {
  for (Operation *user : opA->getUsers()) {
    if (!domInfo.properlyDominates(opB, user, /*enclosingOpOk=*/false)) {
      LDBG("Cannot move Down: User "
           << *user << " appears before the target move-to position " << *opB);
      return false;
    }
  }
  return true;
}

/// Checks if two operations in the same block can be safely moved to be
/// adjacent without violating SSA data dependencies.
static bool canBeMadeAdjacent(Operation *opA, Operation *opB,
                              DominanceInfo &domInfo) {
  if (opA == opB || opA->getBlock() != opB->getBlock())
    return false;

  // Enforce structural order (opA before opB) for simpler reasoning.
  if (opA->isBeforeInBlock(opB)) {
    if (canMoveDownTo(opA, opB, domInfo))
      return true;

  } else {
    if (canMoveUpTo(opA, opB, domInfo))
      return true;
  }

  LDBG("    [Reject] Ops "
       << *opA << " and " << *opB
       << " cannot be made adjacent due to SSA dependencies.");
  return false;
}

/// Identifies candidate loops within a SubStage and clusters them into
/// FusionGroups based on dependency analysis.
static SmallVector<FusionGroup> groupAndTagLoops(SubStage &subStage,
                                                 Builder &builder) {
  SmallVector<Operation *> loops;
  for (Operation *op : subStage.ops) {
    if (isa<scf::ForallOp, scf::ForOp>(op))
      loops.push_back(op);
  }

  if (loops.size() < 2)
    return {};

  DominanceInfo domInfo(loops.front()->getParentOp());
  SmallVector<FusionGroup> fusionGroups;

  SmallVector<Operation *> currentOps;
  SmallVector<StringAttr> currentTags;

  // Finalizes the current cluster of compatible loops.
  auto finalizeGroup = [&]() {
    if (currentOps.size() >= 2) {
      fusionGroups.push_back({subStage.stageId, std::move(currentTags)});
    } else if (!currentOps.empty()) {
      // Revert: remove tags from isolated operations.
      for (auto [op, tag] : llvm::zip(currentOps, currentTags))
        op->removeAttr(tag);
    }
    currentOps.clear();
    currentTags.clear();
  };

  int globalLoopIdx = 0;
  for (Operation *loop : loops) {
    if (currentOps.empty()) {
      currentOps.push_back(loop);
    } else {
      // Check N-to-N compatibility for multi-way sibling fusions.
      bool isCompatible = llvm::all_of(currentOps, [&](Operation *existingOp) {
        if (!canBeMadeAdjacent(existingOp, loop, domInfo)) {
          LDBG("  [Break] Dependency conflict between "
               << loop->getName() << " and group member "
               << existingOp->getName());
          return false;
        }
        return true;
      });

      if (isCompatible) {
        currentOps.push_back(loop);
      } else {
        finalizeGroup();
        currentOps.push_back(loop);
      }
    }

    // Attach a unique tag to the loop for the Transform Dialect to match.
    StringAttr tag = builder.getStringAttr(
        llvm::formatv("{0}_{1}_{2}_{3}", kFuseLoopTagAttr, subStage.stageId,
                      subStage.index, globalLoopIdx++)
            .str());
    loop->setAttr(tag, builder.getUnitAttr());
    currentTags.push_back(tag);
  }

  finalizeGroup();
  return fusionGroups;
}

//===----------------------------------------------------------------------===//
// Transform Dialect Command Generation
//===----------------------------------------------------------------------===//

/// Dispatches a MatchOp returning an opaque transform handle for a tagged op.
static Value getMatchHandle(ImplicitLocOpBuilder &b, Value root,
                            StringAttr tag) {
  auto handleType = b.getType<transform::AnyOpType>();
  auto attrDict = b.getDictionaryAttr(b.getNamedAttr(tag, b.getUnitAttr()));

  return b
      .create<transform::MatchOp>(
          handleType, root, /*ops=*/nullptr, /*interface=*/nullptr,
          /*combined_attr=*/attrDict, /*filter_catalogs=*/nullptr,
          /*filter_on_op_names=*/nullptr)
      .getResult();
}

/// Generates transform dialect IR to perform sibling fusion on a group of
/// loops.
static void fuseLoopsByTags(ImplicitLocOpBuilder &b, Value root,
                            const FusionGroup &group) {
  if (group.loopTags.size() < 2)
    return;

  SmallVector<Value> loopHandles;
  loopHandles.reserve(group.loopTags.size());
  for (StringAttr tag : group.loopTags)
    loopHandles.push_back(getMatchHandle(b, root, tag));

  // Sequentially fuse sibling loops into the first loop.
  Value fusedLoop = loopHandles.front();
  auto handleType = b.getType<transform::AnyOpType>();

  for (size_t i = 1; i < loopHandles.size(); ++i) {
    auto fuseOp = b.create<transform::LoopFuseSiblingOp>(
        handleType, /*target=*/fusedLoop, /*source=*/loopHandles[i]);
    fusedLoop = fuseOp.getFusedLoop();
  }

  // Annotate the final fused loop with the hardware Stage ID.
  auto stageAttr = b.getI32IntegerAttr(group.stageId);
  auto paramType = transform::ParamType::get(b.getContext(), b.getI32Type());
  auto stageParam = b.create<transform::ParamConstantOp>(paramType, stageAttr);

  b.create<transform::AnnotateOp>(fusedLoop, kNPUStageAttrName,
                                  stageParam.getResult());
}

//===----------------------------------------------------------------------===//
// Pass Orchestrator
//===----------------------------------------------------------------------===//

struct FuseLoopPass
    : public mlir::dicp::LinalgExt::impl::FuseLoopBase<FuseLoopPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    Builder builder(ctx);

    // 1. Identify synchronization boundaries.
    SmallVector<Block *> blocks =
        StagePartitioner::findBlocksWithHivmSyncOps(module);
    if (blocks.empty()) {
      LDBG("No synchronization barriers detected. Skipping FuseLoopPass.");
      return;
    }

    // 2. Perform staging analysis.
    bool anyStageFound = false;
    for (Block *block : blocks) {
      if (failed(StagePartitioner::analyzeAndTagBlock(block, ctx,
                                                      anyStageFound))) {
        LDBG("Fatal error during staging topology analysis. Aborting.");
        return signalPassFailure();
      }
    }

    if (!anyStageFound)
      return;

    // 3. Group loops within each substage.
    SmallVector<FusionGroup> allFusionGroups;
    for (Block *block : blocks) {
      for (int32_t stageId : StagePartitioner::getStageIdsInBlock(block)) {
        auto subStages = StagePartitioner::partition(block, stageId).subStages;
        for (SubStage &subStage : subStages) {
          SmallVector<FusionGroup> groups = groupAndTagLoops(subStage, builder);
          allFusionGroups.append(groups.begin(), groups.end());
        }
      }
    }

    if (allFusionGroups.empty())
      return;

    // 4. Apply transformations via Transform Dialect.
    TransformApplier::apply(module,
                            [&](OpBuilder &b, Location loc, Value root) {
                              ImplicitLocOpBuilder implicitB(loc, b);
                              for (const FusionGroup &group : allFusionGroups)
                                fuseLoopsByTags(implicitB, root, group);
                            });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::dicp::LinalgExt::createFuseLoopPass() {
  return std::make_unique<FuseLoopPass>();
}