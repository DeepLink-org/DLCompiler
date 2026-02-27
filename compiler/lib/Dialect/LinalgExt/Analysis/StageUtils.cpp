#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"
#include "dicp/Dialect/LinalgExt/Analysis/StageDependencyAnalyzer.h"
#include "dicp/TransformOps/Transforms.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

// External dialect dependency - assuming HIVM dialect is available
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#define DEBUG_TYPE "dicp-stage-utils"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace mlir::dicp;
using namespace mlir::dicp::LinalgExt;

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

bool mlir::dicp::LinalgExt::isOpInStage(Operation *op, int stageId) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(kNPUStageAttrName)) {
    return attr.getInt() == stageId;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// StagePartitioner Implementation
//===----------------------------------------------------------------------===//

SmallVector<Block *>
StagePartitioner::findBlocksWithHivmSyncOps(ModuleOp module) {
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
                             "] Searching for blocks with HIVM sync ops...\n");

  SetVector<Block *> blockSet;
  module.walk([&](Operation *op) {
    // Using TypeSwitch for extensible sync op detection
    llvm::TypeSwitch<Operation *>(op)
        .Case<hivm::SyncBlockWaitOp, hivm::SyncBlockSetOp>(
            [&](auto) { blockSet.insert(op->getBlock()); });
  });

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Found " << blockSet.size()
                          << " blocks requiring stage partitioning.\n");
  return blockSet.takeVector();
}

LogicalResult StagePartitioner::analyzeAndTagBlock(Block *block,
                                                   MLIRContext *ctx,
                                                   bool &anyStageFound) {
  Operation *parentOp = block->getParentOp();
  if (!parentOp) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] Error: Block has no parent operation.\n");
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
                             "] Analyzing dependencies for block in parent: "
                          << parentOp->getName() << "\n");

  // Perform Alias Analysis to ensure conservative dependency tracking
  AliasAnalysis aliasAnalysis(parentOp);
  StageDependencyAnalyzer analyzer(block, aliasAnalysis);

  FailureOr<std::vector<StageInfo>> stagesOrErr = analyzer.collectStages();
  if (failed(stagesOrErr)) {
    return parentOp->emitError(
        "Dependency cycle or analysis failure detected in stage analysis.");
  }

  anyStageFound = false;
  unsigned tagCount = 0;

  for (const auto &stage : *stagesOrErr) {
    // Only tag operations belonging to Vector-type stages
    if (stage.type == StageType::Vector) {
      IntegerAttr attr = IntegerAttr::get(IntegerType::get(ctx, 32), stage.id);
      for (Operation *op : stage.ops) {
        op->setAttr(kNPUStageAttrName, attr);
        tagCount++;
      }
      anyStageFound = true;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Successfully tagged " << tagCount
                          << " operations across stages.\n");
  return success();
}

SetVector<int> StagePartitioner::getStageIdsInBlock(Block *block) {
  SetVector<int> ids;
  if (!block)
    return ids;

  for (Operation &op : *block) {
    if (auto attr = op.getAttrOfType<IntegerAttr>(kNPUStageAttrName)) {
      ids.insert(static_cast<int>(attr.getInt()));
    }
  }
  return ids;
}

Stage StagePartitioner::partition(Block *block, int stageId) {
  Stage stage;
  stage.id = stageId;

  SmallVector<Operation *, 16> currentOps;
  unsigned subIdx = 0;

  LLVM_DEBUG(
      llvm::dbgs() << "[" DEBUG_TYPE
                      "] Partitioning block into substages for Stage ID: "
                   << stageId << "\n");

  auto finalizeSubStage = [&](SmallVectorImpl<Operation *> &ops) {
    if (ops.empty())
      return;

    SubStage subStage;
    subStage.index = subIdx++;
    subStage.stageId = stageId;
    subStage.ops = std::move(ops);
    stage.subStages.push_back(std::move(subStage));
    ops.clear();

    LLVM_DEBUG(llvm::dbgs()
               << "  -> Created SubStage " << subStage.index << " with "
               << stage.subStages.back().ops.size() << " ops.\n");
  };

  for (Operation &op : *block) {
    // Filter by stage ID
    if (!isOpInStage(&op, stageId))
      continue;

    // Synchronization operations act as hard boundaries for sub-stages.
    // This prevents hoisting/sinking across sync points during later
    // scheduling.
    bool isSync = isa<hivm::SyncBlockWaitOp, hivm::SyncBlockSetOp>(op);

    if (isSync) {
      finalizeSubStage(currentOps);
      // Note: Sync ops themselves are currently excluded from substage op lists
      // as they serve as delimiters.
      continue;
    }

    currentOps.push_back(&op);
  }

  // Handle trailing operations
  finalizeSubStage(currentOps);

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Stage " << stageId
                          << " partitioned into " << stage.subStages.size()
                          << " substages.\n");
  return stage;
}

SmallVector<Stage, 4> StagePartitioner::getAllStagesInBlock(Block *block) {
  SmallVector<Stage, 4> result;
  SetVector<int> stageIds = getStageIdsInBlock(block);

  for (int id : stageIds) {
    result.push_back(partition(block, id));
  }

  return result;
}

namespace {

//===----------------------------------------------------------------------===//
// Helper Utilities
//===----------------------------------------------------------------------===//

static bool isMatMul(Operation *op) {
  return isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op);
}

static bool isCubeSatge(Operation *op) {
  return isa<memref::MemRefDialect, tensor::TensorDialect,
             bufferization::BufferizationDialect>(op->getDialect());
}

static bool isVectorOp(Operation *op) {
  if (!isa<arith::ArithDialect, math::MathDialect>(op->getDialect())) {
    return false;
  }
  return llvm::any_of(op->getOperandTypes(),
                      [](Type t) { return isa<ShapedType>(t); });
}

//===----------------------------------------------------------------------===//
// BlockSegmenter
//===----------------------------------------------------------------------===//

class BlockSegmenter {
public:
  explicit BlockSegmenter(Block &block) : block(block) {}

  LogicalResult run(SmallVectorImpl<Stage> &stages);

private:
  struct OpInfo {
    StageType type = StageType::Vector;
    int64_t index = -1;
    int64_t closestDistance = std::numeric_limits<int64_t>::max();
  };

  LogicalResult validate();
  void indexOperations();
  void classify();
  void buildStages(SmallVectorImpl<Stage> &stages);
  void deduplicate(SmallVectorImpl<Stage> &stages);

  // Classification Helpers
  void processAnchor(Operation *anchor);
  void propagateSlice(Operation *anchor, bool backward);
  void tryClaim(Operation *op, int64_t anchorIndex);

  Block &block;
  DenseMap<Operation *, OpInfo> opInfoMap;
};

//===----------------------------------------------------------------------===//
// Validation Logic
//===----------------------------------------------------------------------===//

LogicalResult BlockSegmenter::validate() {
  LDBG("Validating block constraints...");

  for (Operation &op : block) {
    auto walkRes = op.walk([&](Operation *nested) -> WalkResult {
      auto loop = dyn_cast<LoopLikeOpInterface>(nested);
      if (!loop)
        return WalkResult::advance();

      bool hasMatmul = false;
      bool hasForbidden = false;

      loop->walk([&](Operation *inner) {
        if (inner == loop)
          return;
        if (isMatMul(inner))
          hasMatmul = true;
        else if (isVectorOp(inner))
          hasForbidden = true;
      });

      if (hasMatmul && hasForbidden) {
        LDBG("ERROR: Loop mixes matmul and forbidden vector ops:\n" << *loop);
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (walkRes.wasInterrupted())
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Core Pipeline Implementation
//===----------------------------------------------------------------------===//

void BlockSegmenter::indexOperations() {
  LDBG("Indexing operations in block...");
  int64_t idx = 0;
  for (Operation &op : block) {
    auto &info = opInfoMap[&op];
    info.index = idx++;
    LDBG("  Indexed op #" << info.index << " : " << op);
  }
}

void BlockSegmenter::classify() {
  LDBG("Classifying Cube vs Vector operations...");
  for (Operation &op : block) {
    if (isMatMul(&op)) {
      LDBG("Found Cube anchor (matmul): " << op);
      processAnchor(&op);
    }
  }
}

void BlockSegmenter::processAnchor(Operation *anchor) {
  OpInfo &info = opInfoMap[anchor];
  info.type = StageType::Cube;
  info.closestDistance = 0;

  LDBG("  Anchor forced to Cube stage");

  // Propagate Cube type through dialect-specific slices
  propagateSlice(anchor, /*backward=*/true);
  propagateSlice(anchor, /*backward=*/false);
}

void BlockSegmenter::propagateSlice(Operation *anchor, bool backward) {
  SetVector<Operation *> slice;
  auto filter = [&](Operation *op) {
    return op->getBlock() == &block && isCubeSatge(op);
  };

  if (backward) {
    BackwardSliceOptions opt;
    opt.filter = filter;
    (void)getBackwardSlice(anchor, &slice, opt);
    LDBG("  Backward slice size: " << slice.size());
  } else {
    ForwardSliceOptions opt;
    opt.filter = filter;
    getForwardSlice(anchor, &slice, opt);
    LDBG("  Forward slice size: " << slice.size());
  }

  int64_t anchorIdx = opInfoMap[anchor].index;
  for (Operation *op : slice)
    tryClaim(op, anchorIdx);
}

void BlockSegmenter::tryClaim(Operation *op, int64_t anchorIndex) {
  OpInfo &info = opInfoMap[op];
  int64_t dist = std::abs(info.index - anchorIndex);

  if (dist < info.closestDistance) {
    LDBG("    Claiming op " << op->getName() << " as Cube (dist = " << dist
                            << ")");
    info.type = StageType::Cube;
    info.closestDistance = dist;
  } else {
    LDBG("    Skipping op " << op->getName() << " (closer anchor exists)");
  }
}

void BlockSegmenter::buildStages(SmallVectorImpl<Stage> &stages) {
  LDBG("Building execution stages...");
  int stageId = 0;

  auto ensureStage = [&](StageType type) {
    if (stages.empty() || stages.back().type != type) {
      stages.emplace_back(stageId++, type);
      LDBG("  Created new stage "
           << stages.back().id
           << " type = " << (type == StageType::Cube ? "Cube" : "Vector"));
    }
  };

  for (Operation &op : block) {
    OpInfo &info = opInfoMap[&op];

    // Final sanity check for MatMul classification
    if (isMatMul(&op) && info.type != StageType::Cube) {
      LDBG("  WARNING: Matmul not classified as Cube. Forcing.");
      info.type = StageType::Cube;
    }

    ensureStage(info.type);
    stages.back().addOp(&op);
    LDBG("  Added op " << op << " to stage " << stages.back().id);
  }
}

void BlockSegmenter::deduplicate(SmallVectorImpl<Stage> &stages) {
  LDBG("Running cross-stage deduplication...");

  // Track which stage (index in 'stages' vector) owns which operation
  DenseMap<Operation *, int> ownerStage;

  for (int i = 0, e = stages.size(); i < e; ++i) {
    for (auto &sub : stages[i].subStages) {
      for (Operation *op : sub.ops) {
        if (!ownerStage.count(op)) {
          ownerStage[op] = i;
        } else {
          int prevOwner = ownerStage[op];
          // Use the distance metric to resolve ownership if duplicated
          if (opInfoMap[op].closestDistance < opInfoMap[op].closestDistance) {
            ownerStage[op] = i;
          }
        }
      }
    }
  }

  // Filter stages based on ownership
  for (int i = 0, e = stages.size(); i < e; ++i) {
    for (auto &sub : stages[i].subStages) {
      SmallVector<Operation *, 8> filtered;
      for (Operation *op : sub.ops) {
        if (ownerStage.lookup(op) == i) {
          filtered.push_back(op);
        } else {
          LDBG("  Removing duplicated op " << op->getName() << " from stage "
                                           << i);
        }
      }
      sub.ops.swap(filtered);
    }
  }

  // Clean up empty stages
  llvm::erase_if(stages, [](const Stage &s) {
    bool isEmpty =
        llvm::all_of(s.subStages, [](auto &sub) { return sub.ops.empty(); });
    if (isEmpty)
      LDBG("  Removing empty stage " << s.id);
    return isEmpty;
  });

  LDBG("Deduplication complete. Final stage count = " << stages.size());
}

LogicalResult BlockSegmenter::run(SmallVectorImpl<Stage> &stages) {
  LDBG("=== Starting Cube/Vector Segmentation (Block-level) ===");

  if (failed(validate()))
    return failure();

  indexOperations();
  classify();
  buildStages(stages);
  deduplicate(stages);

  LDBG("=== Segmentation Finished. Total stages = " << stages.size() << " ===");
  return success();
}

//===----------------------------------------------------------------------===//
// Target Block Search Helpers
//===----------------------------------------------------------------------===//

/// Calculates the nesting depth of a block.
/// FuncBody = 0, scf.for loop body = 1, nested scf.for = 2, etc.
static unsigned getBlockNestingLevel(Block *block) {
  unsigned level = 0;
  Region *region = block->getParent();
  while (region) {
    Operation *parentOp = region->getParentOp();
    if (!parentOp)
      break;

    // If the parent operation is a Function, we consider this the top level
    // (0).
    if (isa<func::FuncOp>(parentOp))
      return level;

    level++;
    region = parentOp->getParentRegion();
  }
  return level;
}

} // namespace
//===----------------------------------------------------------------------===//
// CubeVectorSplitter Public API
//===----------------------------------------------------------------------===//


/**
 * Identifies the primary IR Block to be processed based on the function's 
 * "mix_mode" attribute.
 * * Selection Strategies:
 * - "mix": Returns the Block with the highest density of anchor operations 
 * (MatMul or Vector ops).
 * - "aiv"/"aic": Returns the outermost Block (minimum nesting) containing 
 * the mode-specific operations.
 */
Block *CubeVectorSplitter::findTargetBlock(func::FuncOp funcOp) {
  // 1. Extract configuration with StringRef to avoid allocations.
  StringRef mixMode = "mix";
  if (auto attr = funcOp->getAttrOfType<StringAttr>("mix_mode"))
    mixMode = attr.getValue();

  // Define predicates for operation filtering.
  auto isCube = [](Operation *op) { return isMatMul(op); };
  auto isVector = [](Operation *op) { return isVectorOp(op); };

  // 2. State tracking using LLVM-optimized containers.
  DenseMap<Block *, unsigned> anchorCounts;
  Block *bestBlock = nullptr;
  unsigned maxAnchors = 0;
  unsigned minLevel = std::numeric_limits<unsigned>::max();

  bool isMixMode = (mixMode == "mix");
  bool isAIV = (mixMode == "aiv");

  // 3. Single-pass IR traversal.
  funcOp.walk([&](Operation *op) {
    // Determine if the op is relevant to the current mode.
    bool match = isMixMode ? (isCube(op) || isVector(op))
                           : (isAIV ? isVector(op) : isCube(op));
    if (!match)
      return;

    Block *currentBlock = op->getBlock();

    if (isMixMode) {
      // Strategy: Maximize operation density.
      unsigned count = ++anchorCounts[currentBlock];
      if (count > maxAnchors) {
        maxAnchors = count;
        bestBlock = currentBlock;
      }
    } else {
      // Strategy: Find the outermost scope (minimum nesting depth).
      unsigned currentLevel = getBlockNestingLevel(currentBlock);
      if (currentLevel < minLevel) {
        minLevel = currentLevel;
        bestBlock = currentBlock;
      }
    }
  });

  // 4. Diagnostics and Validation.
  if (!bestBlock) {
    funcOp.emitError() << "Target block discovery failed for mode: '" 
                       << mixMode << "'. No matching operations found.";
    return nullptr;
  }

  // Optional: Warn if multiple blocks exist (Mix Mode), selecting the first found.
  if (isMixMode && anchorCounts.size() > 1) {
    llvm::dbgs() << "[Warning] Multiple candidate blocks found in mix mode. "
                 << "Selecting block with " << maxAnchors << " anchors.\n";
  }
  bestBlock->dump();
  return bestBlock;
}

//===----------------------------------------------------------------------===//
// Public API Implementation
//===----------------------------------------------------------------------===//

LogicalResult CubeVectorSplitter::splitBlock(Block &block,
                                             SmallVectorImpl<Stage> &stages) {
  BlockSegmenter segmenter(block);
  return segmenter.run(stages);
}