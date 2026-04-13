#include "dicp/Dialect/LinalgExt/Analysis/StagePartition.h"

#include "dicp/Dialect/LinalgExt/Analysis/StageDependencyAnalyzer.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#define DEBUG_TYPE "dicp-stage-partition"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace mlir::dicp;

namespace {

enum class OpClass { CubeAnchor, VectorAnchor, MemOp, NeutralOp };

static bool isCoreOpForMode(Operation *op, bool isMixMode, bool isVectorMode) {
  return isMixMode ? (isMatMulOp(op) || isSIMDLikeOp(op))
                   : (isVectorMode ? isSIMDLikeOp(op) : isMatMulOp(op));
}

static OpClass classifyOp(Operation *op) {
  if (isMatMulOp(op))
    return OpClass::CubeAnchor;
  if (isSIMDLikeOp(op))
    return OpClass::VectorAnchor;
  if (isWriteOp(op) || isa<memref::AllocOp>(op))
    return OpClass::MemOp;
  return OpClass::NeutralOp;
}

} // namespace

FailureOr<SmallVector<StagePartitionResult, 2>>
StagePartition::run(RewriterBase &rewriter) {
  LDBG("=== Starting StagePartition ===");
  LDBG("Options: reorderEnabled="
       << (options.reorderEnabled ? "true" : "false")
       << ", tileAllBlocks=" << (options.tileAllBlocks ? "true" : "false"));

  SmallVector<Block *> targetBlocks;
  if (failed(collectTargetBlocks(targetBlocks))) {
    LDBG("No target blocks found for stage partitioning.");
    return SmallVector<StagePartitionResult, 2>{};
  }

  SmallVector<StagePartitionResult, 2> allBlockResults;
  SmallVector<StageInfo, 4> finalStagesAllBlocks;

  for (Block *block : targetBlocks) {
    StagePartitionResult result;
    if (failed(partitionBlock(block, result))) {
      LDBG("Partitioning failed for block: " << block);
      return failure();
    }

    auto scheduledStagesOrErr = scheduleBlock(result, rewriter);
    if (failed(scheduledStagesOrErr)) {
      if (options.reorderEnabled) {
        LDBG("Scheduling/Reordering failed. Pipeline aborting.");
        return failure();
      }

      LDBG("Scheduling failed but reorder is disabled. Proceeding with "
           "original partitions.");
    }

    ArrayRef<StageInfo> finalStages =
        succeeded(scheduledStagesOrErr) ? *scheduledStagesOrErr : result.stages;
    if (succeeded(scheduledStagesOrErr))
      result.stages = *scheduledStagesOrErr;

    allBlockResults.push_back(std::move(result));
    finalStagesAllBlocks.append(finalStages.begin(), finalStages.end());
  }

  if (allBlockResults.size() > 1)
    renumberStageIds(allBlockResults);

  LDBG("\n=== Final Stage Summary ===");
  for (const auto &stage : finalStagesAllBlocks) {
    LDBG("  ID " << stage.id << " | Type "
                 << (stage.type == StageType::Cube ? "Cube" : "Vector")
                 << " | SubStages " << stage.subStages.size() << " | Ops "
                 << stage.getTotalOpCount() << " | Level " << stage.level);
  }
  LDBG("=== Pipeline Completed Successfully ===\n");

  return allBlockResults;
}

LogicalResult
StagePartition::collectTargetBlocks(SmallVectorImpl<Block *> &blocks) {
  if (options.tileAllBlocks)
    return collectAllTargetBlocks(blocks);

  SetVector<Block *> preAssignedBlocks;
  funcOp.walk([&](Operation *op) {
    if (op->hasAttr(stage_attrs::kNPUStageAttrName))
      preAssignedBlocks.insert(op->getBlock());
  });

  if (!preAssignedBlocks.empty()) {
    LDBG("Found operations with pre-assigned stage attributes.");
    for (Block *b : preAssignedBlocks) {
      bool hasAncestor = false;
      for (Operation *parentOp = b->getParentOp(); parentOp != nullptr;
           parentOp = parentOp->getParentOp()) {
        if (Block *ancestorBlock = parentOp->getBlock();
            ancestorBlock && preAssignedBlocks.contains(ancestorBlock)) {
          hasAncestor = true;
          break;
        }
      }
      if (!hasAncestor)
        blocks.push_back(b);
    }

    LDBG("Selected " << blocks.size()
                     << " outermost block(s) for PreAssigned-mode.");
    return success();
  }

  SetVector<Block *> syncBlocks;
  funcOp.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<hivm::SyncBlockWaitOp, hivm::SyncBlockSetOp>(
            [&](auto) { syncBlocks.insert(op->getBlock()); });
  });

  if (!syncBlocks.empty()) {
    blocks.assign(syncBlocks.begin(), syncBlocks.end());
    LDBG("Found " << blocks.size() << " blocks for Sync-mode partitioning.");
    return success();
  }

  StringRef mixMode = "mix";
  if (auto attr = funcOp->getAttrOfType<StringAttr>("mix_mode"))
    mixMode = attr.getValue();

  bool isMix = mixMode == "mix";
  bool isAIV = mixMode == "aiv";

  DenseSet<Block *> ineligibleBlocks;
  for (Block &block : funcOp.getBody()) {
    StageInfo tmpStage;
    tmpStage.id = -1;
    SubStage tmpSubStage;
    for (Operation &op : block)
      tmpSubStage.ops.push_back(&op);
    tmpStage.subStages.push_back(std::move(tmpSubStage));

    if (!isStageEligibleForTiling(tmpStage)) {
      LDBG("[CubeVector] Block excluded by eligibility check");
      ineligibleBlocks.insert(&block);
    }
  }

  DenseMap<Block *, unsigned> counts;
  funcOp.walk([&](Operation *op) {
    if (ineligibleBlocks.contains(op->getBlock()))
      return;

    if (isCoreOpForMode(op, isMix, isAIV))
      ++counts[op->getBlock()];
  });

  SmallVector<std::pair<Block *, unsigned>> ranked(counts.begin(),
                                                   counts.end());
  llvm::sort(ranked, [](const auto &lhs, const auto &rhs) {
    return lhs.second > rhs.second;
  });

  if (!ranked.empty()) {
    blocks.push_back(ranked.front().first);
    LDBG("  [CubeVector] Collected densest block with " << ranked.front().second
                                                        << " core ops.");
    return success();
  }

  return failure();
}

LogicalResult
StagePartition::collectAllTargetBlocks(SmallVectorImpl<Block *> &blocks) {
  LDBG("=== collectAllTargetBlocks (unified mode) ===");

  StringRef mixMode = "mix";
  if (auto attr = funcOp->getAttrOfType<StringAttr>("mix_mode"))
    mixMode = attr.getValue();

  bool isMix = mixMode == "mix";
  bool isAIV = mixMode == "aiv";
  DenseMap<Block *, unsigned> coreOpCounts;

  funcOp.walk([&](Operation *op) {
    Block *block = op->getBlock();

    if (op->hasAttr(stage_attrs::kNPUStageAttrName) ||
        isa<hivm::SyncBlockWaitOp, hivm::SyncBlockSetOp>(op)) {
      coreOpCounts[block];
      return;
    }

    if (isCoreOpForMode(op, isMix, isAIV))
      ++coreOpCounts[block];
  });

  if (coreOpCounts.empty()) {
    LDBG("No target blocks found.");
    return failure();
  }

  SmallVector<std::pair<Block *, unsigned>> ranked(coreOpCounts.begin(),
                                                   coreOpCounts.end());
  llvm::sort(ranked, [](const auto &lhs, const auto &rhs) {
    return lhs.second > rhs.second;
  });

  for (auto &[block, count] : ranked) {
    blocks.push_back(block);
    LDBG("  Collected block with " << count << " core ops.");
  }

  LDBG("Total: " << blocks.size() << " target block(s).");
  return success();
}

void StagePartition::renumberStageIds(
    SmallVectorImpl<StagePartitionResult> &results) {
  LDBG("=== Global Stage ID Renumbering ===");
  int nextId = 0;

  for (auto &result : results) {
    DenseMap<int, int> remap;

    for (auto &stage : result.stages) {
      if (!remap.count(stage.id))
        remap[stage.id] = nextId++;
    }

    for (auto &stage : result.stages) {
      int oldId = stage.id;
      stage.id = remap[oldId];
      LDBG("  Renumber: " << oldId << " -> " << stage.id);

      for (auto &subStage : stage.subStages)
        subStage.stageId = stage.id;

      auto remapSet = [&](DenseSet<int> &set) {
        DenseSet<int> remapped;
        for (int stageId : set) {
          auto it = remap.find(stageId);
          remapped.insert(it != remap.end() ? it->second : stageId);
        }
        set = std::move(remapped);
      };
      remapSet(stage.preds);
      remapSet(stage.succs);
    }

    for (auto &entry : result.opToStageId) {
      auto it = remap.find(entry.second);
      if (it != remap.end())
        entry.second = it->second;
    }
  }
}

LogicalResult StagePartition::partitionBlock(Block *block,
                                             StagePartitionResult &result) {
  bool recoverMode = false;
  bool hasSync = false;

  for (Operation &op : *block) {
    recoverMode |= op.hasAttr(stage_attrs::kNPUStageAttrName);
    hasSync |= isa<hivm::SyncBlockWaitOp, hivm::SyncBlockSetOp>(&op);
  }

  LogicalResult status = failure();
  if (recoverMode) {
    LDBG("[StagePartition] Selecting PreAssigned-mode for block: " << block);
    status = partitionRecoverMode(block, result);
  } else if (hasSync) {
    LDBG("[StagePartition] Selecting Sync-mode for block: " << block);
    status = partitionSyncMode(block, result);
  } else {
    LDBG("[StagePartition] Selecting CubeVector-mode for block: " << block);
    status = partitionCubeVectorMode(block, result);
  }

  if (succeeded(status)) {
    LDBG("[StagePartition] Successfully partitioned block.");
    for (const auto &stage : result.stages) {
      LDBG(
          "  -> Stage " << stage.id << " | type="
                        << (stage.type == StageType::Vector ? "Vector" : "Cube")
                        << " | substages=" << stage.subStages.size()
                        << " | ops=" << stage.getTotalOpCount());
    }
  } else {
    LDBG("[StagePartition] Failed to partition block.");
  }
  return status;
}

LogicalResult
StagePartition::partitionRecoverMode(Block *block,
                                     StagePartitionResult &result) {
  result.block = block;
  result.containsSync = false;

  std::map<int, SmallVector<Operation *, 8>> stageOpsMap;
  for (Operation &op : *block) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;

    if (auto attr =
            op.getAttrOfType<IntegerAttr>(stage_attrs::kNPUStageAttrName)) {
      int stageId = attr.getInt();
      stageOpsMap[stageId].push_back(&op);
      result.opToStageId[&op] = stageId;
      if (isa<hivm::SyncBlockWaitOp, hivm::SyncBlockSetOp>(&op))
        result.containsSync = true;
      continue;
    }

    LDBG("Skipping operation without pre-assigned stage ID: " << op.getName());
  }

  if (stageOpsMap.empty()) {
    LDBG("No valid pre-assigned stage operations found in the block.");
    return failure();
  }

  for (auto &[stageId, ops] : stageOpsMap) {
    StageInfo stage;
    stage.id = stageId;

    bool isCube = false;
    for (Operation *op : ops) {
      op->walk([&](Operation *innerOp) {
        if (!isCube && classifyOp(innerOp) == OpClass::CubeAnchor)
          isCube = true;
      });
      if (isCube)
        break;
    }

    stage.type = isCube ? StageType::Cube : StageType::Vector;
    stage.hasSync = llvm::any_of(ops, [](Operation *op) {
      return isa<hivm::SyncBlockWaitOp, hivm::SyncBlockSetOp>(op);
    });

    SubStage subStage;
    subStage.index = 0;
    subStage.stageId = stageId;
    subStage.ops = std::move(ops);
    stage.subStages.push_back(std::move(subStage));
    result.stages.push_back(std::move(stage));
  }

  return success();
}

LogicalResult StagePartition::partitionSyncMode(Block *block,
                                                StagePartitionResult &result) {
  result.block = block;
  result.containsSync = true;

  auto stagesOrFailure = StageDependencyAnalyzer::collectStages(block);
  if (failed(stagesOrFailure))
    return failure();

  result.stages = std::move(*stagesOrFailure);
  llvm::erase_if(result.stages, [](const StageInfo &stage) {
    if (stage.type == StageType::Cube)
      return false;
    if (!isStageEligibleForTiling(stage)) {
      LDBG("[partitionSyncMode] Dropping ineligible Vector stage " << stage.id);
      return true;
    }
    return false;
  });

  for (const auto &stage : result.stages) {
    for (Operation *op : stage.getOps())
      result.opToStageId[op] = stage.id;
  }
  return success();
}

LogicalResult
StagePartition::partitionCubeVectorMode(Block *block,
                                        StagePartitionResult &result) {
  result.block = block;
  result.containsSync = false;

  DenseMap<Operation *, int> anchorIDs;
  DenseMap<int, StageType> stageTypeOfId;
  int currentAnchorId = 0;
  std::optional<StageType> currentType;

  for (Operation &op : *block) {
    OpClass opClass = classifyOp(&op);
    if (opClass != OpClass::CubeAnchor && opClass != OpClass::VectorAnchor)
      continue;

    StageType stageType =
        opClass == OpClass::CubeAnchor ? StageType::Cube : StageType::Vector;
    if (!currentType || *currentType != stageType) {
      currentAnchorId++;
      currentType = stageType;
      stageTypeOfId[currentAnchorId] = stageType;
    }
    anchorIDs[&op] = currentAnchorId;
  }

  if (currentAnchorId == 0) {
    StageInfo stage;
    stage.id = 1;
    stage.type = StageType::Vector;

    SubStage subStage;
    subStage.index = 0;
    subStage.stageId = 1;
    for (Operation &op : *block) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;
      subStage.ops.push_back(&op);
      result.opToStageId[&op] = 1;
    }

    if (!subStage.empty()) {
      stage.subStages.push_back(std::move(subStage));
      result.stages.push_back(std::move(stage));
    }
    return success();
  }

  SmallVector<Operation *, 32> blockOps;
  std::vector<int> logicalStages;
  int lastAssignedId = 1;

  for (Operation &op : *block) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;

    blockOps.push_back(&op);
    if (anchorIDs.count(&op)) {
      logicalStages.push_back(anchorIDs[&op]);
      lastAssignedId = anchorIDs[&op];
      continue;
    }

    int foundId = -1;
    SetVector<Operation *> backwardSlice;
    (void)getBackwardSlice(&op, &backwardSlice);
    for (Operation *dep : llvm::reverse(backwardSlice)) {
      if (dep->getBlock() == block && anchorIDs.count(dep)) {
        foundId = anchorIDs[dep];
        break;
      }
    }

    if (foundId == -1) {
      SetVector<Operation *> forwardSlice;
      getForwardSlice(&op, &forwardSlice);
      for (Operation *dep : forwardSlice) {
        if (dep->getBlock() == block && anchorIDs.count(dep)) {
          foundId = anchorIDs[dep];
          break;
        }
      }
    }

    logicalStages.push_back(foundId != -1 ? foundId : lastAssignedId);
    if (foundId != -1)
      lastAssignedId = foundId;
  }

  size_t numOps = blockOps.size();
  std::vector<int> finalStages(numOps);
  if (numOps > 0) {
    finalStages[numOps - 1] = logicalStages[numOps - 1];
    for (int i = static_cast<int>(numOps) - 2; i >= 0; --i)
      finalStages[i] = std::min(logicalStages[i], finalStages[i + 1]);
  }

  for (size_t i = 0; i < numOps; ++i) {
    int stageId = finalStages[i];
    if (result.stages.empty() || result.stages.back().id != stageId) {
      StageInfo stage;
      stage.id = stageId;
      stage.type = stageTypeOfId.lookup(stageId);

      SubStage subStage;
      subStage.index = 0;
      subStage.stageId = stageId;
      stage.subStages.push_back(std::move(subStage));
      result.stages.push_back(std::move(stage));
    }

    StageInfo &currentStageInfo = result.stages.back();
    SubStage &currentSubStage = currentStageInfo.subStages.back();
    Operation *op = blockOps[i];
    currentSubStage.ops.push_back(op);
    result.opToStageId[op] = stageId;

    if (isa<hivm::SyncBlockWaitOp, hivm::SyncBlockSetOp>(op)) {
      currentStageInfo.hasSync = true;
      result.containsSync = true;
    }

    if (!isStructuredControlFlowOp(op))
      continue;

    bool hasRemainingOpsInStage = false;
    for (size_t j = i + 1; j < numOps; ++j) {
      if (finalStages[j] == stageId) {
        hasRemainingOpsInStage = true;
        break;
      }
    }

    if (!hasRemainingOpsInStage) {
      LDBG("  [ControlFlowBoundary] No remaining ops in Stage "
           << stageId << ", skipping SubStage split after: " << *op);
      continue;
    }

    SubStage nextSubStage;
    nextSubStage.index = currentStageInfo.subStages.size();
    nextSubStage.stageId = stageId;
    currentStageInfo.subStages.push_back(std::move(nextSubStage));
    LDBG("  [ControlFlowBoundary] Created SubStage "
         << currentStageInfo.subStages.back().index
         << " after control flow op in Stage " << stageId << ": " << *op);
  }

  return success();
}

FailureOr<SmallVector<StageInfo, 4>>
StagePartition::scheduleBlock(StagePartitionResult &result,
                              RewriterBase &rewriter) {
  (void)rewriter;
  if (!options.reorderEnabled)
    return result.stages;

  AliasAnalysis aliasAnalysis(result.block->getParentOp());
  StageDependencyAnalyzer analyzer(result.block, aliasAnalysis);
  auto reorderedOrErr =
      analyzer.analyzeAndReorderStages(std::move(result.stages));
  if (failed(reorderedOrErr)) {
    LDBG("StageDependencyAnalyzer encountered a cycle or failure during "
         "reorder.");
    return failure();
  }

  return reorderedOrErr.value();
}
