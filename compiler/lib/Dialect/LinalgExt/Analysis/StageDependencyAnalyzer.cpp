#include "dicp/Dialect/LinalgExt/Analysis/StageDependencyAnalyzer.h"
#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#include <numeric>

#define DEBUG_TYPE "stage-dep-analyzer"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace mlir::dicp;

static StringRef getStageTypeStr(StageType type) {
  switch (type) {
  case StageType::Vector:
    return "Vector";
  case StageType::Cube:
    return "Cube";
  }

  llvm_unreachable("unexpected stage type");
}

FailureOr<SmallVector<StageInfo, 4>>
StageDependencyAnalyzer::collectStages(Block *block) {
  LLVM_DEBUG(
      llvm::dbgs() << "\n>>> [StageAnalyzer] Starting analysis for Block "
                   << block << "\n");
  SmallVector<StageInfo, 4> collectedStages;

  // Temporary state holders
  StageInfo currentStage;
  currentStage.id = 0;
  // Use a temporary buffer for ops to facilitate std::move into SubStage later
  SmallVector<Operation *, 8> currentSubStageOps;

  // Commits the current accumulation of operations into a SubStage.
  auto commitSubStage = [&]() {
    if (currentSubStageOps.empty())
      return;

    SubStage subStage;
    subStage.index = currentStage.subStages.size();
    subStage.stageId = currentStage.id;
    // Transfer ownership of the vector data to avoid copying
    subStage.ops = std::move(currentSubStageOps);

    LLVM_DEBUG(llvm::dbgs() << "    -> Created SubStage " << subStage.index
                            << " (Ops: " << subStage.ops.size() << ")\n");

    currentStage.subStages.push_back(std::move(subStage));

    // Reset buffer capacity after move (state is valid but unspecified)
    currentSubStageOps.clear();
  };

  // Commits the current Stage to the final list and resets state.
  auto commitStage = [&]() {
    // Ensure any pending ops are packed into a SubStage first
    commitSubStage();

    if (currentStage.subStages.empty())
      return;

    LLVM_DEBUG(llvm::dbgs()
               << "  => Finalizing Stage " << currentStage.id
               << " (Total SubStages: " << currentStage.subStages.size()
               << ", Type: "
               << (currentStage.type == StageType::Cube ? "Cube" : "Vector")
               << ")\n");

    collectedStages.push_back(std::move(currentStage));

    // Reset StageInfo for the next iteration
    currentStage = StageInfo();
    currentStage.id = collectedStages.size();
    currentStage.type = StageType::Vector; // Reset to default
  };

  // Helper function: Looks ahead in the block to check if there are any compute
  // operations (Cube or SIMD) left in the current stage.
  // We utilize MLIR's built-in intrusive linked list (getNextNode) for
  // zero-cost lookahead.
  auto hasComputeInRemainingStage = [](Operation *startOp) -> bool {
    for (Operation *op = startOp; op != nullptr; op = op->getNextNode()) {
      // A WaitOp indicates a hard barrier and the start of a completely new
      // Stage. We only care about the remainder of the *current* stage.
      if (isa<hivm::SyncBlockWaitOp>(op) ||
          startOp->getBlock() != op->getBlock()) {
        break;
      }
      if (isMatMulOp(op) || isSIMDLikeOp(op)) {
        return true;
      }
    }
    return false;
  };

  // --- Main Analysis Loop ---
  for (Operation &op : block->without_terminator()) {
    bool isWait = isa<hivm::SyncBlockWaitOp>(&op);
    bool isSet = isa<hivm::SyncBlockSetOp>(&op);
    bool isControlFlow = isStructuredControlFlowOp(&op);

    // 1. Check for hard boundaries (WaitOp starts a NEW Stage)
    if (isWait) {
      // SyncBlockWaitOp represents a hard barrier.
      // It forces the completion of the previous Stage.
      LLVM_DEBUG(llvm::dbgs() << "  [Boundary] SyncBlockWaitOp detected: "
                              << op.getName() << "\n");
      commitStage();
      currentStage.hasSync = true;
    }

    // 2. Update Stage properties based on Op attributes
    if (isMatMulOp(&op)) {
      if (currentStage.type != StageType::Cube) {
        LLVM_DEBUG(llvm::dbgs()
                   << "    [Prop] Stage upgraded to Cube type due to: "
                   << op.getName() << "\n");
        currentStage.type = StageType::Cube;
      }
    }

    // 3. Accumulate operation
    // Notice: We push the operation into the buffer FIRST.
    // This ensures that if this op is a SyncBlockSetOp or control flow,
    // it will be included in the current SubStage before we optionally
    // commit/split it.
    currentSubStageOps.push_back(&op);

    // 4. Check for soft boundaries (SetOp or ControlFlow ends CURRENT SubStage)
    if (isSet) {
      LLVM_DEBUG(llvm::dbgs() << "  [Boundary] SyncBlockSetOp detected: "
                              << op.getName() << "\n");
      currentStage.hasSync = true;
    } else if (isControlFlow) {
      LLVM_DEBUG(llvm::dbgs() << "  [Boundary] ControlFlow detected: "
                              << op.getName() << "\n");
    }

    // Handle SubStage split for both SetOp and ControlFlow boundaries
    if (isSet || isControlFlow) {
      // Look ahead to see if creating a new SubStage is actually necessary.
      // If there are no compute ops left in this stage, we avoid creating an
      // empty/trivial SubStage.
      if (hasComputeInRemainingStage(op.getNextNode())) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "    [SubStage Split] Valid compute operations (Cube/SIMD) "
            << "found ahead. Splitting into a new SubStage.\n");

        // Commit the SubStage NOW. Because we already called push_back(&op)
        // above, the boundary op (SetOp or ControlFlow) is safely packed
        // into THIS SubStage.
        commitSubStage();
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "    [SubStage Merge] No compute operations found before "
                   << "the end of the current stage. Skipping SubStage split "
                   << "to optimize pipeline overhead.\n");
      }
    }
  }

  // --- Finalization ---

  // Flush any remaining operations and stages at the end of the block
  commitStage();

  LLVM_DEBUG(
      llvm::dbgs() << "<<< [StageAnalyzer] Analysis Complete. Total Stages: "
                   << collectedStages.size() << "\n\n");

  return collectedStages;
}

void StageDependencyAnalyzer::collectEffects(
    SmallVector<StageInfo, 4> &stages) {
  nodes.clear();
  nodes.resize(stages.size());
  for (auto it : llvm::enumerate(stages)) {
    StageInfo &info = it.value();
    StageNode &node = nodes[it.index()];
    node.stageInfo = &(info);
    for (Operation *op : info.getOps()) {
      // 1. SSA Def-Use Analysis
      for (Value res : op->getResults())
        node.producedValues.insert(res);

      for (Value operand : op->getOperands()) {
        if (auto *defOp = operand.getDefiningOp()) {
          if (defOp->getBlock() == block)
            node.consumedValues.insert(operand);
        }
      }

      // 2. Memory Side Effects Analysis
      if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
        SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
        memEffect.getEffects(effects);
        for (auto &effect : effects) {
          Value val = effect.getValue();
          if (!val)
            continue;
          if (isa<MemoryEffects::Write>(effect.getEffect()))
            node.writeBuffers.insert(val);
          else if (isa<MemoryEffects::Read>(effect.getEffect()))
            node.readBuffers.insert(val);
        }
      } else if (auto matOp =
                     dyn_cast<bufferization::MaterializeInDestinationOp>(op)) {
        node.readBuffers.insert(matOp.getSource());
        node.writeBuffers.insert(matOp.getDest());
      } else if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
        node.readBuffers.insert(copyOp.getSource());
        node.writeBuffers.insert(copyOp.getTarget());
      }
    }
  }
}

void StageDependencyAnalyzer::buildDependencyGraph() {
  LDBG(">>> [Analysis] Building Dependency Graph...");
  for (int i = 0; i < nodes.size(); ++i) {
    for (int j = 0; j < nodes.size(); ++j) {
      if (i == j)
        continue;
      bool hasDependency = false;

      // 1. Check SSA Dependencies (Direct Data Flow)
      // If Stage J consumes a value produced by Stage I, J depends on I.
      for (Value consumed : nodes[j].consumedValues) {
        if (nodes[i].producedValues.count(consumed)) {
          hasDependency = true;
          break;
        }
      }

      // 2. Check Memory Dependencies
      if (!hasDependency) {
        for (Value writeVal : nodes[i].writeBuffers) {
          for (Value readVal : nodes[j].readBuffers) {
            AliasResult result = aliasAnalysis.alias(writeVal, readVal);
            if (result.isMust() || result.isPartial()) {
              hasDependency = true;
              LDBG("  MEM DEPENDENCY: Stage " << i << " -> Stage " << j);
              break;
            }
          }
          if (hasDependency)
            break;
        }
      }

      if (hasDependency) {
        nodes[i].stageInfo->succs.insert(j);
        nodes[j].stageInfo->preds.insert(i);
      }
    }
  }
}

LogicalResult StageDependencyAnalyzer::computeStageLevels() {
  std::vector<int> visitState(nodes.size(),
                              0); // 0: unvisited, 1: visiting, 2: visited
  std::function<LogicalResult(int)> dfs = [&](int u) -> LogicalResult {
    visitState[u] = 1;
    int maxPredLevel = -1;
    for (int v : nodes[u].stageInfo->preds) {
      if (visitState[v] == 1) {
        llvm::errs() << "Error: Cycle detected involving stages " << u
                     << " and " << v << "\n";
        return failure();
      }
      if (visitState[v] == 0) {
        if (failed(dfs(v)))
          return failure();
      }
      if (nodes[v].level > maxPredLevel)
        maxPredLevel = nodes[v].level;
    }
    nodes[u].level = maxPredLevel + 1;
    visitState[u] = 2;
    return success();
  };

  for (int i = 0; i < nodes.size(); ++i) {
    if (visitState[i] == 0) {
      if (failed(dfs(i)))
        return failure();
    }
  }
  return success();
}
FailureOr<SmallVector<StageInfo, 4>>
StageDependencyAnalyzer::analyzeAndReorderStages(
    SmallVector<StageInfo, 4> &&initialStages) {
  SmallVector<StageInfo, 4> workingStages = std::move(initialStages);
  size_t n = workingStages.size();
  if (n == 0)
    return workingStages;

  // 1. Prepare local ephemeral analysis nodes
  collectEffects(workingStages);

  // 2. Analyze dependencies and compute topological levels
  buildDependencyGraph();
  if (failed(computeStageLevels()))
    return failure();

  // 3. Determine new schedule (Order by level, then original ID)
  SmallVector<int, 8> schedule(n);
  std::iota(schedule.begin(), schedule.end(), 0);
  std::stable_sort(schedule.begin(), schedule.end(), [&](int a, int b) {
    if (nodes[a].level != nodes[b].level)
      return nodes[a].level < nodes[b].level;
    return a < b;
  });

  LDBG(">>> Final Logical Schedule:");
  for (int idx : schedule) {
    LDBG("  Stage " << idx << " [Level: " << nodes[idx].level << ", Type: "
                    << getStageTypeStr(workingStages[idx].type) << "]");
  }

  // 4. Physical Materialization: Move operations in the block
  Operation *terminator = block->getTerminator();
  for (int idx : schedule) {
    for (Operation *op : workingStages[idx].getOps()) {
      if (op != terminator)
        op->moveBefore(terminator);
    }
  }

  // 5. Reconstruct the sorted StageInfo vector for the caller
  SmallVector<StageInfo, 4> sortedStages;
  sortedStages.reserve(n);
  for (int idx : schedule) {
    sortedStages.push_back(std::move(workingStages[idx]));
  }

  return sortedStages;
}

FailureOr<SmallVector<StageInfo, 4>>
StageDependencyAnalyzer::runAndReorder(RewriterBase &rewriter) {
  (void)rewriter;
  auto stagesOrErr = collectStages(block);
  if (failed(stagesOrErr))
    return failure();

  return analyzeAndReorderStages(std::move(*stagesOrErr));
}
