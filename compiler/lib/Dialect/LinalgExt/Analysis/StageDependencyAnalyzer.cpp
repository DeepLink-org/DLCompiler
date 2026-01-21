
#include "dicp/Dialect/LinalgExt/Analysis/StageDependencyAnalyzer.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "npu-stage-dep-analyzer"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace dicp;

void StageDependencyAnalyzer::collectStages() {
  LDBG(">>> [Analysis] Collecting Stages...");
  stages.clear();
  StageInfo currentStage;
  currentStage.id = 0;

  Block *body = forOp.getBody();
  for (Operation &op : body->without_terminator()) {
    // If the current operation is a SyncBlockWaitOp, it marks the start of a
    // new stage. We complete the current stage (if it's not empty) and start a
    // new one. The SyncBlockWaitOp will become the first operation of the new
    // stage.
    if (isa<hivm::SyncBlockWaitOp>(op)) {
      if (!currentStage.ops.empty()) {
        stages.push_back(currentStage);
        currentStage = StageInfo();
        currentStage.id = stages.size();
      }
    }

    currentStage.ops.push_back(&op);

    // Mark the stage if it contains a sync wait operation
    if (isa<hivm::SyncBlockWaitOp>(op)) {
      currentStage.hasSync = true;
    }
  }

  if (!currentStage.ops.empty()) {
    stages.push_back(currentStage);
  }

  nodes.resize(stages.size());
  for (size_t i = 0; i < stages.size(); ++i) {
    nodes[i].id = i;
    nodes[i].stageInfo = &stages[i];
  }

  LDBG("Collected " << stages.size() << " stages.");

  // Debug: dump the ops contained in each stage (print full op IR).
  LLVM_DEBUG({
    llvm::dbgs() << "[" DEBUG_TYPE "] Detailed stage contents:\n";
    for (const auto &stage : stages) {
      llvm::dbgs() << "[" DEBUG_TYPE << "] Stage " << stage.id
                   << (stage.hasSync ? " (hasSync)" : "")
                   << " - ops: " << stage.ops.size() << "\n";
      for (Operation *op : stage.ops) {
        llvm::dbgs() << "  - ";
        op->print(llvm::dbgs());
        llvm::dbgs() << "\n";
      }
    }
  });
}

void StageDependencyAnalyzer::collectEffects() {
  for (auto &node : nodes) {
    for (Operation *op : node.stageInfo->ops) {
      // 1. SSA Def-Use (Produced Values)
      for (Value res : op->getResults()) {
        node.producedValues.insert(res);
      }
      // 1. SSA Def-Use (Consumed Values)
      for (Value operand : op->getOperands()) {
        // We only care about operands defined within the loop (not block args
        // or invariant)
        if (auto defOp = operand.getDefiningOp()) {
          if (defOp->getParentOp() == forOp) {
            node.consumedValues.insert(operand);
          }
        }
      }

      // 2. Memory Effects
      if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
        SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
        memEffect.getEffects(effects);
        for (auto &effect : effects) {
          Value val = effect.getValue();
          if (!val)
            continue;
          if (isa<MemoryEffects::Write>(effect.getEffect()))
            node.writeValues.insert(val);
          else if (isa<MemoryEffects::Read>(effect.getEffect()))
            node.readValues.insert(val);
        }
        continue;
      }
      // Explicit handling for ops not implementing MemoryEffects but having
      // semantics
      if (auto matOp =
              dyn_cast<bufferization::MaterializeInDestinationOp>(op)) {
        node.readValues.insert(matOp.getSource());
        node.writeValues.insert(matOp.getDest());
      } else if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
        node.readValues.insert(copyOp.getSource());
        node.writeValues.insert(copyOp.getTarget());
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
        for (Value writeVal : nodes[i].writeValues) {
          for (Value readVal : nodes[j].readValues) {
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

void StageDependencyAnalyzer::reorderStagesLogical() {
  std::vector<StageNode> sortedNodes = nodes;
  std::stable_sort(sortedNodes.begin(), sortedNodes.end(),
                   [](const StageNode &a, const StageNode &b) {
                     if (a.level != b.level)
                       return a.level < b.level;
                     return a.id < b.id;
                   });
  std::vector<StageInfo> newStages;
  newStages.reserve(stages.size());
  LDBG(">>> [Analysis] Reordered Stages (Logical Order):");
  for (const auto &node : sortedNodes) {
    LDBG("  Stage ID: " << node.id << ", Level: " << node.level);
    newStages.push_back(*node.stageInfo);
  }
  stages = std::move(newStages);
}

void StageDependencyAnalyzer::materializeScheduleToIR() {
  LDBG(">>> [Analysis] Materializing Schedule to IR (Physical Move)...");
  Operation *terminator = forOp.getBody()->getTerminator();
  for (const auto &stage : stages) {
    for (Operation *op : stage.ops) {
      if (op == terminator)
        continue;
      op->moveBefore(terminator);
    }
  }
}

FailureOr<std::vector<StageInfo>>
StageDependencyAnalyzer::runAndReorder(RewriterBase &rewriter) {
  collectStages();
  collectEffects();
  buildDependencyGraph();
  if (failed(computeStageLevels()))
    return failure();
  reorderStagesLogical();
  LDBG(">>> [Result] Final Stage Dependency Summary:");
  LLVM_DEBUG(for (const auto &stage
                  : stages) {
    llvm::dbgs() << "[" DEBUG_TYPE "] Stage " << stage.id << ":\n";
    llvm::dbgs() << "    Predecessors (Depends on): { stage: ";
    for (int p : stage.preds)
      llvm::dbgs() << p << " ";
    llvm::dbgs() << "}\n";
    llvm::dbgs() << "    Successors (Depended by): { stage: ";
    for (int s : stage.succs)
      llvm::dbgs() << s << " ";
    llvm::dbgs() << "}\n";
  });
  materializeScheduleToIR();
  return stages;
}