#include "dicp/Dialect/LinalgExt/Analysis/StageDependencyAnalyzer.h"
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include <algorithm>
#include <optional>

#define DEBUG_TYPE "pipeline-loop-unroll"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace dicp;
using namespace LinalgExt;

namespace mlir {
namespace dicp {
namespace LinalgExt {
#define GEN_PASS_DEF_PIPELINELOOPUNROLL
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace LinalgExt
} // namespace dicp
} // namespace mlir

namespace {

LogicalResult verifyLoopForPipelining(scf::ForOp forOp) {
  auto lbOpt = getConstantIntValue(forOp.getLowerBound());
  auto ubOpt = getConstantIntValue(forOp.getUpperBound());
  auto stepOpt = getConstantIntValue(forOp.getStep());

  if (!lbOpt.has_value() || !ubOpt.has_value() || !stepOpt.has_value()) {
    LDBG("[Verify] reject dynamic loop bounds or step");
    return failure();
  }

  int64_t step = stepOpt.value();
  if (step == 0) {
    LDBG("[Verify] reject zero step");
    return failure();
  }

  int64_t lb = lbOpt.value();
  int64_t ub = ubOpt.value();
  if (step > 0 && lb >= ub) {
    LDBG("[Verify] reject empty iteration space");
    return failure();
  }

  int64_t tripCount = (ub - lb + step - 1) / step;
  LDBG("[Verify] static trip count = " << tripCount);
  return success();
}

class PipelineLoopUnroller {
public:
  PipelineLoopUnroller(scf::ForOp forOp, int unrollFactor,
                       const SmallVector<StageInfo, 4> &orderedStages)
      : forOp(forOp), unrollFactor(unrollFactor), stages(orderedStages) {}

  LogicalResult run(RewriterBase &rewriter);

private:
  struct ScheduledStageInstance {
    int stageIdx = -1;
    int iterIdx = -1;
  };

  scf::ForOp forOp;
  int unrollFactor;
  const SmallVector<StageInfo, 4> &stages;
  int maxFlagPerIter = 0;

  // Map: OriginalValue -> Vector of Unrolled Values (one per iteration)
  DenseMap<Value, SmallVector<Value>> valueMapping;

  void calculateMaxFlagStride();
  void prepareInitialMappings(RewriterBase &rewriter);
  void updateHivmFlag(Operation *op, int iterIdx, RewriterBase &rewriter);
  FailureOr<SmallVector<ScheduledStageInstance, 16>> buildExecutionSchedule();
  FailureOr<std::optional<int>> findSourceStageForLoopCarriedValue(
      Value value, const DenseMap<Operation *, int> &opToStageIndex,
      DenseSet<unsigned> &visitedIterArgs);

  FailureOr<Value> resolveValueForIteration(Value value, int iterIdx);
  FailureOr<Operation *> cloneAndUpdateOperands(RewriterBase &rewriter,
                                                Operation *op, int iterIdx);
};

void PipelineLoopUnroller::calculateMaxFlagStride() {
  int maxFlag = -1;
  for (const auto &stage : stages) {
    for (Operation *op : stage.getOps()) {
      if (auto syncSetOp = dyn_cast<hivm::SyncBlockSetOp>(op)) {
        int flag = getConstantIntValue(syncSetOp.getFlagId()).value_or(-1);
        if (flag > maxFlag)
          maxFlag = flag;
      } else if (auto syncWaitOp = dyn_cast<hivm::SyncBlockWaitOp>(op)) {
        int flag = getConstantIntValue(syncWaitOp.getFlagId()).value_or(-1);
        if (flag > maxFlag)
          maxFlag = flag;
      }
    }
  }
  this->maxFlagPerIter = (maxFlag < 0) ? 0 : (maxFlag + 1);
  LDBG("[Flags] per-iteration flag stride = " << maxFlagPerIter);
}

void PipelineLoopUnroller::prepareInitialMappings(RewriterBase &rewriter) {
  LDBG("[Unroll] prepare initial IV and iter_arg mappings");
  Location loc = forOp.getLoc();
  Value lb = forOp.getLowerBound();
  Value step = forOp.getStep();
  Value iv = forOp.getInductionVar();
  Type ivType = iv.getType();

  valueMapping[iv].resize(unrollFactor);
  auto iterArgs = forOp.getRegionIterArgs();
  for (Value arg : iterArgs)
    valueMapping[arg].resize(unrollFactor, nullptr);

  for (int i = 0; i < unrollFactor; ++i) {
    // Materialize the induction variable for each logical iteration.
    Value idxVal = rewriter.create<arith::ConstantIndexOp>(loc, i);
    Value idxValTyped = idxVal;
    if (ivType != idxVal.getType())
      idxValTyped = rewriter.create<arith::IndexCastOp>(loc, ivType, idxVal);

    Value stepOffset = rewriter.create<arith::MulIOp>(loc, step, idxValTyped);
    Value currentIV = rewriter.create<arith::AddIOp>(loc, lb, stepOffset);
    valueMapping[iv][i] = currentIV;
  }
}

FailureOr<std::optional<int>>
PipelineLoopUnroller::findSourceStageForLoopCarriedValue(
    Value value, const DenseMap<Operation *, int> &opToStageIndex,
    DenseSet<unsigned> &visitedIterArgs) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getOwner() != forOp.getBody())
      return std::optional<int>{};

    unsigned numInductionVars = forOp.getNumInductionVars();
    if (arg.getArgNumber() < numInductionVars)
      return std::optional<int>{};

    unsigned iterArgIdx = arg.getArgNumber() - numInductionVars;
    if (!visitedIterArgs.insert(iterArgIdx).second) {
      LDBG("  Loop-carried iter_arg #" << iterArgIdx
                                       << " is forwarded without introducing "
                                          "a new stage-local producer");
      return std::optional<int>{};
    }

    Value yieldedValue = cast<scf::YieldOp>(forOp.getBody()->getTerminator())
                             .getOperand(iterArgIdx);
    return findSourceStageForLoopCarriedValue(yieldedValue, opToStageIndex,
                                              visitedIterArgs);
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp || !forOp->isAncestor(defOp))
    return std::optional<int>{};

  auto it = opToStageIndex.find(defOp);
  if (it == opToStageIndex.end()) {
    defOp->emitError()
        << "Failed to locate the producing stage for a loop-carried value";
    return failure();
  }

  return std::optional<int>(it->second);
}

FailureOr<SmallVector<PipelineLoopUnroller::ScheduledStageInstance, 16>>
PipelineLoopUnroller::buildExecutionSchedule() {
  if (stages.empty())
    return SmallVector<ScheduledStageInstance, 16>{};

  DenseMap<int, int> stageIdToIndex;
  DenseMap<Operation *, int> opToStageIndex;
  for (auto [stageIdx, stage] : llvm::enumerate(stages)) {
    stageIdToIndex[stage.id] = stageIdx;
    for (Operation *op : stage.getOps())
      opToStageIndex[op] = stageIdx;
  }

  unsigned numIterArgs = forOp.getRegionIterArgs().size();
  SmallVector<DenseSet<int>, 8> iterArgConsumers(numIterArgs);
  for (auto [stageIdx, stage] : llvm::enumerate(stages)) {
    int currentStageIdx = stageIdx;
    for (Operation *op : stage.getOps()) {
      op->walk([&](Operation *nestedOp) -> WalkResult {
        for (Value operand : nestedOp->getOperands()) {
          auto arg = dyn_cast<BlockArgument>(operand);
          if (!arg || arg.getOwner() != forOp.getBody())
            continue;

          unsigned numInductionVars = forOp.getNumInductionVars();
          if (arg.getArgNumber() < numInductionVars)
            continue;

          unsigned iterArgIdx = arg.getArgNumber() - numInductionVars;
          iterArgConsumers[iterArgIdx].insert(currentStageIdx);
        }
        return WalkResult::advance();
      });
    }
  }

  auto getNodeIndex = [&](int stageIdx, int iterIdx) {
    return stageIdx * unrollFactor + iterIdx;
  };
  auto decodeNodeIndex = [&](int nodeIdx) -> ScheduledStageInstance {
    ScheduledStageInstance item;
    item.stageIdx = nodeIdx / unrollFactor;
    item.iterIdx = nodeIdx % unrollFactor;
    return item;
  };

  int nodeCount = stages.size() * unrollFactor;
  SmallVector<int, 16> indegree(nodeCount, 0);
  SmallVector<SmallVector<int, 4>, 16> succs(nodeCount);
  SmallVector<DenseSet<int>, 16> succSets(nodeCount);

  auto addEdge = [&](int fromNode, int toNode, StringRef reason) {
    if (!succSets[fromNode].insert(toNode).second)
      return;

    succs[fromNode].push_back(toNode);
    ++indegree[toNode];

    ScheduledStageInstance from = decodeNodeIndex(fromNode);
    ScheduledStageInstance to = decodeNodeIndex(toNode);
    LDBG("[Schedule] edge stage "
         << stages[from.stageIdx].id << " iter " << from.iterIdx << " -> stage "
         << stages[to.stageIdx].id << " iter " << to.iterIdx << " because "
         << reason);
  };

  for (auto [stageIdx, stage] : llvm::enumerate(stages)) {
    for (int predStageId : stage.preds) {
      auto predIt = stageIdToIndex.find(predStageId);
      if (predIt == stageIdToIndex.end()) {
        forOp.emitError() << "Unknown predecessor stage id " << predStageId
                          << " while building the pipeline execution schedule";
        return failure();
      }

      for (int iterIdx = 0; iterIdx < unrollFactor; ++iterIdx)
        addEdge(getNodeIndex(predIt->second, iterIdx),
                getNodeIndex(stageIdx, iterIdx), "intra-iteration stage DAG");
    }
  }

  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  for (auto [iterArgIdx, yieldedValue] :
       llvm::enumerate(yieldOp.getOperands())) {
    DenseSet<unsigned> visitedIterArgs;
    FailureOr<std::optional<int>> sourceStage =
        findSourceStageForLoopCarriedValue(yieldedValue, opToStageIndex,
                                           visitedIterArgs);
    if (failed(sourceStage))
      return failure();
    if (!*sourceStage)
      continue;

    if (iterArgConsumers[iterArgIdx].empty()) {
      LDBG("[Schedule] loop-carried iter_arg #"
           << iterArgIdx << " has no stage-local consumers");
      continue;
    }

    for (int iterIdx = 1; iterIdx < unrollFactor; ++iterIdx) {
      for (int consumerStageIdx : iterArgConsumers[iterArgIdx]) {
        addEdge(getNodeIndex(*(*sourceStage), iterIdx - 1),
                getNodeIndex(consumerStageIdx, iterIdx),
                llvm::formatv("loop-carried iter_arg #{0}", iterArgIdx).str());
      }
    }
  }

  SmallVector<int, 16> ready;
  ready.reserve(nodeCount);
  for (int nodeIdx = 0; nodeIdx < nodeCount; ++nodeIdx) {
    if (indegree[nodeIdx] == 0)
      ready.push_back(nodeIdx);
  }
  llvm::sort(ready);

  SmallVector<ScheduledStageInstance, 16> schedule;
  schedule.reserve(nodeCount);
  while (!ready.empty()) {
    int currentNode = ready.front();
    ready.erase(ready.begin());

    schedule.push_back(decodeNodeIndex(currentNode));
    for (int succ : succs[currentNode]) {
      --indegree[succ];
      if (indegree[succ] == 0)
        ready.push_back(succ);
    }
    llvm::sort(ready);
  }

  if (schedule.size() != static_cast<size_t>(nodeCount)) {
    forOp.emitError()
        << "Failed to build a legal expanded execution schedule for pipeline "
           "unroll. A cross-iteration cycle was detected.";
    return failure();
  }

  LDBG("[Schedule] expanded execution order:");
  for (auto [order, item] : llvm::enumerate(schedule)) {
    LDBG("[Schedule] [" << order << "] stage " << stages[item.stageIdx].id
                        << ", iter " << item.iterIdx);
  }

  return schedule;
}

FailureOr<Value> PipelineLoopUnroller::resolveValueForIteration(Value value,
                                                                int iterIdx) {
  if (iterIdx < 0 || iterIdx >= unrollFactor) {
    forOp.emitError() << "Invalid logical iteration index " << iterIdx
                      << " while resolving a loop value";
    return failure();
  }

  if (auto it = valueMapping.find(value); it != valueMapping.end()) {
    if (iterIdx < static_cast<int>(it->second.size()) && it->second[iterIdx]) {
      LDBG("  Resolved mapped value for iteration "
           << iterIdx << ": " << value << " -> " << it->second[iterIdx]);
      return it->second[iterIdx];
    }
  }

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    Operation *parentOp = arg.getOwner()->getParentOp();
    if (parentOp == forOp.getOperation()) {
      unsigned numInductionVars = forOp.getNumInductionVars();
      if (arg.getArgNumber() < numInductionVars) {
        forOp.emitError() << "Induction variable mapping missing for logical "
                             "iteration "
                          << iterIdx;
        return failure();
      }

      unsigned iterArgIdx = arg.getArgNumber() - numInductionVars;
      if (iterArgIdx >= forOp.getRegionIterArgs().size()) {
        forOp.emitError() << "Invalid iter_arg index " << iterArgIdx
                          << " while resolving loop-carried value";
        return failure();
      }

      if (iterIdx == 0) {
        Value initValue = forOp.getInitArgs()[iterArgIdx];
        LDBG("  Resolved iter_arg #" << iterArgIdx
                                     << " for logical iteration 0 from init "
                                     << "operand: " << initValue);
        return initValue;
      }

      Value yieldedValue = cast<scf::YieldOp>(forOp.getBody()->getTerminator())
                               .getOperand(iterArgIdx);
      LDBG("  Resolving iter_arg #"
           << iterArgIdx << " for logical iteration " << iterIdx
           << " from previous yield operand: " << yieldedValue);
      return resolveValueForIteration(yieldedValue, iterIdx - 1);
    }

    if (parentOp && forOp->isAncestor(parentOp)) {
      forOp.emitError()
          << "Unsupported nested block argument captured from inside the "
             "original loop body";
      return failure();
    }

    return value;
  }

  if (Operation *defOp = value.getDefiningOp()) {
    if (forOp->isAncestor(defOp)) {
      defOp->emitError()
          << "Unresolved loop-local value during pipeline unroll. The value "
             "is defined inside the original loop body but no cloned value "
             "exists for logical iteration "
          << iterIdx;
      return failure();
    }
  }

  return value;
}

FailureOr<Operation *>
PipelineLoopUnroller::cloneAndUpdateOperands(RewriterBase &rewriter,
                                             Operation *op, int iterIdx) {
  IRMapping mapper;
  LogicalResult status = success();

  // Remap every captured SSA value to the correct logical iteration before
  // cloning. Loop-local values must never fall back to the original loop body,
  // otherwise the transformed IR would violate dominance once the loop is
  // replaced.
  op->walk([&](Operation *nestedOp) -> WalkResult {
    for (OpOperand &operand : nestedOp->getOpOperands()) {
      Value value = operand.get();
      if (mapper.contains(value))
        continue;

      bool isExternal = false;
      if (auto arg = dyn_cast<BlockArgument>(value)) {
        Operation *parentOp = arg.getOwner()->getParentOp();
        if (parentOp != op && !op->isAncestor(parentOp))
          isExternal = true;
      } else if (Operation *defOp = value.getDefiningOp()) {
        if (defOp != op && !op->isAncestor(defOp))
          isExternal = true;
      }

      if (!isExternal)
        continue;

      FailureOr<Value> replacement = resolveValueForIteration(value, iterIdx);
      if (failed(replacement)) {
        status = failure();
        return WalkResult::interrupt();
      }

      mapper.map(value, *replacement);
      LDBG("    Remapped captured operand for logical iteration "
           << iterIdx << ": " << value << " -> " << *replacement);
    }
    return WalkResult::advance();
  });

  if (failed(status))
    return failure();

  Operation *clone = rewriter.clone(*op, mapper);
  return clone;
}

void PipelineLoopUnroller::updateHivmFlag(Operation *op, int iterIdx,
                                          RewriterBase &rewriter) {
  if (maxFlagPerIter == 0)
    return;
  auto update = [&](auto syncOp) {
    if (auto attr = syncOp.getStaticFlagIdAttr()) {
      int64_t newFlag = attr.getInt() + iterIdx * maxFlagPerIter;
      syncOp.setStaticFlagIdAttr(rewriter.getI64IntegerAttr(newFlag));
    }
  };
  if (auto setOp = dyn_cast<hivm::SyncBlockSetOp>(op))
    update(setOp);
  else if (auto waitOp = dyn_cast<hivm::SyncBlockWaitOp>(op))
    update(waitOp);
}

LogicalResult PipelineLoopUnroller::run(RewriterBase &rewriter) {
  calculateMaxFlagStride();

  // Resize mappings
  for (Operation &op : forOp.getBody()->without_terminator()) {
    for (Value res : op.getResults())
      valueMapping[res].resize(unrollFactor, nullptr);
  }

  rewriter.setInsertionPoint(forOp);
  prepareInitialMappings(rewriter);

  FailureOr<SmallVector<ScheduledStageInstance, 16>> schedule =
      buildExecutionSchedule();
  if (failed(schedule))
    return failure();

  LDBG("[Unroll] start cloning in expanded schedule order");

  for (const ScheduledStageInstance &item : *schedule) {
    const StageInfo &stage = stages[item.stageIdx];
    SmallVector<Operation *, 16> stageOps = stage.getOps();
    LDBG("[Unroll] process stage " << stage.id << " at iter " << item.iterIdx
                                   << " with " << stage.subStages.size()
                                   << " substages and " << stageOps.size()
                                   << " ops");
    for (Operation *op : stageOps) {
      if (isa<scf::YieldOp>(op))
        continue;

      FailureOr<Operation *> clonedOpOrFailure =
          cloneAndUpdateOperands(rewriter, op, item.iterIdx);
      if (failed(clonedOpOrFailure))
        return failure();
      Operation *clonedOp = *clonedOpOrFailure;

      LLVM_DEBUG({
        llvm::dbgs() << "[" DEBUG_TYPE "]     [Stg " << stage.id << "][Iter "
                     << item.iterIdx << "] Cloned Op: ";
        clonedOp->print(llvm::dbgs());
        llvm::dbgs() << "\n";
      });

      updateHivmFlag(clonedOp, item.iterIdx, rewriter);

      // Update value mapping for results immediately so later scheduled stage
      // instances can resolve both intra-iteration and loop-carried uses.
      for (auto it : llvm::zip(op->getResults(), clonedOp->getResults())) {
        Value originalRes = std::get<0>(it);
        Value newRes = std::get<1>(it);
        if (item.iterIdx < valueMapping[originalRes].size())
          valueMapping[originalRes][item.iterIdx] = newRes;
      }
    }
  }

  LDBG("[Unroll] replace original loop results");
  Operation *terminator = forOp.getBody()->getTerminator();
  SmallVector<Value> finalResults;

  // The final results correspond to the yield values of the LAST iteration
  int lastIter = unrollFactor - 1;

  for (Value operand : terminator->getOperands()) {
    FailureOr<Value> remapped = resolveValueForIteration(operand, lastIter);
    if (failed(remapped))
      return failure();
    LDBG("  Final loop result resolved for logical iteration "
         << lastIter << ": " << operand << " -> " << *remapped);
    finalResults.push_back(*remapped);
  }

  if (forOp.getNumResults() != finalResults.size()) {
    return forOp.emitError("Unroll result count mismatch");
  }

  rewriter.replaceOp(forOp, finalResults);
  LDBG("[Unroll] completed");
  return success();
}

struct PipelineLoopUnrollPass
    : public mlir::dicp::LinalgExt::impl::PipelineLoopUnrollBase<
          PipelineLoopUnrollPass> {
  PipelineLoopUnrollPass() = default;

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    SmallVector<scf::ForOp> loops;
    func.walk([&](scf::ForOp loop) {
      if (loop->hasAttr(mlir::triton::kNumStagesAttrName))
        loops.push_back(loop);
    });

    if (loops.size() != 1) {
      LDBG("[Pass] expected exactly one pipeline candidate loop");
      return;
    }

    scf::ForOp targetLoop = loops[0];
    if (failed(verifyLoopForPipelining(targetLoop))) {
      LDBG("[Pass] skip loop after verification failure");
      return;
    }

    int numStages = mlir::cast<IntegerAttr>(
                        targetLoop->getAttr(mlir::triton::kNumStagesAttrName))
                        .getInt();

    LDBG("[Pass] process pipeline loop with num_stages = " << numStages);

    mlir::IRRewriter rewriter(func.getContext());
    AliasAnalysis &aa = getAnalysis<AliasAnalysis>();
    // 1. Analyze and Reorder Stages (Topological Sort)
    StageDependencyAnalyzer analyzer(targetLoop.getBody(), aa);
    auto orderedStagesOrFailure = analyzer.runAndReorder(rewriter);

    if (failed(orderedStagesOrFailure)) {
      LDBG("[Pass] failed to reorder stages due to dependency cycle");
      signalPassFailure();
      return;
    }
    // 2. Execute unroll while preserving stage order and SCF loop-carried
    // semantics.
    PipelineLoopUnroller unroller(targetLoop, numStages,
                                  orderedStagesOrFailure.value());
    if (failed(unroller.run(rewriter))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::dicp::LinalgExt::createPipelineLoopUnrollPass() {
  return std::make_unique<PipelineLoopUnrollPass>();
}
