#include "dicp/Dialect/LinalgExt/Analysis/DimAnalyzer.h"
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
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include <algorithm>

#define DEBUG_TYPE "npu-unroll-pipeline"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace dicp;
using namespace LinalgExt;

namespace mlir {
namespace dicp {
namespace LinalgExt {
#define GEN_PASS_DEF_NPUUNROOLPIPELINE
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
    LDBG("Verification FAILED: Loop bounds or step are dynamic.");
    return failure();
  }

  int64_t step = stepOpt.value();
  if (step == 0) {
    LDBG("Verification FAILED: Infinite loop (step = 0).");
    return failure();
  }

  int64_t lb = lbOpt.value();
  int64_t ub = ubOpt.value();
  if (step > 0 && lb >= ub) {
    LDBG("Verification FAILED: Loop body is never executed.");
    return failure();
  }

  int64_t tripCount = (ub - lb + step - 1) / step;
  LDBG("Verification PASSED. Static Trip Count: " << tripCount);
  return success();
}

// Marks operations that define yielded values for tensor/memref iter_args
// This allows us to track loop-carried dependencies across unrolled iterations.
static LogicalResult markYieldSources(scf::ForOp forOp) {
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

  for (auto [idx, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
    Value yieldVal = yieldOp.getOperand(idx);

    // Only strictly necessary for SSA values (tensors/scalars), but harmless
    // for others.
    if (auto defOp = yieldVal.getDefiningOp()) {
      std::string attrName = "dicp.yield_for_iter_arg." + std::to_string(idx);
      // We assume one op might feed multiple yield args, though rare.
      // Ideally we check if attr exists, but simple overwrite is okay for 1:1.
      defOp->setAttr(
          attrName,
          IntegerAttr::get(IntegerType::get(forOp.getContext(), 32), idx));
      LDBG("  Marked op '" << defOp->getName()
                           << "' as yield source for iter_arg " << idx);
    }
  }
  return success();
}

static Operation *getYieldSourceForIterArg(scf::ForOp forOp, int iterArgIdx) {
  // Linear scan is acceptable for loop bodies which are typically small-ish
  for (Operation &op : forOp.getBody()->without_terminator()) {
    std::string attrName =
        "dicp.yield_for_iter_arg." + std::to_string(iterArgIdx);
    if (op.hasAttr(attrName)) {
      return &op;
    }
  }
  return nullptr;
}

class NPUUnrollPipeline {
public:
  NPUUnrollPipeline(scf::ForOp forOp, int unrollFactor,
                    const std::vector<StageInfo> &orderedStages)
      : forOp(forOp), unrollFactor(unrollFactor), stages(orderedStages) {}

  LogicalResult run(RewriterBase &rewriter);

private:
  scf::ForOp forOp;
  int unrollFactor;
  const std::vector<StageInfo> &stages;
  int maxFlagPerIter = 0;

  // Map: OriginalValue -> Vector of Unrolled Values (one per iteration)
  DenseMap<Value, SmallVector<Value>> valueMapping;
  // Map: OriginalOp -> Vector of Unrolled Ops (one per iteration)
  // Needed to find cloned yield sources.
  DenseMap<Operation *, SmallVector<Operation *>> opMapping;

  void calculateMaxFlagStride();
  void prepareInitialMappings(RewriterBase &rewriter);
  void updateHivmFlag(Operation *op, int iterIdx, RewriterBase &rewriter);

  Value getUnrolledValue(Value originalVal, int iterIdx);
  Operation *cloneAndUpdateOperands(RewriterBase &rewriter, Operation *op,
                                    int iterIdx);
};

void NPUUnrollPipeline::calculateMaxFlagStride() {
  int maxFlag = -1;
  for (const auto &stage : stages) {
    for (Operation *op : stage.ops) {
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
  LDBG("Flag Stride calculated: " << maxFlagPerIter);
}

void NPUUnrollPipeline::prepareInitialMappings(RewriterBase &rewriter) {
  LDBG(">>> [Unroll] Preparing Initial Mappings (Constants & IVs)...");
  Location loc = forOp.getLoc();
  Value lb = forOp.getLowerBound();
  Value step = forOp.getStep();
  Value iv = forOp.getInductionVar();
  Type ivType = iv.getType();

  valueMapping[iv].resize(unrollFactor);
  auto iterArgs = forOp.getRegionIterArgs();
  for (Value arg : iterArgs) {
    valueMapping[arg].resize(unrollFactor, nullptr);
  }

  for (int i = 0; i < unrollFactor; ++i) {
    // 1. IV Calculation
    Value idxVal = rewriter.create<arith::ConstantIndexOp>(loc, i);
    Value idxValTyped = idxVal;
    if (ivType != idxVal.getType())
      idxValTyped = rewriter.create<arith::IndexCastOp>(loc, ivType, idxVal);

    Value stepOffset = rewriter.create<arith::MulIOp>(loc, step, idxValTyped);
    Value currentIV = rewriter.create<arith::AddIOp>(loc, lb, stepOffset);
    valueMapping[iv][i] = currentIV;

    // 2. Simple IterArg Calculation (e.g. arithmetic induction)
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    for (auto it : llvm::enumerate(iterArgs)) {
      Value iterArg = it.value();
      Value yieldVal = yieldOp.getOperand(it.index());

      Operation *defOp = yieldVal.getDefiningOp();
      bool isSimpleIV = false;
      int64_t stepConst = 0;

      if (auto addOp = dyn_cast_or_null<arith::AddIOp>(defOp)) {
        Value lhs = addOp.getLhs();
        Value rhs = addOp.getRhs();
        Value constOp = nullptr;
        if (lhs == iterArg)
          constOp = rhs;
        else if (rhs == iterArg)
          constOp = lhs;

        if (constOp) {
          if (auto cst = constOp.getDefiningOp<arith::ConstantIndexOp>()) {
            stepConst = cst.value();
            isSimpleIV = true;
          } else if (auto cst = constOp.getDefiningOp<arith::ConstantIntOp>()) {
            stepConst = cst.value();
            isSimpleIV = true;
          }
        }
      }

      if (isSimpleIV) {
        Value initVal = forOp.getInitArgs()[it.index()];
        Value kVal = rewriter.create<arith::ConstantIndexOp>(loc, i);
        Value kValTyped = kVal;
        if (iterArg.getType() != kVal.getType())
          kValTyped =
              rewriter.create<arith::IndexCastOp>(loc, iterArg.getType(), kVal);

        Value stepVal;
        if (iterArg.getType().isIndex())
          stepVal = rewriter.create<arith::ConstantIndexOp>(loc, stepConst);
        else
          stepVal = rewriter.create<arith::ConstantIntOp>(
              loc, iterArg.getType(), stepConst);

        Value offset = rewriter.create<arith::MulIOp>(loc, kValTyped, stepVal);
        Value currVal = rewriter.create<arith::AddIOp>(loc, initVal, offset);
        valueMapping[iterArg][i] = currVal;
      }
    }
  }
}

Value NPUUnrollPipeline::getUnrolledValue(Value originalVal, int iterIdx) {
  // 1. Check existing mapping (simple IVs or previously cloned ops)
  if (valueMapping.count(originalVal)) {
    if (iterIdx >= 0 && iterIdx < valueMapping[originalVal].size()) {
      Value mapped = valueMapping[originalVal][iterIdx];
      if (mapped)
        return mapped;
    }
  }

  // 2. Handle BlockArguments (IterArgs)
  if (auto arg = dyn_cast<BlockArgument>(originalVal)) {
    if (arg.getOwner() == forOp.getBody()) {
      // IV is handled in prepareInitialMappings (Slot 0 of args)
      if (arg.getArgNumber() == 0)
        return nullptr;

      // IterArgs start at index 1
      int iterArgIdx = arg.getArgNumber() - 1;

      // Case 2a: Iteration 0 uses the Loop Init Args (Full unroll)
      if (iterIdx == 0) {
        return forOp.getInitArgs()[iterArgIdx];
      }

      // Case 2b: Iteration K > 0 uses Yield result from K-1
      // Strategy: Find the op marked as yield source and look up its clone.
      Operation *yieldSourceOp = getYieldSourceForIterArg(forOp, iterArgIdx);
      if (yieldSourceOp) {
        // The YieldOp operand tells us which result of the source op is used
        auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        Value yieldOperand = yieldOp.getOperand(iterArgIdx);

        if (auto res = dyn_cast<OpResult>(yieldOperand)) {
          // If yield operand is a direct result of the marked op
          if (res.getOwner() == yieldSourceOp) {
            int resIdx = res.getResultNumber();
            // Check if the source op for the previous iteration was cloned
            if (opMapping.count(yieldSourceOp) &&
                iterIdx - 1 < opMapping[yieldSourceOp].size()) {
              Operation *prevClone = opMapping[yieldSourceOp][iterIdx - 1];
              if (prevClone) {
                return prevClone->getResult(resIdx);
              } else {
                LDBG("    WARNING: Yield source clone missing for iter "
                     << iterIdx - 1);
              }
            }
          }
        } else if (auto argOperand = dyn_cast<BlockArgument>(yieldOperand)) {
          // The yield operand is an IterArg itself (Pass-through)
          // Recursively resolve it
          return getUnrolledValue(argOperand, iterIdx - 1);
        }
      }

      // Fallback: If no complex logic found, try recursive lookup on yield
      // operand (This handles cases where the yield val is invariant or defined
      // elsewhere)
      Value directYieldVal =
          cast<scf::YieldOp>(forOp.getBody()->getTerminator())
              .getOperand(iterArgIdx);
      return getUnrolledValue(directYieldVal, iterIdx - 1);
    }
  }

  // 3. Invariant or Global values
  return originalVal;
}

Operation *NPUUnrollPipeline::cloneAndUpdateOperands(RewriterBase &rewriter,
                                                     Operation *op,
                                                     int iterIdx) {
  IRMapping mapper;

  // Walk the operation to identify and map all externally defined values used
  // within 'op' or its nested regions. This ensures that when 'op' is cloned,
  // any references to values defined in the original loop scope are correctly
  // remapped to their unrolled counterparts for the current iteration.
  op->walk([&](Operation *nestedOp) {
    for (Value operand : nestedOp->getOperands()) {
      // Skip if already mapped.
      if (mapper.contains(operand))
        continue;

      bool isExternal = false;
      // Check if the operand is a BlockArgument defined outside of 'op'.
      if (auto arg = dyn_cast<BlockArgument>(operand)) {
        Operation *parentOp = arg.getOwner()->getParentOp();
        // It is external if the parent op is neither 'op' nor a descendant of 'op'.
        if (parentOp != op && !op->isAncestor(parentOp))
          isExternal = true;
      }
      // Check if the operand is an OpResult defined outside of 'op'.
      else if (auto defOp = operand.getDefiningOp()) {
        // It is external if the defining op is neither 'op' nor a descendant of 'op'.
        if (defOp != op && !op->isAncestor(defOp))
          isExternal = true;
      }

      if (isExternal) {
        // Retrieve the unrolled value for the current iteration.
        Value replacement = getUnrolledValue(operand, iterIdx);
        if (replacement) {
          mapper.map(operand, replacement);
        }
      }
    }
  });

  // Clone the operation using the populated mapper.
  // This handles deep cloning and operand remapping for both the op and its
  // nested regions (like scf.for body).
  Operation *clone = rewriter.clone(*op, mapper);

  // Record the cloned op in the mapping for future lookups (Yield Source
  // resolution)
  if (opMapping[op].size() <= iterIdx)
    opMapping[op].resize(unrollFactor);
  opMapping[op][iterIdx] = clone;

  return clone;
}

void NPUUnrollPipeline::updateHivmFlag(Operation *op, int iterIdx,
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

LogicalResult NPUUnrollPipeline::run(RewriterBase &rewriter) {
  calculateMaxFlagStride();

  // Resize mappings
  for (Operation &op : forOp.getBody()->without_terminator()) {
    opMapping[&op].resize(unrollFactor, nullptr);
    for (Value res : op.getResults()) {
      valueMapping[res].resize(unrollFactor, nullptr);
    }
  }

  rewriter.setInsertionPoint(forOp);
  prepareInitialMappings(rewriter);

  LDBG(">>> [Unroll] Starting Clone (Stage-Major Order)...");

  for (const auto &stage : stages) {
    LDBG("  Processing Stage " << stage.id);
    for (int iterIdx = 0; iterIdx < unrollFactor; ++iterIdx) {
      for (Operation *op : stage.ops) {
        if (isa<scf::YieldOp>(op))
          continue;

        Operation *clonedOp = cloneAndUpdateOperands(rewriter, op, iterIdx);

        LLVM_DEBUG({
          llvm::dbgs() << "[" DEBUG_TYPE "]     [Stg " << stage.id << "][Iter "
                       << iterIdx << "] Cloned Op: ";
          clonedOp->print(llvm::dbgs());
          llvm::dbgs() << "\n";
        });

        updateHivmFlag(clonedOp, iterIdx, rewriter);

        // Update value mapping for results
        for (auto it : llvm::zip(op->getResults(), clonedOp->getResults())) {
          Value originalRes = std::get<0>(it);
          Value newRes = std::get<1>(it);
          if (iterIdx < valueMapping[originalRes].size())
            valueMapping[originalRes][iterIdx] = newRes;
        }
      }
    }
  }

  LDBG(">>> [Unroll] Replacing Loop Results...");
  Operation *terminator = forOp.getBody()->getTerminator();
  SmallVector<Value> finalResults;

  // The final results correspond to the yield values of the LAST iteration
  int lastIter = unrollFactor - 1;

  for (Value operand : terminator->getOperands()) {
    Value remapped = getUnrolledValue(operand, lastIter);
    if (!remapped)
      remapped = operand;
    finalResults.push_back(remapped);
  }

  if (forOp.getNumResults() != finalResults.size()) {
    return forOp.emitError("Unroll result count mismatch");
  }

  rewriter.replaceOp(forOp, finalResults);
  LDBG("<<< Pass Complete.");
  return success();
}

struct NPUUnroolPipelinePass
    : public mlir::dicp::LinalgExt::impl::NPUUnroolPipelineBase<
          NPUUnroolPipelinePass> {
  NPUUnroolPipelinePass() = default;

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    SmallVector<scf::ForOp> loops;
    func.walk([&](scf::ForOp loop) {
      if (loop->hasAttr(mlir::triton::kNumStagesAttrName))
        loops.push_back(loop);
    });

    if (loops.size() != 1) {
      LDBG("The number of candidate loops is not one.");
      return;
    }

    scf::ForOp targetLoop = loops[0];
    if (failed(verifyLoopForPipelining(targetLoop))) {
      LDBG("Loop verification failed, skipping.");
      return;
    }

    int numStages = mlir::cast<IntegerAttr>(
                        targetLoop->getAttr(mlir::triton::kNumStagesAttrName))
                        .getInt();

    LDBG("Processing Loop with num_stages = " << numStages);

    mlir::IRRewriter rewriter(func.getContext());
    AliasAnalysis &aa = getAnalysis<AliasAnalysis>();
    // 1. Analyze and Reorder Stages (Topological Sort)
    StageDependencyAnalyzer analyzer(targetLoop.getBody(), aa);
    auto orderedStagesOrFailure = analyzer.runAndReorder(rewriter);

    if (failed(orderedStagesOrFailure)) {
      LDBG("Failed to reorder stages (cyclic dependency detected).");
      signalPassFailure();
      return;
    }
    // 2. Mark Yield Sources for complex iter_args
    if (failed(markYieldSources(targetLoop))) {
      signalPassFailure();
      return;
    }

    // 3. Execute Unroll (Stage-Major)
    NPUUnrollPipeline unroller(targetLoop, numStages,
                               orderedStagesOrFailure.value());
    if (failed(unroller.run(rewriter))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::dicp::LinalgExt::createNPUUnroolPipelinePass() {
  return std::make_unique<NPUUnroolPipelinePass>();
}