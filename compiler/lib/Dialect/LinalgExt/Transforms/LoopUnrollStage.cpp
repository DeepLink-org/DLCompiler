#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"
#include "dicp/TransformOps/Transforms.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dicp-loop-unroll"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace mlir::dicp;

namespace mlir::dicp::LinalgExt {
#define GEN_PASS_DEF_LOOPUNROLLSTAGE
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace mlir::dicp::LinalgExt

namespace {

// --- Utilities ---

/**
 * @brief Cleans up internal DICP attributes from all operations in the
 * function.
 *
 * Removes attributes starting with `kDicpStagePrefix` to ensure clean IR output
 * after the pass completes.
 */
static void cleanupInternalAttributes(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();
  int totalRemoved = 0;

  func.walk([&](Operation *op) {
    if (op->getAttrs().empty())
      return;

    // Identify attributes to keep
    SmallVector<NamedAttribute> filteredAttrs;
    bool needsUpdate = false;

    for (NamedAttribute attr : op->getAttrs()) {
      if (attr.getName().strref().starts_with(kDicpStagePrefix)) {
        needsUpdate = true;
        totalRemoved++;
      } else {
        filteredAttrs.push_back(attr);
      }
    }

    if (needsUpdate) {
      op->setAttrs(DictionaryAttr::get(ctx, filteredAttrs));
    }
  });

  if (totalRemoved > 0) {
    LDBG("Cleaned up " << totalRemoved << " internal stage attributes.");
  }
}

// --- Patterns ---

/// Helper: Rewrite logic (slightly adapted to accept PatternRewriter &).
static LogicalResult rewriteForallToFor(PatternRewriter &rewriter,
                                        scf::ForallOp forallOp,
                                        SmallVectorImpl<Operation *> &loops) {
  Location loc = forallOp.getLoc();

  // 1. Gather Bounds and Steps
  SmallVector<Value> lbs = forallOp.getLowerBound(rewriter);
  SmallVector<Value> ubs = forallOp.getUpperBound(rewriter);
  SmallVector<Value> steps = forallOp.getStep(rewriter);

  // 2. Prepare Init Args for the outermost loop (from shared_outs)
  SmallVector<Value> currentIterOperands(forallOp.getOutputs().begin(),
                                         forallOp.getOutputs().end());

  LLVM_DEBUG(llvm::dbgs() << "[ForallToFor] Creating loop nest of rank "
                          << lbs.size() << "\n");

  // 3. Create the Loop Nest
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forallOp);

  for (auto [lb, ub, step] : llvm::zip(lbs, ubs, steps)) {
    auto loop =
        rewriter.create<scf::ForOp>(loc, lb, ub, step, currentIterOperands);

    if (!loop.getBody()->empty()) {
      rewriter.eraseOp(loop.getBody()->getTerminator());
    }

    // The iter_args of this loop become the init_args for the next inner loop
    currentIterOperands.assign(loop.getRegionIterArgs().begin(),
                               loop.getRegionIterArgs().end());
    // Set insertion point to the beginning of the body for the next
    // loop/content
    rewriter.setInsertionPointToStart(loop.getBody());
    loops.push_back(loop);
  }

  if (loops.empty()) {
    // Zero-rank forall -> nothing to do; simply erase or signal failure.
    return forallOp.emitError("expected rank > 0 for scf.forall");
  }

  scf::ForOp innermostLoop = cast<scf::ForOp>(loops.back());

  // 4. Inline the Forall Body
  Block *forallBody = forallOp.getBody();
  Block *innermostBlock = innermostLoop.getBody();

  IRMapping mapping;
  // Map induction vars (first N block args of forall)
  llvm::SmallVector<Value> ivs;
  ivs.reserve(loops.size());
  for (Operation *op : loops)
    ivs.push_back(cast<scf::ForOp>(op).getInductionVar());

  for (auto [forallArg, newIv] :
       llvm::zip(forallBody->getArguments().take_front(lbs.size()), ivs)) {
    mapping.map(forallArg, newIv);
  }

  // Map the remaining forall block args (shared outputs) to innermost
  // iter_args.
  for (auto [forallArg, iterArg] :
       llvm::zip(forallBody->getArguments().drop_front(lbs.size()),
                 innermostLoop.getRegionIterArgs())) {
    mapping.map(forallArg, iterArg);
  }

  SmallVector<Value> mappedArgs;
  mappedArgs.reserve(forallBody->getNumArguments());
  for (Value arg : forallBody->getArguments())
    mappedArgs.push_back(mapping.lookup(arg));

  rewriter.mergeBlocks(forallBody, innermostBlock, mappedArgs);

  // 5. Handle the Terminator (scf.forall.in_parallel)
  auto inParallelOp =
      dyn_cast<scf::InParallelOp>(innermostBlock->getTerminator());
  if (!inParallelOp) {
    return forallOp.emitError("expected scf.forall.in_parallel terminator");
  }

  LLVM_DEBUG(llvm::dbgs() << "[ForallToFor] Processing terminator: "
                          << *inParallelOp << "\n");

  // Convert parallel updates to sequential updates on the iter_args by
  // chaining.
  DenseMap<Value, Value> accumulatorMap;
  for (Value iterArg : innermostLoop.getRegionIterArgs())
    accumulatorMap[iterArg] = iterArg;

  Block &parallelBody = inParallelOp.getRegion().front();
  rewriter.setInsertionPoint(inParallelOp);

  for (Operation &op : llvm::make_early_inc_range(parallelBody)) {
    if (auto parallelInsert = dyn_cast<tensor::ParallelInsertSliceOp>(op)) {

      Value destIterArg = parallelInsert.getDest();

      auto it = accumulatorMap.find(destIterArg);
      if (it == accumulatorMap.end())
        return parallelInsert.emitError(
            "parallel_insert_slice dest is not a mapped loop iter_arg");

      Value currentAcc = it->second;

      auto insertSlice = rewriter.create<tensor::InsertSliceOp>(
          parallelInsert.getLoc(), parallelInsert.getSource(), currentAcc,
          parallelInsert.getOffsets(), parallelInsert.getSizes(),
          parallelInsert.getStrides(), parallelInsert.getStaticOffsets(),
          parallelInsert.getStaticSizes(), parallelInsert.getStaticStrides());

      accumulatorMap[destIterArg] = insertSlice.getResult();
      continue;
    }

    if (isa<scf::YieldOp>(op))
      continue;

    op.moveBefore(inParallelOp);
  }

  // 6. Yield the accumulated results from innermost loop body
  SmallVector<Value> yieldOperands;
  for (Value iterArg : innermostLoop.getRegionIterArgs())
    yieldOperands.push_back(accumulatorMap[iterArg]);

  rewriter.create<scf::YieldOp>(loc, yieldOperands);

  // Remove the old terminator
  rewriter.eraseOp(inParallelOp);

  // 7. Generate yields for outer loops (inner loop results become yielded
  // values)
  for (size_t i = loops.size() - 1; i > 0; --i) {
    auto innerLoop = cast<scf::ForOp>(loops[i]);
    auto outerLoop = cast<scf::ForOp>(loops[i - 1]);

    rewriter.setInsertionPointToEnd(outerLoop.getBody());
    rewriter.create<scf::YieldOp>(loc, innerLoop.getResults());
  }

  DictionaryAttr forallAttrs = forallOp->getAttrDictionary();
  loops.front()->setAttrs(forallAttrs);
  // 8. Replace Forall uses with the results of the outermost loop
  rewriter.replaceOp(forallOp, loops.front()->getResults());
  LLVM_DEBUG(llvm::dbgs() << "[ForallToFor] Transformation complete.\n");
  return success();
}
/// OpRewritePattern that wraps the above helper.
struct ForallToForPattern : public OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForallOp forallOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Operation *> generatedLoops;
    if (failed(rewriteForallToFor(rewriter, forallOp, generatedLoops)))
      return failure();
    return success();
  }
};

// --- Pass Definition ---

struct LoopUnrollStagePass
    : public mlir::dicp::LinalgExt::impl::LoopUnrollStageBase<
          LoopUnrollStagePass> {
  using LoopUnrollStageBase::LoopUnrollStageBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    LDBG("--- Starting LoopUnrollStagePass on @" << func.getName() << " ---");

    // ---------------------------------------------------------
    // Phase 1: Normalize scf.forall -> scf.for
    // ---------------------------------------------------------
    {
      RewritePatternSet patterns(ctx);
      patterns.add<ForallToForPattern>(ctx);

      // We use GreedyPatternRewriteDriver to handle potential nesting
      // and ensure the IR is in a stable state before collection.
      if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
    // ---------------------------------------------------------
    // Phase 2: Collect Candidate Loops
    // ---------------------------------------------------------
    // We collect loops into a vector first. If we unroll while walking,
    // we risk invalidating the iterator or missing nested loops.
    // func.walk defaults to PostOrder, which is ideal (inner loops first).
    SmallVector<scf::ForOp> candidateLoops;
    func.walk([&](scf::ForOp forOp) {
      if (forOp->hasAttr(kNPUStageAttrName)) {
        candidateLoops.push_back(forOp);
      }
    });

    if (candidateLoops.empty()) {
      LDBG("No scf.for loops found with stage attributes.");
      cleanupInternalAttributes(func);
      return;
    }

    LDBG("Collected " << candidateLoops.size() << " candidate scf.for loops.");
    return;
    // ---------------------------------------------------------
    // Phase 3: Unroll Loops
    // ---------------------------------------------------------
    int successCount = 0;

    for (auto [index, forOp] : llvm::enumerate(candidateLoops)) {
      if (failed(loopUnrollFull(forOp)))
        return forOp.emitError("Failed to unroll loop ") << index,
               signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::dicp::LinalgExt::createLoopUnrollStagePass() {
  return std::make_unique<LoopUnrollStagePass>();
}