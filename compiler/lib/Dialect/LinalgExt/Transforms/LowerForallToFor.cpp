#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lower-forall-to-for"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;

namespace mlir::dicp::LinalgExt {
#define GEN_PASS_DEF_LOWERFORALLTOFOR
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace mlir::dicp::LinalgExt

namespace {

//===----------------------------------------------------------------------===//
// ForallToFor rewrite logic
//===----------------------------------------------------------------------===//

/// Rewrites a single `scf.forall` into a nest of `scf.for` loops with
/// sequential accumulation via `tensor.insert_slice`.  Attributes on the
/// original forall are preserved on the outermost generated for.
static void createOuterLoopYields(PatternRewriter &rewriter, Location loc,
                                  ArrayRef<scf::ForOp> loops) {
  for (size_t i = loops.size() - 1; i > 0; --i) {
    scf::ForOp innerLoop = loops[i];
    scf::ForOp outerLoop = loops[i - 1];
    rewriter.setInsertionPointToEnd(outerLoop.getBody());
    rewriter.create<scf::YieldOp>(loc, innerLoop.getResults());
  }
}

static LogicalResult rewriteForallToFor(PatternRewriter &rewriter,
                                        scf::ForallOp forallOp) {
  Location loc = forallOp.getLoc();
  SmallVector<scf::ForOp> loops;

  // 1. Gather bounds and steps.
  SmallVector<Value> lbs = forallOp.getLowerBound(rewriter);
  SmallVector<Value> ubs = forallOp.getUpperBound(rewriter);
  SmallVector<Value> steps = forallOp.getStep(rewriter);

  // 2. Prepare init args from shared_outs.
  SmallVector<Value> currentIterOperands(forallOp.getOutputs().begin(),
                                         forallOp.getOutputs().end());

  LDBG("[Lower] build scf.for nest of rank " << lbs.size());

  // 3. Create the loop nest.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forallOp);

  for (auto [lb, ub, step] : llvm::zip(lbs, ubs, steps)) {
    scf::ForOp loop =
        rewriter.create<scf::ForOp>(loc, lb, ub, step, currentIterOperands);

    if (!loop.getBody()->empty())
      rewriter.eraseOp(loop.getBody()->getTerminator());

    currentIterOperands.assign(loop.getRegionIterArgs().begin(),
                               loop.getRegionIterArgs().end());
    rewriter.setInsertionPointToStart(loop.getBody());
    loops.push_back(loop);
  }

  if (loops.empty())
    return forallOp.emitError("expected rank > 0 for scf.forall");

  scf::ForOp innermostLoop = loops.back();

  // 4. Inline the forall body.
  Block *forallBody = forallOp.getBody();
  Block *innermostBlock = innermostLoop.getBody();

  IRMapping mapping;

  // Map induction variables.
  SmallVector<Value> ivs = llvm::map_to_vector(
      loops, [](scf::ForOp loop) { return loop.getInductionVar(); });

  for (auto [forallArg, newIv] :
       llvm::zip(forallBody->getArguments().take_front(lbs.size()), ivs))
    mapping.map(forallArg, newIv);

  // Map shared outputs to innermost iter_args.
  for (auto [forallArg, iterArg] :
       llvm::zip(forallBody->getArguments().drop_front(lbs.size()),
                 innermostLoop.getRegionIterArgs()))
    mapping.map(forallArg, iterArg);

  SmallVector<Value> mappedArgs;
  mappedArgs.reserve(forallBody->getNumArguments());
  for (Value arg : forallBody->getArguments())
    mappedArgs.push_back(mapping.lookup(arg));

  rewriter.mergeBlocks(forallBody, innermostBlock, mappedArgs);

  // 5. Handle the terminator (scf.forall.in_parallel).
  auto inParallelOp =
      dyn_cast<scf::InParallelOp>(innermostBlock->getTerminator());
  if (!inParallelOp)
    return forallOp.emitError("expected scf.forall.in_parallel terminator");

  LDBG("[Lower] convert parallel_insert_slice to sequential insert_slice");

  // Convert parallel updates to sequential accumulation.
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

  // 6. Yield the accumulated results.
  SmallVector<Value> yieldOperands;
  for (Value iterArg : innermostLoop.getRegionIterArgs())
    yieldOperands.push_back(accumulatorMap[iterArg]);

  rewriter.create<scf::YieldOp>(loc, yieldOperands);
  rewriter.eraseOp(inParallelOp);

  // 7. Generate yields for outer loops.
  createOuterLoopYields(rewriter, loc, loops);

  // Preserve original attributes and replace the forall.
  loops.front()->setAttrs(forallOp->getAttrDictionary());
  rewriter.replaceOp(forallOp, loops.front().getResults());

  LDBG("[Lower] forall-to-for rewrite completed");
  return success();
}

/// Pattern wrapper for the ForallToFor rewrite.
struct ForallToForPattern : public OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForallOp forallOp,
                                PatternRewriter &rewriter) const override {
    return rewriteForallToFor(rewriter, forallOp);
  }
};

//===----------------------------------------------------------------------===//
// Pass: LowerForallToFor
//===----------------------------------------------------------------------===//

struct LowerForallToForPass
    : public mlir::dicp::LinalgExt::impl::LowerForallToForBase<
          LowerForallToForPass> {
  using LowerForallToForBase::LowerForallToForBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    LDBG("[Pass] LowerForallToFor on @" << func.getName());

    RewritePatternSet patterns(ctx);
    patterns.add<ForallToForPattern>(ctx);

    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::dicp::LinalgExt::createLowerForallToForPass() {
  return std::make_unique<LowerForallToForPass>();
}
