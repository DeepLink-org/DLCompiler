#include "dicp/Dialect/LinalgExt/Transforms/Transforms.h"

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#define DEBUG_TYPE "remove-single-iteration"

using namespace mlir;
using namespace dicp;
using namespace LinalgExt;

namespace mlir::dicp::LinalgExt {

/**
 * Checks if the scf::ForOp loop body will never execute a second iteration.
 *
 * This holds true if:
 * 1. The upper bound (ub) is less than or equal to the step size. (ub <= step)
 * 2. The lower bound (lb) is non-negative. (lb >= 0)
 *
 * If lb >= 0 and ub <= step, then lb + step >= ub, guaranteeing termination
 * after the first iteration, assuming the loop runs at least once.
 *
 * @param op The scf::ForOp to analyze.
 * @return True if the loop runs at most one time.
 */
static bool neverRunsSecondIteration(scf::ForOp op) {
  // Can't perform the analysis if the loops's bounds aren't index-typed.
  if (!op.getInductionVar().getType().isIndex())
    return false;
  // If the upper bound (ub) is less than or equal to the loop step, then
  // lower bound + step must be greater than the upper bound, assuming the
  // lower bound is non-negative.
  FailureOr<bool> isUbUnderStep = ValueBoundsConstraintSet::compare(
      getAsOpFoldResult(op.getUpperBound()), ValueBoundsConstraintSet::LE,
      getAsOpFoldResult(op.getStep()));
  FailureOr<bool> isLbNonNegative = ValueBoundsConstraintSet::compare(
      getAsOpFoldResult(op.getLowerBound()), ValueBoundsConstraintSet::GE,
      getAsIndexOpFoldResult(op.getContext(), 0));
  return isUbUnderStep.value_or(false) && isLbNonNegative.value_or(false);
}

// ---

/**
 * Checks if the scf::ForOp loop body is guaranteed to execute at least once.
 *
 * This holds true if the lower bound (lb) is strictly less than the
 * upper bound (ub). (lb < ub)
 *
 * @param op The scf::ForOp to analyze.
 * @return True if the loop is guaranteed to run the first iteration.
 */
static bool alwaysRunsFirstIteration(scf::ForOp op) {
  // Can't perform the analysis if the loops's bounds aren't index-typed.
  if (!op.getInductionVar().getType().isIndex())
    return false;
  FailureOr<bool> isLb = ValueBoundsConstraintSet::compare(
      getAsOpFoldResult(op.getLowerBound()), ValueBoundsConstraintSet::LT,
      getAsOpFoldResult(op.getUpperBound()));
  return isLb.value_or(false);
}

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(PatternRewriter &rewriter, scf::ForOp op,
                                ValueRange blockArgs = {}) {
  Block *block = op.getBody();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

/// Same as `replaceOpWithRegion` function but within an scf.if region.
static void replaceForWithIf(PatternRewriter &rewriter, scf::ForOp op,
                             ValueRange blockArgs = {}) {
  Block *block = op.getBody();
  ValueRange initArgs = op.getInitArgs();
  Value count =
      rewriter.create<arith::CmpIOp>(op->getLoc(), arith::CmpIPredicate::sgt,
                                     op.getUpperBound(), op.getLowerBound());
  auto ifOp =
      rewriter.create<scf::IfOp>(op->getLoc(), op.getResultTypes(), count,
                                 /*withElseRegion=*/initArgs.size() != 0);
  Operation *terminator = block->getTerminator();
  rewriter.inlineBlockBefore(block, &ifOp.getThenRegion().front(),
                             ifOp.getThenRegion().front().begin(), blockArgs);
  if (initArgs.size() == 0) {
    rewriter.eraseOp(terminator);
  } else {
    rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
    rewriter.create<scf::YieldOp>(ifOp.getLoc(), initArgs);
  }
  rewriter.replaceOp(op, ifOp);
}

namespace {
/// Rewriting pattern that replaces single-iteration loops with their bodies.
struct SimplifyTrivialLoops : public OpRewritePattern<scf::ForOp> {

  SimplifyTrivialLoops(MLIRContext *context, ForControlFnRef controlFn)
      : OpRewritePattern(context), controlFn(controlFn) {}

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (controlFn && !controlFn(op)) {
      return rewriter.notifyMatchFailure(
          op, "doesn't match according to the the control function");
    }
    if (!neverRunsSecondIteration(op)) {
      return rewriter.notifyMatchFailure(op,
                                         "is not a single-iteration for loop");
    }
    // The second iteration is never run so the loop atmost can have 1
    // iteration. Inline its body and remove the loop.
    SmallVector<Value> blockArgs;
    blockArgs.reserve(op.getInitArgs().size() + 1);
    blockArgs.push_back(op.getLowerBound());
    llvm::append_range(blockArgs, op.getInitArgs());
    if (alwaysRunsFirstIteration(op)) {
      replaceOpWithRegion(rewriter, op, blockArgs);
    } else {
      replaceForWithIf(rewriter, op, blockArgs);
    }
    return success();
  }

private:
  ForControlFnRef controlFn;
};

} // namespace

void populateRemoveSingleIterationLoopPattern(RewritePatternSet &patterns,
                                              ForControlFnRef controlFn) {
  patterns.add<SimplifyTrivialLoops>(patterns.getContext(), controlFn);
}

} // namespace mlir::dicp::LinalgExt
