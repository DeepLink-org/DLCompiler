#include "dicp/AscendLegalize/AscendLegalizePass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "ascend-legalize"

using namespace mlir;
using namespace triton;

// Flip cmpi predicates that are incompatible with MaskState::parseCmp.
//
// MaskState::parseCmp requires lhs = tensor (has range/offset) and rhs =
// scalar (splat constant). When the IR produces sge(scalar, tensor) instead,
// this pattern rewrites it to the mathematically equivalent
// sle(tensor, scalar) by swapping operands and flipping the predicate.
//
// Supported flips:
//   sge(A, B) -> sle(B, A)
//   sgt(A, B) -> slt(B, A)
LogicalResult
FlipCmpiPredicatePattern::matchAndRewrite(arith::CmpIOp cmpOp,
                                          PatternRewriter &rewriter) const {
  auto predicate = cmpOp.getPredicate();

  // Only handle predicates that have a symmetric flip
  arith::CmpIPredicate flippedPredicate;
  switch (predicate) {
  case arith::CmpIPredicate::sge:
    flippedPredicate = arith::CmpIPredicate::sle;
    break;
  case arith::CmpIPredicate::sgt:
    flippedPredicate = arith::CmpIPredicate::slt;
    break;
  default:
    return failure();
  }

  auto lhs = cmpOp.getLhs();
  auto rhs = cmpOp.getRhs();

  // Only flip when lhs is a splat (scalar-like) and rhs is not a splat
  // (tensor-like with range/offset). This ensures the result satisfies
  // MaskState's constraint: lhs=tensor, rhs=scalar.
  auto lhsSplat = lhs.getDefiningOp<triton::SplatOp>();
  if (!lhsSplat)
    return failure();

  auto rhsSplat = rhs.getDefiningOp<triton::SplatOp>();
  if (rhsSplat)
    return failure();

  // Swap operands and flip the predicate
  auto newCmp = rewriter.create<arith::CmpIOp>(cmpOp.getLoc(), flippedPredicate,
                                               rhs, lhs);
  rewriter.replaceOp(cmpOp, newCmp.getResult());
  return success();
}

void AscendLegalizePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  RewritePatternSet patterns(&getContext());
  patterns.add<FlipCmpiPredicatePattern>(patterns.getContext());

  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    moduleOp->emitError("failed to apply ascend-legalize patterns");
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> triton::createAscendLegalizePass() {
  return std::make_unique<AscendLegalizePass>();
}
