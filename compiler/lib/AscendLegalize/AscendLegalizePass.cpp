#include "dicp/AscendLegalize/AscendLegalizePass.h"

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define DEBUG_TYPE "ascend-legalize"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ASCENDLEGALIZE
#include "dicp/AscendLegalize/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

/// Pattern: Flip cmpi predicates that are incompatible with MaskState::parseCmp.
///
/// MaskState::parseCmp requires lhs = tensor (has range/offset) and rhs =
/// scalar (splat constant). When the IR produces sge(scalar, tensor) instead,
/// this pattern rewrites it to the mathematically equivalent
/// sle(tensor, scalar) by swapping operands and flipping the predicate.
///
/// Supported flips:
///   sge(A, B) -> sle(B, A)
///   sgt(A, B) -> slt(B, A)
struct FlipCmpiPredicatePattern : public OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp cmpOp,
                                PatternRewriter &rewriter) const override {
    auto predicate = cmpOp.getPredicate();

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

    auto lhsSplat = lhs.getDefiningOp<triton::SplatOp>();
    if (!lhsSplat)
      return failure();

    auto rhsSplat = rhs.getDefiningOp<triton::SplatOp>();
    if (rhsSplat)
      return failure();

    auto newCmp = rewriter.create<arith::CmpIOp>(cmpOp.getLoc(), flippedPredicate,
                                                 rhs, lhs);
    rewriter.replaceOp(cmpOp, newCmp.getResult());
    return success();
  }
};

/// Pattern: Replace arith::MaxNumFOp (NaN-quiet) with arith::MaximumFOp
/// (NaN-propagating).
///
/// On Ascend NPU, online-softmax reductions need NaN propagation so that
/// m_ij = max(m_i, max(qk)) correctly propagates NaN through the reduce
/// region. MaxNumFOp silently swallows NaN, leading to wrong exp() results.
struct MaxNumFToMaximumFPattern : public OpRewritePattern<arith::MaxNumFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MaxNumFOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::MaximumFOp>(op, op.getType(), op.getLhs(),
                                                    op.getRhs());
    return success();
  }
};

/// The pass: AscendLegalizePass
struct AscendLegalizePass
    : public ::mlir::triton::impl::AscendLegalizeBase<AscendLegalizePass> {
  using AscendLegalizeBase::AscendLegalizeBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<FlipCmpiPredicatePattern>(patterns.getContext());

    // On Ascend NPU, replace arith::MaxNumFOp (NaN-quiet) with
    // arith::MaximumFOp (NaN-propagating) so that online-softmax
    // reductions propagate NaN correctly through the reduce region.
    if (auto targetAttr =
            moduleOp->getAttrOfType<hacc::TargetAttr>(hacc::TargetAttr::name)) {
      patterns.add<MaxNumFToMaximumFPattern>(patterns.getContext());
    }

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      moduleOp->emitError("failed to apply ascend-legalize patterns");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createAscendLegalizePass() {
  return std::make_unique<AscendLegalizePass>();
}
