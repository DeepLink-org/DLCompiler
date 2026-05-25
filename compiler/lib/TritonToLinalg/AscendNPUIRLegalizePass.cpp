

#include "dicp/TritonToLinalg/AscendNPUIRLegalizePass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace triton;

struct ReifyUnsignedMulViaSignedPattern
    : public OpRewritePattern<arith::MulUIExtendedOp> {
  bool unsafeMode;

  ReifyUnsignedMulViaSignedPattern(MLIRContext *ctx, bool unsafeMode)
      : OpRewritePattern<arith::MulUIExtendedOp>(ctx, /*benefit=*/1),
        unsafeMode(unsafeMode) {}

  LogicalResult matchAndRewrite(arith::MulUIExtendedOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    // Only handle i32 element type
    Type elemType;
    if (auto shapedType = dyn_cast<ShapedType>(lhs.getType())) {
      if (!shapedType.getElementType().isInteger(32))
        return failure();
      elemType = shapedType.getElementType();
    } else {
      if (!lhs.getType().isInteger(32))
        return failure();
      elemType = lhs.getType();
    }

    auto mulsiOp = rewriter.create<arith::MulSIExtendedOp>(
        op.getLoc(), op.getResultTypes(), lhs, rhs);

    if (unsafeMode) {
      rewriter.replaceOp(op, {mulsiOp.getLow(), mulsiOp.getHigh()});
      return success();
    }

    // high_unsigned = high_signed + (a >> 31) * b + (b >> 31) * a
    // Build a constant matching the operand type (scalar or tensor).
    auto makeConst = [&](int64_t val) -> Value {
      if (auto shaped = dyn_cast<ShapedType>(lhs.getType()))
        return rewriter.create<arith::ConstantOp>(
            op.getLoc(), shaped,
            DenseElementsAttr::get(
                shaped, APInt(shaped.getElementTypeBitWidth(), val)));
      return rewriter.create<arith::ConstantOp>(
          op.getLoc(), rewriter.getIntegerAttr(elemType, val));
    };
    Value c31 = makeConst(31);
    Value sA = rewriter.create<arith::ShRUIOp>(op.getLoc(), lhs, c31);
    Value sB = rewriter.create<arith::ShRUIOp>(op.getLoc(), rhs, c31);
    Value corrA = rewriter.create<arith::MulIOp>(op.getLoc(), sA, rhs);
    Value corrB = rewriter.create<arith::MulIOp>(op.getLoc(), sB, lhs);
    Value tmp =
        rewriter.create<arith::AddIOp>(op.getLoc(), mulsiOp.getHigh(), corrA);
    Value highUnsigned =
        rewriter.create<arith::AddIOp>(op.getLoc(), tmp, corrB);

    rewriter.replaceOp(op, {mulsiOp.getLow(), highUnsigned});
    return success();
  }
};

void AscendNPUIRLegalizePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<ReifyUnsignedMulViaSignedPattern>(&getContext(), unsafeMode);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
triton::createAscendNPUIRLegalizePass() {
  return std::make_unique<AscendNPUIRLegalizePass>();
}

std::unique_ptr<OperationPass<ModuleOp>> triton::createAscendNPUIRLegalizePass(
    const AscendNPUIRLegalizeOptions &options) {
  return std::make_unique<AscendNPUIRLegalizePass>(options);
}
