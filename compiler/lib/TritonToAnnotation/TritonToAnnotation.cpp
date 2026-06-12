

#include "dicp/TritonToAnnotation/Passes.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "dicp/Dialect/TritonDicp/IR/TritonDicpDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONTOANNOTATION
#include "dicp/TritonToAnnotation/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {
struct TritonToAnnotationPass
    : public mlir::triton::impl::TritonToAnnotationBase<
          TritonToAnnotationPass> {
  void runOnOperation() override;
};
} // namespace

struct TritonAnnotationConversionPattern
    : OpRewritePattern<mlir::triton::dicp::AnnotationOp> {
  using OpRewritePattern<mlir::triton::dicp::AnnotationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::triton::dicp::AnnotationOp op,
                                PatternRewriter &rewriter) const final {
    auto markOp = rewriter.create<annotation::MarkOp>(op.getLoc(), op.getSrc());
    // Forward all annotations.
    markOp->setAttrs(op->getAttrs());
    rewriter.eraseOp(op);
    return success();
  }
};

void TritonToAnnotationPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<annotation::AnnotationDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<TritonAnnotationConversionPattern>(patterns.getContext());
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createTritonToAnnotationPass() {
  return std::make_unique<TritonToAnnotationPass>();
}
