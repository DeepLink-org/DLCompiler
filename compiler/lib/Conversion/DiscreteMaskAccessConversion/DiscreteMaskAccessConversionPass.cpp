#include "dicp/Conversion/DiscreteMaskAccessConversion/MaskAnalysis.h"
#include "dicp/Conversion/DiscreteMaskAccessConversion/Passes.h"

#include "dicp/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

bool compileOn91095Flag = false;
bool forceSimtTemplateFlag = false;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DISCRETEMASKACCESSCONVERSION
#include "dicp/Conversion/DiscreteMaskAccessConversion/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::dicp;

LogicalResult isDiscreteMask(Operation *op, Value mask,
                             PatternRewriter &rewriter) {
  if (!mask)
    return failure();

  mlir::dicp::MaskState mstate;
  auto isContMask = mstate.parse(mask, op->getLoc(), rewriter);
  if (!isContMask.failed()) {
    mstate.eraseInsertedOps(op, rewriter);
    return failure();
  }
  return success();
}

struct DiscreteMaskStoreConversion : OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::StoreOp op,
                                PatternRewriter &rewriter) const final {
    auto mask = op.getMask();
    auto loc = op.getLoc();
    auto dst = op.getPtr();
    auto src = op.getValue();

    if (failed(isDiscreteMask(op, mask, rewriter)))
      return failure();

    auto loadFromDstOp = rewriter.create<triton::LoadOp>(
        loc, dst, op.getCache(), op.getEvict(), false);

    auto selOp = rewriter.create<arith::SelectOp>(loc, mask, src,
                                                  loadFromDstOp.getResult());
    auto newStore = rewriter.create<triton::StoreOp>(
        loc, dst, selOp, op.getCache(), op.getEvict());
    newStore->setAttr(mlir::dicp::discreteMaskAttrName,
                      UnitAttr::get(rewriter.getContext()));
    rewriter.replaceOp(op, newStore);
    return success();
  }
};

struct DiscreteMaskLoadConversion : OpRewritePattern<triton::LoadOp> {
  using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::LoadOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto other = op.getOther();
    auto mask = op.getMask();
    auto ptr = op.getPtr();

    if (failed(isDiscreteMask(op, mask, rewriter)))
      return failure();
    if (compileOn91095Flag && forceSimtTemplateFlag)
      return failure();

    if (!other) {
      FailureOr<Value> constant = specializeTypelessValueToConstant(
          TypelessValue::Zero, ptr.getType(), loc, rewriter);
      // TODO: fix me
      if (failed(constant)) {
        ptr.getType().dump();
        op->emitRemark() << " Unsupported type for constant creation";
        return failure();
      }
      other = *constant;
    }

    auto newLoadOp = rewriter.create<triton::LoadOp>(
        loc, ptr, op.getCache(), op.getEvict(), op.getIsVolatile());
    auto discreteMaskOp =
        rewriter.create<arith::SelectOp>(loc, mask, newLoadOp, other);
    rewriter.replaceOp(op, discreteMaskOp);
    return success();
  }
};

struct DiscreteMaskAtomicConversion : OpRewritePattern<triton::AtomicRMWOp> {
  using OpRewritePattern<triton::AtomicRMWOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::AtomicRMWOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto ptr = op.getPtr();
    auto src = op.getVal();
    auto mask = op.getMask();
    auto rmwOp = op.getAtomicRmwOp();

    if (failed(isDiscreteMask(op, mask, rewriter)))
      return failure();

    const std::map<RMWOp, TypelessValue> initMap = {
        {RMWOp::FADD, TypelessValue::Zero},
        {RMWOp::ADD, TypelessValue::Zero},
        {RMWOp::UMAX, TypelessValue::Zero},
        {RMWOp::OR, TypelessValue::Zero},
        {RMWOp::MIN, TypelessValue::Max},
        {RMWOp::UMIN, TypelessValue::Max},
        {RMWOp::AND, TypelessValue::Max},
        {RMWOp::MAX, TypelessValue::Min},
        {RMWOp::XOR, TypelessValue::Zero},
        {RMWOp::XCHG, TypelessValue::Undefined},
    };
    assert(initMap.find(rmwOp) != initMap.end());
    auto typelessVal = initMap.at(rmwOp);
    if (typelessVal == TypelessValue::Undefined) {
      // Undefined default value atomic op will be decomposed in AscendNPU-IR
      op->setAttr(mlir::dicp::discreteMaskAttrName,
                  UnitAttr::get(rewriter.getContext()));
      return failure();
    }

    FailureOr<mlir::Value> fill = specializeTypelessValueToConstant(
        typelessVal, src.getType(), loc, rewriter);
    if (failed(fill))
      op->emitError("Unsupported atomic operation.");

    auto maskedValue = rewriter.create<arith::SelectOp>(loc, mask, src, *fill);
    auto newAtomicOp = rewriter.create<triton::AtomicRMWOp>(
        loc, src.getType(), rmwOp, ptr, maskedValue, mlir::Value(), op.getSem(),
        op.getScope());
    rewriter.replaceOp(op, newAtomicOp);
    return success();
  }
};

struct DiscreteMaskAccessConversionPass
    : mlir::triton::impl::DiscreteMaskAccessConversionBase<
          DiscreteMaskAccessConversionPass> {

  DiscreteMaskAccessConversionPass(
      const DiscreteMaskAccessConversionOptions &options)
      : DiscreteMaskAccessConversionBase(options) {}

  void runOnOperation() override {
    compileOn91095Flag = this->compileOn91095;
    forceSimtTemplateFlag = this->forceSimtTemplate;

    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<DiscreteMaskLoadConversion, DiscreteMaskStoreConversion,
                 DiscreteMaskAtomicConversion>(patterns.getContext());
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      moduleOp->emitError("failed to apply discrete mask access patterns");
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createDiscreteMaskAccessConversionPass(
    const DiscreteMaskAccessConversionOptions &options) {
  return std::make_unique<DiscreteMaskAccessConversionPass>(options);
}
