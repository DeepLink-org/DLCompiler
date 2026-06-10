#include "dicp/AscendLegalize/AscendLegalizePass.h"

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

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

static bool isNonNegativeIntegerConstant(Value value, unsigned depth = 0) {
  if (depth > 4)
    return false;

  if (auto splatOp = value.getDefiningOp<triton::SplatOp>())
    return isNonNegativeIntegerConstant(splatOp.getSrc(), depth + 1);

  if (auto bitcastOp = value.getDefiningOp<arith::BitcastOp>())
    return isNonNegativeIntegerConstant(bitcastOp.getIn(), depth + 1);

  auto constantOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return false;

  Attribute attr = constantOp.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return !intAttr.getValue().isNegative();

  if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(attr)) {
    return llvm::all_of(denseAttr.getValues<APInt>(),
                        [](const APInt &value) { return !value.isNegative(); });
  }

  return false;
}

static bool isBoolExtSumReduce(triton::ReduceOp reduceOp) {
  if (reduceOp.getSrcs().size() != 1)
    return false;

  auto extUIOp = reduceOp.getSrcs()[0].getDefiningOp<arith::ExtUIOp>();
  if (!extUIOp)
    return false;

  auto boolSrcType = dyn_cast<RankedTensorType>(extUIOp.getOperand().getType());
  if (!boolSrcType || !boolSrcType.getElementType().isInteger(1))
    return false;

  Block &body = reduceOp.getCombineOp().front();
  if (body.getNumArguments() != 2)
    return false;

  auto termOp = dyn_cast<triton::ReduceReturnOp>(body.getTerminator());
  if (!termOp || termOp.getOperands().size() != 1)
    return false;

  auto addOp = termOp.getOperands()[0].getDefiningOp<arith::AddIOp>();
  if (!addOp)
    return false;

  Value lhs = addOp.getLhs();
  Value rhs = addOp.getRhs();
  return (lhs == body.getArgument(0) && rhs == body.getArgument(1)) ||
         (lhs == body.getArgument(1) && rhs == body.getArgument(0));
}

static std::optional<arith::CmpIPredicate>
getSignedPredicate(arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::ugt:
    return arith::CmpIPredicate::sgt;
  case arith::CmpIPredicate::uge:
    return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::ult:
    return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::ule:
    return arith::CmpIPredicate::sle;
  default:
    return std::nullopt;
  }
}

/// Pattern: Flip cmpi predicates that are incompatible with
/// MaskState::parseCmp.
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

    auto newCmp = rewriter.create<arith::CmpIOp>(cmpOp.getLoc(),
                                                 flippedPredicate, rhs, lhs);
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
    rewriter.replaceOpWithNewOp<arith::MaximumFOp>(op, op.getType(),
                                                   op.getLhs(), op.getRhs());
    return success();
  }
};

/// Pattern: Give masked tt.load an explicit zero `other` operand when it is
/// omitted.
///
/// The TA Triton frontend defaults masked tl.load(..., mask=...) to
/// care_padding=True, which lowers to tt.load(ptr, mask, zero). The upstream
/// Triton 3.5 frontend leaves `other` absent instead. Ascend's downstream
/// unstructure/linalg lowering uses the explicit zero-fill operand to preserve
/// masked lanes, especially for dot operands where it becomes zero-padding
/// slices. Normalize here so both frontends feed the same IR shape to DICP.
struct AddZeroOtherToMaskedLoadPattern
    : public OpRewritePattern<triton::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    if (!loadOp.getMask() || loadOp.getOther())
      return failure();

    auto zeroAttr = rewriter.getZeroAttr(loadOp.getType());
    if (!zeroAttr)
      return failure();

    Location loc = loadOp.getLoc();
    Value zeroOther =
        rewriter.create<arith::ConstantOp>(loc, loadOp.getType(), zeroAttr);
    auto newLoad = rewriter.create<triton::LoadOp>(
        loc, loadOp.getPtr(), loadOp.getMask(), zeroOther,
        loadOp.getBoundaryCheck(), loadOp.getPadding(), loadOp.getCache(),
        loadOp.getEvict(), loadOp.getIsVolatile());

    for (auto attr : loadOp->getAttrs()) {
      if (!newLoad->hasAttr(attr.getName()))
        newLoad->setAttr(attr.getName(), attr.getValue());
    }

    rewriter.replaceOp(loadOp, newLoad.getResult());
    return success();
  }
};

/// Pattern: Rewrite unsigned comparisons on bool-sum counts to signed
/// comparisons.
///
/// Triton 3.5's tl.sum defaults integer inputs narrower than i32 to an i32
/// accumulator with the same signedness. For bool input, that produces the IR
/// shape:
///
///   arith.extui i1 -> i32
///   tt.reduce(add i32)
///   arith.cmpi ugt/uge/ult/ule reduce, non_negative_constant
///
/// TA keeps the same kernel on a bool/signed-friendly path, while Ascend
/// BiSheng HIR can fail on the unsigned route with an unsupported
/// uint32_t_to_uint64_t_rintmode vcast. The reduced value here is a count of
/// bool lanes, so it is non-negative and bounded by the reduction axis. For
/// non-negative signed constants, signed and unsigned comparisons are
/// equivalent. Canonicalize only this proven bool-count pattern and leave
/// ordinary unsigned integer comparisons untouched.
struct BoolSumUnsignedCmpToSignedPattern
    : public OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp cmpOp,
                                PatternRewriter &rewriter) const override {
    auto signedPredicate = getSignedPredicate(cmpOp.getPredicate());
    if (!signedPredicate)
      return failure();

    if (!isNonNegativeIntegerConstant(cmpOp.getRhs()))
      return failure();

    auto reduceOp = cmpOp.getLhs().getDefiningOp<triton::ReduceOp>();
    if (!reduceOp || !isBoolExtSumReduce(reduceOp))
      return failure();

    auto newCmp = rewriter.create<arith::CmpIOp>(
        cmpOp.getLoc(), *signedPredicate, cmpOp.getLhs(), cmpOp.getRhs());
    rewriter.replaceOp(cmpOp, newCmp.getResult());
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
    patterns.add<AddZeroOtherToMaskedLoadPattern>(patterns.getContext());
    patterns.add<BoolSumUnsignedCmpToSignedPattern>(patterns.getContext());

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
