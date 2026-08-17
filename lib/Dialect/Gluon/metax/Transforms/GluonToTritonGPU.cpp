/*
 * 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
 * Reserved.
 */
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h"

#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gluon-to-tritongpu"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[gluon-to-ttg] " << X << "\n")

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace gd = mlir::triton::gluon;

namespace mlir::triton::gluon {
#define GEN_PASS_DEF_GLUONTOTRITONGPUCONVERSIONPASS
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h.inc"
} // namespace mlir::triton::gluon

namespace {

constexpr StringLiteral kStoreCoalesceAttr = "ttg.gluon.store-coalesce";

struct PhysicalViewIndices {
  SmallVector<int64_t> cta;
  SmallVector<int64_t> element;
};

// `maca::calExtractTensorIdx` is the existing C500 physical-view definition
// used by the MetaX lowering.  Gluon owns only the logical slice operation;
// once layouts are concrete, this pass delegates CTA/register selection to
// that established implementation and does not add conversion carriers.
static FailureOr<PhysicalViewIndices>
deriveDirectViewIndices(RankedTensorType fullType, RankedTensorType subType,
                        ArrayRef<int64_t> offsets) {
  if (!fullType || !subType || fullType.getRank() != 2 ||
      subType.getRank() != 2 || offsets.size() != 2)
    return failure();

  Attribute fullEncoding = fullType.getEncoding();
  Attribute subEncoding = subType.getEncoding();
  bool sameEncodingClass =
      (isa<ttg::BlockedEncodingAttr>(fullEncoding) &&
       isa<ttg::BlockedEncodingAttr>(subEncoding)) ||
      (isa<ttg::MACAMmaEncodingAttr>(fullEncoding) &&
       isa<ttg::MACAMmaEncodingAttr>(subEncoding)) ||
      (isa<ttg::DotOperandEncodingAttr>(fullEncoding) &&
       isa<ttg::DotOperandEncodingAttr>(subEncoding));
  if (!sameEncodingClass)
    return failure();

  SmallVector<int64_t, 2> logicalSubTensorIndex;
  for (auto [fullDim, subDim, offset] :
       llvm::zip(fullType.getShape(), subType.getShape(), offsets)) {
    if (ShapedType::isDynamic(fullDim) || ShapedType::isDynamic(subDim) ||
        fullDim <= 0 || subDim <= 0 || offset < 0 || subDim > fullDim ||
        offset > fullDim - subDim || offset % subDim != 0)
      return failure();
    logicalSubTensorIndex.push_back(offset / subDim);
  }

  auto [cta, element] =
      calExtractTensorIdx(fullType, subType, logicalSubTensorIndex);
  return PhysicalViewIndices{std::move(cta), std::move(element)};
}

// `storeCoalesce` keeps a logical register view layout-parametric until this
// late lowering.  A complete [K, N] B panel split into N panels must keep the
// panel index in the parent register carrier; otherwise every 32-column view
// is misrepresented as a distinct CTA replica.  This is the same physical
// relationship the established `calExtractTensorIdx` implementation expects:
//
//   parent Blocked[..., sizePerThread[N] * panels]
//       -> child Blocked[..., sizePerThread[N]]
//
// We intentionally support only this two-dimensional B-panel form here.  M
// panels and accumulator views retain their concrete type and are handled by
// their own existing lowering contracts.
static FailureOr<RankedTensorType>
deriveStoreCoalesceBPanelCarrier(RankedTensorType fullType,
                                 RankedTensorType subType) {
  auto blocked = fullType
                     ? dyn_cast<ttg::BlockedEncodingAttr>(fullType.getEncoding())
                     : nullptr;
  if (!blocked || !subType || fullType.getRank() != 2 ||
      subType.getRank() != 2 || fullType.getShape()[0] != subType.getShape()[0] ||
      fullType.getShape()[1] <= subType.getShape()[1] ||
      fullType.getShape()[1] % subType.getShape()[1] != 0)
    return failure();

  const int64_t panels = fullType.getShape()[1] / subType.getShape()[1];
  SmallVector<unsigned> sizePerThread(blocked.getSizePerThread());
  if (panels <= 1 ||
      panels > std::numeric_limits<unsigned>::max() / sizePerThread[1])
    return failure();
  sizePerThread[1] *= static_cast<unsigned>(panels);
  auto carrierEncoding = ttg::BlockedEncodingAttr::get(
      fullType.getContext(), sizePerThread, blocked.getThreadsPerWarp(),
      blocked.getWarpsPerCTA(), blocked.getOrder(), blocked.getCTALayout());
  return RankedTensorType::get(fullType.getShape(), fullType.getElementType(),
                               carrierEncoding);
}

// MACA accumulator views use the ownership axis selected by colMajor.  The
// full accumulator carries the N (colMajor=0) or M (colMajor=1) panel slot;
// each leaf dot continues to use its normal M0.  This is physical view
// materialization only: the logical Gluon slice contract remains same-layout
// until this late pass.
static FailureOr<RankedTensorType>
deriveStoreCoalesceAccumulatorCarrier(RankedTensorType fullType,
                                      RankedTensorType subType) {
  auto fullMma = fullType
                     ? dyn_cast<ttg::MACAMmaEncodingAttr>(fullType.getEncoding())
                     : nullptr;
  auto subMma = subType
                    ? dyn_cast<ttg::MACAMmaEncodingAttr>(subType.getEncoding())
                    : nullptr;
  if (!fullMma || !subMma || fullType.getRank() != 2 ||
      subType.getRank() != 2 || fullMma != subMma)
    return failure();

  const unsigned ownershipAxis = fullMma.getColMajor() ? 0 : 1;
  if (fullType.getShape()[ownershipAxis] <= subType.getShape()[ownershipAxis] ||
      fullType.getShape()[ownershipAxis] % subType.getShape()[ownershipAxis] !=
          0)
    return failure();
  const int64_t panels = fullType.getShape()[ownershipAxis] /
                         subType.getShape()[ownershipAxis];
  SmallVector<unsigned> elementsMNK(fullMma.getElementsMNK());
  const unsigned elementAxis = ownershipAxis == 0 ? 0 : 1;
  if (panels <= 1 || panels > std::numeric_limits<unsigned>::max() /
                                   elementsMNK[elementAxis])
    return failure();
  elementsMNK[elementAxis] *= static_cast<unsigned>(panels);
  auto carrierEncoding = ttg::MACAMmaEncodingAttr::get(
      fullType.getContext(), fullMma.getVersionMajor(),
      fullMma.getVersionMinor(), fullMma.getWarpsPerCTA(), elementsMNK,
      fullMma.getColMajor(), fullMma.getIsATrans(), fullMma.getIsBTrans(),
      fullMma.getElementsStride());
  return RankedTensorType::get(fullType.getShape(), fullType.getElementType(),
                               carrierEncoding);
}

class ExtractSlicePattern final : public OpRewritePattern<gd::ExtractSliceOp> {
public:
  ExtractSlicePattern(MLIRContext *context, bool storeCoalesce)
      : OpRewritePattern(context), storeCoalesce(storeCoalesce) {}

  LogicalResult matchAndRewrite(gd::ExtractSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto source = dyn_cast<RankedTensorType>(op.getSource().getType());
    auto result = dyn_cast<RankedTensorType>(op.getResult().getType());
    RankedTensorType physicalSource = source;
    if (storeCoalesce) {
      if (FailureOr<RankedTensorType> carrier =
              deriveStoreCoalesceBPanelCarrier(source, result);
          !failed(carrier))
        physicalSource = *carrier;
      else if (FailureOr<RankedTensorType> carrier =
                   deriveStoreCoalesceAccumulatorCarrier(source, result);
               !failed(carrier))
        physicalSource = *carrier;
    }
    // The logical slice result already owns the selected B0/M0 encoding.
    // Only a store-coalesce B parent uses a wider physical carrier; the
    // existing index helper derives the corresponding CTA/register indices.
    auto physicalResult = result;
    Value sourceValue = op.getSource();
    if (physicalSource && physicalSource != source) {
      rewriter.setInsertionPoint(op);
      sourceValue = rewriter.create<ttg::ConvertLayoutOp>(
          op.getLoc(), physicalSource, sourceValue);
    }
    auto ids = physicalSource && physicalResult
                   ? deriveDirectViewIndices(physicalSource, physicalResult,
                                             op.getOffsets())
                   : FailureOr<PhysicalViewIndices>(failure());
    if (failed(ids))
      return op.emitError()
             << "requires a complete rank-2 CTA/register partition; this "
                "Gluon slice cannot be represented by ttg.extract_tensor "
                "without a layout conversion";
    rewriter.replaceOpWithNewOp<ttg::ExtractTensorOp>(
        op, physicalResult, sourceValue,
        rewriter.getDenseI64ArrayAttr(ids->cta),
        rewriter.getDenseI64ArrayAttr(ids->element));
    LDBG("extract " << op.getLoc());
    return success();
  }

private:
  bool storeCoalesce;
};

class InsertSlicePattern final : public OpRewritePattern<gd::InsertSliceOp> {
public:
  InsertSlicePattern(MLIRContext *context, bool storeCoalesce)
      : OpRewritePattern(context), storeCoalesce(storeCoalesce) {}

  LogicalResult matchAndRewrite(gd::InsertSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto base = dyn_cast<RankedTensorType>(op.getBase().getType());
    auto update = dyn_cast<RankedTensorType>(op.getUpdate().getType());
    auto result = dyn_cast<RankedTensorType>(op.getResult().getType());
    RankedTensorType physicalBase = base;
    if (storeCoalesce) {
      if (FailureOr<RankedTensorType> carrier =
              deriveStoreCoalesceAccumulatorCarrier(base, update);
          !failed(carrier))
        physicalBase = *carrier;
    }

    Value baseValue = op.getBase();
    if (physicalBase && physicalBase != base) {
      if (auto convert = baseValue.getDefiningOp<ttg::ConvertLayoutOp>();
          convert && convert.getSrc().getType() == physicalBase)
        baseValue = convert.getSrc();
      else {
        rewriter.setInsertionPoint(op);
        baseValue = rewriter.create<ttg::ConvertLayoutOp>(
            op.getLoc(), physicalBase, baseValue);
      }
    }

    auto ids = physicalBase && update && result
                   ? deriveDirectViewIndices(physicalBase, update,
                                             op.getOffsets())
                   : FailureOr<PhysicalViewIndices>(failure());
    if (failed(ids) || base != result ||
        (!storeCoalesce && base.getEncoding() != update.getEncoding()))
      return op.emitError()
             << "requires a complete rank-2 CTA/register partition; this "
                "Gluon slice update cannot be represented by "
                "ttg.insert_tensor without a layout conversion";
    auto physicalInsert = rewriter.create<ttg::InsertTensorOp>(
        op.getLoc(), physicalBase, baseValue, op.getUpdate(),
        rewriter.getDenseI64ArrayAttr(ids->cta),
        rewriter.getDenseI64ArrayAttr(ids->element));
    if (physicalBase == result)
      rewriter.replaceOp(op, physicalInsert);
    else
      rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(
          op, result, physicalInsert.getResult());
    LDBG("insert " << op.getLoc());
    return success();
  }

private:
  bool storeCoalesce;
};

static bool supportsPhysicalBsmTypes(RankedTensorType source,
                                    RankedTensorType result) {
  if (!source || !result || source.getShape() != result.getShape() ||
      source.getEncoding() != result.getEncoding() ||
      !source.getElementType().isInteger(32) ||
      (!result.getElementType().isF16() &&
       !result.getElementType().isBF16()))
    return false;

  auto dot = dyn_cast<ttg::DotOperandEncodingAttr>(result.getEncoding());
  auto mma = dot ? dyn_cast<ttg::MACAMmaEncodingAttr>(dot.getParent()) : nullptr;
  if (!dot || !mma || dot.getOpIdx() != 1)
    return false;

  ArrayRef<unsigned> elements = mma.getElementsMNK();
  if (elements.size() != 3 || elements[1] == 0 || elements[2] == 0 ||
      elements[1] % 2 != 0 || elements[2] % 2 != 0)
    return false;

  // The existing LLVM pattern materializes one 32-slot split fragment.  Check
  // its actual register-index mapping instead of using eN*eK==32 as a proxy:
  // that product neither proves the source indices are valid nor that every
  // output slot is written exactly once.
  constexpr unsigned kPhysicalSlots = 32;
  if (ttg::getTotalElemsPerThread(source) != kPhysicalSlots ||
      ttg::getTotalElemsPerThread(result) != kPhysicalSlots)
    return false;

  const uint64_t elementsN = elements[1];
  const uint64_t elementsK = elements[2];
  const uint64_t elementsSize = elementsN * elementsK;
  SmallVector<unsigned, kPhysicalSlots> writes(kPhysicalSlots, 0);
  for (uint64_t j = 0; j < elementsSize / 2; j += 2 * elementsN) {
    for (uint64_t vector = 0; vector < elementsN / 2; ++vector) {
      const uint64_t input = (j / elementsN + vector) * elementsK;
      for (uint64_t index : {input, input + 1, input + elementsN,
                             input + elementsN + 1})
        if (index >= kPhysicalSlots)
          return false;

      const uint64_t output = 2 * vector + elementsN * (j / elementsK);
      for (uint64_t index :
           {output, output + 1, output + elementsK, output + elementsK + 1,
            output + 2 * elementsK, output + 2 * elementsK + 1,
            output + 3 * elementsK, output + 3 * elementsK + 1}) {
        if (index >= kPhysicalSlots)
          return false;
        ++writes[index];
      }
    }
  }
  return llvm::all_of(writes, [](unsigned count) { return count == 1; });
}

static bool isPhysicalBsm(ttg::BsmPermOp op) {
  auto source = dyn_cast<RankedTensorType>(op.getSrc1().getType());
  auto result = dyn_cast<RankedTensorType>(op.getResult().getType());
  auto sourceLoad = op.getSrc1().getDefiningOp<ttg::LocalLoadOp>();
  return sourceLoad && sourceLoad->hasOneUse() &&
         sourceLoad.getMmaMode() == 2 &&
         supportsPhysicalBsmTypes(source, result);
}

class BsmPermPattern final : public OpRewritePattern<ttg::BsmPermOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttg::BsmPermOp op,
                                PatternRewriter &rewriter) const override {
    if (isPhysicalBsm(op))
      return failure();
    auto sourceLoad = op.getSrc1().getDefiningOp<ttg::LocalLoadOp>();
    if (!sourceLoad)
      return op.emitError()
             << "requires its input to be directly defined by ttg.local_load";
    if (!sourceLoad->hasOneUse())
      return op.emitError()
             << "requires its defining ttg.local_load to be used only by this "
                "BSM operation";

    auto target = dyn_cast<RankedTensorType>(op.getResult().getType());
    auto sourceType = dyn_cast<RankedTensorType>(sourceLoad.getType());
    if (!sourceType || sourceType.getRank() != 2 || sourceType != target)
      return op.emitError()
             << "requires logical local_load and BSM input/output to have the "
                "same rank-2 tensor type";

    auto i32Type = target.clone(IntegerType::get(op.getContext(), 32));
    // BSM is transparent to AccelerateMatmul.  Keep the split i32 ABI only
    // when the current LLVM permutation can represent the selected M0 exactly;
    // otherwise erase the logical marker and let the existing local_load
    // lowering perform its fused permutation (or ordinary load) for that M0.
    if (!supportsPhysicalBsmTypes(i32Type, target)) {
      rewriter.replaceOp(op, sourceLoad.getResult());
      LDBG("folded unsupported split BSM into local_load " << op.getLoc());
      return success();
    }

    rewriter.modifyOpInPlace(sourceLoad, [&] {
      sourceLoad.getResult().setType(i32Type);
      sourceLoad->setAttr("mmaMode", rewriter.getI32IntegerAttr(2));
    });
    LDBG("materialized C500 BSM " << op.getLoc());
    return success();
  }
};

class GluonToTritonGPUConversionPass final
    : public gd::impl::GluonToTritonGPUConversionPassBase<
          GluonToTritonGPUConversionPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    const bool storeCoalesce = getOperation()->hasAttr(kStoreCoalesceAttr);
    patterns.add<ExtractSlicePattern>(context, storeCoalesce);
    patterns.add<InsertSlicePattern>(context, storeCoalesce);
    patterns.add<BsmPermPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    bool hasUnloweredOp = false;
    getOperation()->walk([&](Operation *op) {
      if (isa<gd::ExtractSliceOp, gd::InsertSliceOp>(op) ||
          (isa<ttg::BsmPermOp>(op) && !isPhysicalBsm(cast<ttg::BsmPermOp>(op)))) {
        op->emitError() << "could not lower this Gluon logical view/BSM to "
                           "its C500 TTG physical form";
        hasUnloweredOp = true;
      }
    });
    if (hasUnloweredOp)
      signalPassFailure();
    getOperation()->removeAttr(kStoreCoalesceAttr);
  }
};

} // namespace

namespace mlir::triton::gluon {

std::unique_ptr<Pass> createGluonToTritonGPUConversionPass() {
  return std::make_unique<GluonToTritonGPUConversionPass>();
}

} // namespace mlir::triton::gluon

#undef LDBG
#undef DEBUG_TYPE
