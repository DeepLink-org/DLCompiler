#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"
#include "dicp/Dialect/LinalgExt/Transforms/Transforms.h"

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "tensor-transform"

using namespace mlir;
using namespace dicp;
using namespace LinalgExt;

namespace mlir {
namespace dicp {
namespace LinalgExt {
#define GEN_PASS_DEF_NORMALIZESLICEOPS
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace LinalgExt
} // namespace dicp
} // namespace mlir

namespace {

/// Compute the dropped dimensions of a rank-reducing tensor.extract_slice op or
/// rank-extending tensor.insert_slice op.
static llvm::SmallBitVector
getDroppedDimsForInterleave(ArrayRef<int64_t> reducedShape,
                            ArrayRef<OpFoldResult> mixedSizes) {
  // TODO this is old community function, delete it afterwards
  llvm::SmallBitVector droppedDims(mixedSizes.size());
  int64_t shapePos = 0;

  for (const auto &size : enumerate(mixedSizes)) {
    // Rank-reduced dims must have a static unit dimension.
    bool isStaticUnitSize =
        isa<Attribute>(size.value()) &&
        llvm::cast<IntegerAttr>(cast<Attribute>(size.value())).getInt() == 1;

    if (shapePos == static_cast<int64_t>(reducedShape.size())) {
      // There are no more dims in the reduced shape. All remaining sizes must
      // be rank-reduced dims.
      assert(isStaticUnitSize && "expected unit dim");
      droppedDims.set(size.index());
      continue;
    }

    // Dim is preserved if the size is not a static 1.
    if (!isStaticUnitSize) {
      ++shapePos;
      continue;
    }

    // Dim is preserved if the reduced shape dim is also 1.
    if (reducedShape[shapePos] == 1) {
      ++shapePos;
      continue;
    }

    // Otherwise: Dim is dropped.
    droppedDims.set(size.index());
  }

  return droppedDims;
}

/// This pattern detects a chain of `tensor::InsertSliceOp` that together
/// implement an interleave write: multiple source tensors are inserted into the
/// same destination along the last dimension using different static offsets.
/// Once detected, the pattern normalizes this chain into a canonical form
/// where the last-dimension offsets are [0..channelNum-1] and the stride is
/// `channelNum`, making the interleave structure explicit and ready for further
/// fusion or replacement by a dedicated Interleave op.
struct NormalizeInsertSliceOpToInterleaveOp
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  std::optional<int> getInterLeaveChannelIdx(Operation *op) const {
    if (auto insertSliceOp = llvm::dyn_cast<tensor::InsertSliceOp>(op)) {
      if (insertSliceOp.getStaticOffsets().empty())
        return std::nullopt;
      // dynamic offset return INT64_MIN
      if (insertSliceOp.getStaticOffsets().back() == INT64_MIN)
        return std::nullopt;
      return insertSliceOp.getStaticOffsets().back();
    }
    return std::nullopt;
  }

  bool isInterLeavePartialPattern(tensor::InsertSliceOp insertSliceOp,
                                  Value curSource,
                                  int64_t interLeaveChannelNums) const {
    SmallVector<int64_t> srcShape(
        dyn_cast<ShapedType>(curSource.getType()).getShape());
    SmallVector<int64_t> dstShape(
        dyn_cast<ShapedType>(insertSliceOp.getDest().getType()).getShape());

    if (dstShape.size() != srcShape.size())
      return false;

    // Correspond to rule 1&2
    if (!std::equal(srcShape.begin(), srcShape.end() - 1, dstShape.begin()) ||
        ShapedType::isDynamic(dstShape.back()) ||
        ShapedType::isDynamic(srcShape.back()) ||
        dstShape.back() != srcShape.back() * interLeaveChannelNums) {
      return false;
    }

    // Collect layout offset
    std::optional<int> interLeaveChannelIdx =
        getInterLeaveChannelIdx(insertSliceOp);
    if (!interLeaveChannelIdx.has_value()) {
      return false;
    }
    if (interLeaveChannelIdx > interLeaveChannelNums) {
      return false;
    }

    // Correspond to rule 3
    if (insertSliceOp.getStaticStrides().back() != 1) {
      return false;
    }
    return true;
  }

  // Actually, tensor::InsertSliceOp could express any dimension insertion
  // flexibly with layout info.
  //
  // While current hfusion::InterleaveOp just wanna match pattern where
  // 1. non-last diemsnion shape of input and output should be equal
  // 2. tail shape scale between dst and src equals channelNum, which also
  // means
  //    last dimension of all types could only be static
  // 3. In layout, last dimension stride equals channelNum
  // 4. In layout, last dimension offsets of all candidate InsertSliceOp
  // should
  //    make a range of [0, channelNum)
  //
  // And for rank-extended state of tensor::InsertSliceOp, where source rank
  // may less than destination, InterleaveOp just support extended rank is
  // last dimension and size equals channelNum, then explicitly expand shape
  // on src before create InterleaveOp
  LogicalResult traceInterLeavePattern(tensor::InsertSliceOp insertSliceOp,
                                       llvm::BitVector &findChannels,
                                       SmallVector<Value> &inputs,
                                       int64_t interLeaveChannelNums,
                                       PatternRewriter &rewriter) const {
    Value curSrc = insertSliceOp.getSource();

    llvm::SmallBitVector extendedRankRecord =
        getDroppedDimsForInterleave(insertSliceOp.getSourceType().getShape(),
                                    insertSliceOp.getMixedSizes());
    if (extendedRankRecord.any()) {
      if (extendedRankRecord.find_first() !=
          static_cast<int>(extendedRankRecord.size()) - 1)
        return rewriter.notifyMatchFailure(
            insertSliceOp, "extended rank could only be last dimension");

      // Extented rank size of destination must be interLeaveChannelNums
      if (dyn_cast<ShapedType>(insertSliceOp.getDest().getType())
              .getShape()
              .back() != interLeaveChannelNums)
        return rewriter.notifyMatchFailure(
            insertSliceOp,
            "size of extended rank axis should equal channel num");

      auto originType = llvm::dyn_cast<RankedTensorType>(curSrc.getType());
      SmallVector<int64_t> shape(originType.getShape());

      // Here represents last dimension which is only extended rank
      shape.push_back(1);
      RankedTensorType newType =
          RankedTensorType::get(shape, originType.getElementType());

      std::optional<SmallVector<ReassociationIndices>> reassociation =
          getReassociationIndicesForReshape(originType, newType);
      assert(reassociation.has_value());

      auto expandOp = rewriter.create<tensor::ExpandShapeOp>(
          insertSliceOp.getLoc(), newType, curSrc, reassociation.value());
      curSrc = expandOp.getResult();
    }

    if (!isInterLeavePartialPattern(insertSliceOp, curSrc,
                                    interLeaveChannelNums))
      return rewriter.notifyMatchFailure(
          insertSliceOp,
          "current tensor::InsertSliceOp layout doen't satisfy condition "
          "to be converted to hfusion::InterleaveOp");

    auto channelIdxMaybe = getInterLeaveChannelIdx(insertSliceOp);
    if (!channelIdxMaybe.has_value()) {
      return failure();
    }
    int channelIdx = channelIdxMaybe.value();
    // set channelIdx-bit to findChannels and push corresponding input
    findChannels[channelIdx] = true;
    inputs[channelIdx] = curSrc;

    // findChannels all true
    // Correspond to rule 4
    if (findChannels == llvm::BitVector(findChannels.size(), true)) {
      return success();
    }

    // trace further
    auto dstDefiningOp =
        insertSliceOp->getOperand(1).getDefiningOp<tensor::InsertSliceOp>();
    if (!dstDefiningOp) {
      return rewriter.notifyMatchFailure(
          insertSliceOp,
          "tensor::InsertSliceOp chain from current op can't reach "
          "interLeave channel num");
    }
    return traceInterLeavePattern(dstDefiningOp, findChannels, inputs,
                                  interLeaveChannelNums, rewriter);
  }

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const override {
    if (!insertSliceOp.hasPureTensorSemantics()) {
      return failure();
    }

    // TODO: find interLeaveChannelNums greedily.
    const int64_t interLeaveChannelNums = 2;
    llvm::BitVector findChannels(interLeaveChannelNums, false);
    SmallVector<Value> inputs(interLeaveChannelNums);

    // 1. Trace the pattern and collect inputs
    if (traceInterLeavePattern(insertSliceOp, findChannels, inputs,
                               interLeaveChannelNums, rewriter)
            .failed()) {
      return failure();
    }

    // 2. Find the root destination tensor.
    // traceInterLeavePattern has confirmed the chain exists.
    // We traverse back (interLeaveChannelNums - 1) times to find the
    // initial buffer that the first slice was inserted into.
    Value accumulatedDest = insertSliceOp.getDest();
    for (int i = 0; i < interLeaveChannelNums - 1; ++i) {
      auto parentOp = accumulatedDest.getDefiningOp<tensor::InsertSliceOp>();
      // Use cast because traceInterLeavePattern ensured the chain exists
      if (!parentOp)
        break;
      accumulatedDest = parentOp.getDest();
    }

    Location loc = insertSliceOp.getLoc();

    // 3. Rebuild the InsertSliceOp chain with normalized strides.
    for (int i = 0; i < interLeaveChannelNums; ++i) {
      Value src = inputs[i];
      if (!src) {
        // Should not happen if traceInterLeavePattern returns success
        // and findChannels is fully set.
        return failure();
      }

      auto srcType = llvm::cast<RankedTensorType>(src.getType());
      // Not support dynamic shape
      if (!srcType.hasStaticShape()) {
        return failure();
      }
      int64_t rank = srcType.getRank();

      // Prepare Offsets, Sizes, Strides
      SmallVector<OpFoldResult> offsets, sizes, strides;
      ArrayRef<int64_t> srcShape = srcType.getShape();

      // Populate dimensions 0 to Rank-2 (standard dims)
      // and Rank-1 (last dim/channel dim)
      for (int64_t d = 0; d < rank; ++d) {
        // Offset: 0 for all dims, except last dim is the channel index 'i'
        if (d == rank - 1) {
          offsets.push_back(rewriter.getIndexAttr(i));
        } else {
          offsets.push_back(rewriter.getIndexAttr(0));
        }

        // Size: Get from src shape (Static only)
        sizes.push_back(rewriter.getIndexAttr(srcShape[d]));

        // Stride: 1 for all dims, except last dim is 'interLeaveChannelNums'
        if (d == rank - 1) {
          strides.push_back(rewriter.getIndexAttr(interLeaveChannelNums));
        } else {
          strides.push_back(rewriter.getIndexAttr(1));
        }
      }

      // Create the new standardized InsertSliceOp
      auto newInsertOp = rewriter.create<tensor::InsertSliceOp>(
          loc, src, accumulatedDest, offsets, sizes, strides);

      // Update accumulatedDest for the next iteration
      accumulatedDest = newInsertOp.getResult();
    }

    // 4. Replace the original op with the result of the new chain
    rewriter.replaceOp(insertSliceOp, accumulatedDest);
    return success();
  }
}; // struct NormalizeInsertSliceOpInInterleavePattern

struct NormalizeSliceOpsPass
    : public mlir::dicp::LinalgExt::impl::NormalizeSliceOpsBase<
          NormalizeSliceOpsPass> {
  NormalizeSliceOpsPass() = default;
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    // Build patterns
    RewritePatternSet patterns(context);
    populateNormalizeInsertSliceOpInInterleavePattern(patterns);

    // Apply patterns greedily on this function
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::dicp::LinalgExt::populateNormalizeInsertSliceOpInInterleavePattern(
    RewritePatternSet &patterns) {
  patterns.add<NormalizeInsertSliceOpToInterleaveOp>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::dicp::LinalgExt::createNormalizeSliceOpsPass() {
  return std::make_unique<NormalizeSliceOpsPass>();
}