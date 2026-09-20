#include "flagtree/Common/UnifiedHardware.h"

#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/MemrefCopyToDMA_FlagTree/MemrefCopyToDMAFlagTree.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR//MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#define DEBUG_TYPE "memref-copy-to-dma-flagtree"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

namespace {
struct CopyConverter : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  // Get the parameter list of Strides, Sizes and Offsets
  SmallVector<Value> getValueList(OpBuilder &builder, Location loc,
                                  ArrayRef<OpFoldResult> ofrs) const {
    SmallVector<Value> values;
    for (OpFoldResult ofr : ofrs) {
      if (Attribute attr = ofr.dyn_cast<Attribute>()) {
        values.push_back(builder.create<arith::ConstantIndexOp>(
            loc, mlir::cast<IntegerAttr>(attr).getInt()));
      } else {
        values.push_back(ofr.dyn_cast<Value>());
      }
    }
    return values;
  }

  // Calculate the total number of DMA handling elements
  Value getTotalElementCount(OpBuilder &builder, Location loc,
                             ArrayRef<Value> sizes) const {
    assert(!sizes.empty());
    Value total = sizes.front();
    for (size_t i = 1; i < sizes.size(); ++i) {
      total = builder.create<arith::MulIOp>(loc, total, sizes[i]);
    }
    return total;
  }

  // Check whether the stride is 1
  bool isAllStrideOne(ArrayRef<OpFoldResult> strides) const {
    for (OpFoldResult ofr : strides) {
      if (auto attr = ofr.dyn_cast<Attribute>()) {
        auto intAttr = dyn_cast<IntegerAttr>(attr);
        if (!intAttr || intAttr.getInt() != 1)
          return false;
        continue;
      }

      if (auto val = ofr.dyn_cast<Value>()) {
        if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>())
          if (constOp.value() == 1)
            continue;

        if (auto intOp = val.getDefiningOp<arith::ConstantIntOp>())
          if (intOp.value() == 1)
            continue;
      }
      return false;
    }
    return true;
  }

  LogicalResult rewriteCopyToDma(memref::CopyOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
    auto hardwareManager = mlir::flagtree::createUnifiedHardwareManager();
    auto dmaTag = hardwareManager->getDMATag();
    if (!dmaTag)
      return failure();

    Location loc = op.getLoc();
    Value src = adaptor.getSource();
    Value dst = adaptor.getTarget();

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    SmallVector<Value> srcIndices, dstIndices;
    Value numElements;
    Operation *srcDef = src.getDefiningOp();
    //
    // Rewriting memref.copy to asynchronous DMA transfer
    //
    // This transform replaces memref.copy with memref.dma_start
    // and memref.dma_wait, There are two cases: Mask and Structured.
    //   - memref.subview ops with static offsets and sizes, or
    //   - memref.reinterpret_cast ops that preserve the shape.
    //
    // Key DMA parameters:
    //
    //   - srcIndices / dstIndices:
    //       * For memref.subview, use the offset list from getOffsets().
    //         E.g.,memref.subview %src[%i, %j][M, N][1, 1] → indices = [%i,%j]
    //       * For memref.reinterpret_cast, offsets are not meaningful for DMA.
    //         indices default to all-zero (e.g., [%c0, %c0, ...]).
    //
    //   - numElements:
    //       * For both cases, compute as the product of the sizes.
    //         E.g., [M, N] → M * N
    //
    //   - tag:
    //       * A synchronization buffer of type memref<1xi32, *>.
    //         The memory space `*` denotes a hardware-reserved region for DMA
    //         completion signaling, determined by the unified hardware layer.
    int64_t rank = mlir::cast<MemRefType>(src.getType()).getRank();
    srcIndices.assign(rank, zero);
    dstIndices.assign(rank, zero);

    if (auto srcSubview = dyn_cast_or_null<memref::SubViewOp>(srcDef)) {
      auto dstSubview = dyn_cast<memref::SubViewOp>(dst.getDefiningOp());
      auto sizes = getValueList(rewriter, loc, srcSubview.getMixedSizes());
      numElements = getTotalElementCount(rewriter, loc, sizes);
    } else if (auto castOp = dyn_cast<memref::ReinterpretCastOp>(srcDef)) {
      auto sizes = getValueList(rewriter, loc, castOp.getMixedSizes());
      numElements = getTotalElementCount(rewriter, loc, sizes);
    } else {
      return failure();
    }

    Type i32Type = rewriter.getIntegerType(32);
    Attribute tagMemSpace = IntegerAttr::get(i32Type, dmaTag);

    MemRefType tagType = MemRefType::get({1}, i32Type, nullptr, tagMemSpace);
    Value tag = rewriter.create<memref::AllocOp>(loc, tagType);
    SmallVector<Value> tagIndices = {zero};

    rewriter.create<memref::DmaStartOp>(loc, src, srcIndices, dst, dstIndices,
                                        numElements, tag, tagIndices);

    rewriter.create<memref::DmaWaitOp>(loc, tag, tagIndices, numElements);

    rewriter.eraseOp(op);
    return success();
  }

  // StructuredToMemrefPass will generate the memref.copy operation, which
  // can be selectively converted to DMA operation later
  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto strAttr = op->getAttrOfType<mlir::StringAttr>("flagtree_hints");
    if (!strAttr || strAttr.getValue() != "dma") {
      return success();
    }

    bool isStrideOne = false;
    Value src = adaptor.getSource();
    Operation *srcDef = src.getDefiningOp();
    if (auto srcSubview = dyn_cast_or_null<memref::SubViewOp>(srcDef)) {
      isStrideOne = isAllStrideOne(srcSubview.getMixedStrides());
    } else if (auto castOp = dyn_cast<memref::ReinterpretCastOp>(srcDef)) {
      isStrideOne = isAllStrideOne(castOp.getMixedStrides());
    }

    // Skip stride is not 1
    if (isStrideOne) {
      return rewriteCopyToDma(op, adaptor, rewriter);
    }

    return success();
  }
};

} // namespace

void mlir::triton::populateMemrefCopyToDMAFlagTreeConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<CopyConverter>(patterns.getContext());
}
