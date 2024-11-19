#pragma once


#include "compiler/include/Dialect/NPU/IR/NPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>
#include <optional>
#include <type_traits>

using namespace mlir;
using namespace deeplink;

namespace {

struct CopyConverter : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // assert(false);
    

    auto args = adaptor.getOperands();
    auto subview0 = dyn_cast_or_null<memref::SubViewOp>(
          args[0].getDefiningOp());
    if (subview0) {
      llvm::outs() << "zcx0 log : \n";
      auto staticOffsets = subview0.getStaticOffsets();
      auto staticSizes = subview0.getStaticSizes();
      auto staticStrides = subview0.getStaticStrides();
      for (auto x : staticSizes){
        llvm::outs() << "zcx0 x:" << x << "\n";
      }
      // SmallVector<int64_t> staticStrides, staticOffsets, staticSizes;
      // SmallVector<Value> dynamicStrides, dynamicOffsets, dynamicSizes;

      // dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
      // dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
      // dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);

      //  auto replacement = rewriter.create<ascend::CopyOp>(
      //   op.getLoc(), args[0],args[1], staticOffsets, staticSizes, staticStrides);

      // rewriter.replaceOp(op, replacement);
    }
    if (auto subview1 = dyn_cast_or_null<memref::SubViewOp>(
          args[1].getDefiningOp())) {
      llvm::outs() << "zcx1 log : \n";
      // subview0.getOperands();
    }

   
    // if (isa<memref::SubViewOp>(args[0].getDefiningOp())) {
    //   Operation *definingOp = args[0].getDefiningOp()
    // }
    auto replacement = rewriter.create<npu::CopyOp>(
        op.getLoc(), args[0],args[1]);

    rewriter.replaceOp(op, replacement);
    return success();
  }
};

} // namespace