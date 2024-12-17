#pragma once
#include "dicp/Dialect/NPU/IR/NPUDialect.h"
#include "dicp/Dialect/NPU/IR/NPUTypes.h"
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
using namespace dicp;

namespace {

template <typename T, typename OP>
static Value insertFront(T type, Location loc,
                          PatternRewriter &rewriter) {
  auto tq = rewriter.create<OP>(loc, type);
  auto *parentBlock = tq->getBlock();
  tq->moveBefore(&parentBlock->front());
  return tq;
}

struct CopyConverter : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto args = adaptor.getOperands();
    auto replacement = rewriter.create<npu::CopyOp>(
        op.getLoc(), args[0], args[1]);
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

struct SubViewConverter : public OpConversionPattern<memref::SubViewOp> {
  using OpConversionPattern<memref::SubViewOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::SubViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto args = adaptor.getOperands();
    auto replacement = rewriter.create<npu::SliceOp>(
        op.getLoc(), op.getResult().getType(), args[0]);
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

// struct AddFConverter : public OpConversionPattern<arith::AddFOp> {
//   using OpConversionPattern<arith::AddFOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(arith::AddFOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     auto loc = op.getLoc();
//     auto args = adaptor.getOperands();
//     Type resultType = op.getResult().getType();
    
//     auto tPipType = npu::TPipType::get(resultType.getContext(), 1);
//     auto tpip = rewriter.create<npu::CreateTPipOp>(loc,tPipType);//, rewriter.getI32IntegerAttr(1));

//     auto tQueueType = npu::TQueueType::get(resultType.getContext(), 1);
//     rewriter.create<npu::CreateTQueueOp>(loc,tQueueType);//, rewriter.getI32IntegerAttr(1));

//     Value replacement = rewriter.create<npu::AddFOp>(
//         loc, resultType, args[0], args[1]);

//     rewriter.replaceOp(op, replacement);
//     return success();
//   }
// };

struct LinalgGenericConverter : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto args = adaptor.getOperands();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Operation* lDefOp = lhs.getDefiningOp();
    Operation* rDefOp = rhs.getDefiningOp();
    rewriter.setInsertionPointAfter(lDefOp);
    // global tpip
    auto tPipType = npu::TPipType::get(getContext());
    Value tpip = rewriter.create<npu::CreateTPipOp>(loc,tPipType);
    // global tqueue
    auto vecInQueueType = npu::TQueueType::get(getContext(), 0, 2);
    Value lQueue = rewriter.create<npu::CreateTQueueOp>(loc, vecInQueueType);
    Value rQueue = rewriter.create<npu::CreateTQueueOp>(loc, vecInQueueType);
    auto vecOutQueueType = npu::TQueueType::get(getContext(), 1, 2);
    Value outQueue = rewriter.create<npu::CreateTQueueOp>(loc, vecOutQueueType);
    // global tensor
    auto GlobalTensorType = npu::GlobalTensorType::get(getContext(), 0);
    Value lGlobalTensor = rewriter.create<npu::CreateGlobalTensorOp>(loc, GlobalTensorType);
    Value rGlobalTensor = rewriter.create<npu::CreateGlobalTensorOp>(loc, GlobalTensorType);
    Value outGlobalTensor = rewriter.create<npu::CreateGlobalTensorOp>(loc, GlobalTensorType);
    // init
    lGlobalTensor = rewriter.create<npu::SetGlobalBufferOp>(loc, GlobalTensorType, lGlobalTensor);
    rGlobalTensor = rewriter.create<npu::SetGlobalBufferOp>(loc, GlobalTensorType, rGlobalTensor);
    outGlobalTensor = rewriter.create<npu::SetGlobalBufferOp>(loc, GlobalTensorType, outGlobalTensor);
    lQueue = rewriter.create<npu::InitlBufferOp>(loc, vecInQueueType, tpip, lQueue);
    rQueue = rewriter.create<npu::InitlBufferOp>(loc, vecInQueueType, tpip, rQueue);
    outQueue = rewriter.create<npu::InitlBufferOp>(loc, vecOutQueueType, tpip, outQueue);
   
    rewriter.setInsertionPointAfter(op);
    // copy in
    if (isa<memref::AllocOp>(rDefOp)) {
      auto rLocal = rewriter.create<npu::AllocLocalOp>(loc, rhs.getType(), rQueue);
      auto *parentBlock = rLocal->getBlock();
      rLocal->moveBefore(&parentBlock->front());
      rewriter.replaceOp(rDefOp, rLocal);
      for (auto user : rhs.getUsers()) {
        if (isa<npu::CopyOp>(user)) {
          rewriter.setInsertionPointAfter(user);
          rQueue = rewriter.create<npu::EnQueOp>(loc, vecInQueueType, rQueue, rLocal);
        }
      }
    }
    if (isa<memref::AllocOp>(lDefOp)) {
      auto lLocal = rewriter.create<npu::AllocLocalOp>(loc, lhs.getType(), lQueue);
      auto *parentBlock = lLocal->getBlock();
      lLocal->moveBefore(&parentBlock->front());
      rewriter.replaceOp(lDefOp, lLocal);
      for (auto user : lhs.getUsers()) {
        if (isa<npu::CopyOp>(user)) {
          rewriter.setInsertionPointAfter(user);
          lQueue = rewriter.create<npu::EnQueOp>(loc, vecInQueueType, lQueue, lLocal);
        }
      }
    }
    // compute
    rewriter.setInsertionPointAfter(op);
    Value lLocal = rewriter.create<npu::DeQueOp>(loc, lhs.getType(), lQueue);
    Value rLocal = rewriter.create<npu::DeQueOp>(loc, rhs.getType(), rQueue);
    ValueRange outputs = op.getOutputs();
    Value outLocal = rewriter.create<npu::AllocLocalOp>(loc, outputs[0].getType(), outQueue);
    Operation* outDefOp = outputs[0].getDefiningOp();
    if (isa<memref::AllocOp>(outDefOp)) {
      rewriter.replaceOp(outDefOp, outLocal);
    }
    auto replacement = rewriter.create<npu::AddFOp>(loc, lLocal, rLocal, outLocal);
    rewriter.replaceOp(op, replacement);
    outQueue = rewriter.create<npu::EnQueOp>(loc, vecOutQueueType, outQueue, outLocal);
    lQueue = rewriter.create<npu::FreeLocalOp>(loc, vecInQueueType, lQueue, lLocal);
    rQueue = rewriter.create<npu::FreeLocalOp>(loc, vecInQueueType, rQueue, rLocal);
    // copy out
    outLocal = rewriter.create<npu::DeQueOp>(loc, outputs[0].getType(), outQueue);
    for (auto user : outputs[0].getUsers()) {
      if (isa<memref::CopyOp>(user)) {
        user->setOperand(0, outLocal);
        rewriter.setInsertionPointAfter(user);
        outQueue = rewriter.create<npu::FreeLocalOp>(loc, vecOutQueueType, outQueue, outLocal);
      }
    }
    return success();
  }
};

} // namespace