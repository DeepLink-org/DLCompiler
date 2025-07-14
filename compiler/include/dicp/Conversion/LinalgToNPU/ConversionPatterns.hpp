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
#include "mlir/IR/BuiltinAttributes.h"
// #include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/Linalg/IR/LinalgAttributes.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"

#include <iostream>
#include <numeric>
#include <optional>
#include <type_traits>

using namespace mlir;
using namespace dicp;
using namespace mlir::utils;

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



struct ConvertLinalgGenericToArith
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {

    std::cout << "[ConvertLinalgGenericToArith] Starting matchAndRewrite for LinalgGenericOp at location: ";
    genericOp.getLoc().print(llvm::outs());
    std::cout << "\n";

    // === 1. 检查迭代器类型是否全部为 parallel ===
    SmallVector<StringRef> expectedIterTypes;
    if (failed(getIteratorTypeNames(genericOp, expectedIterTypes)))
      return failure();

    if (!llvm::all_of(expectedIterTypes, [](StringRef type) {
          return type == "parallel";
        })) {
      std::cout << "[ERROR] Only 'parallel' iterator types are supported.\n";
      return failure();
    }
    std::cout << "[INFO] All iterator types are 'parallel'.\n";

    // === 2. 检查索引映射是否为 identity 并一致 ===
    if (!hasIdentityIndexingMaps(genericOp)) {
      std::cout << "[ERROR] Indexing maps are not all identity.\n";
      return failure();
    }
    std::cout << "[INFO] Indexing maps are all identity.\n";

    // === 3. 检查 block 结构 ===
    Region &region = genericOp->getRegion(0);
    if (!region.hasOneBlock()) {
      std::cout << "[ERROR] Region must have exactly one block.\n";
      return failure();
    }

    Block &block = region.front();
    if (block.empty() || !isa<linalg::YieldOp>(block.back())) {
      std::cout << "[ERROR] Block does not end with linalg.yield.\n";
      return failure();
    }

    Operation *innerOp = block.getTerminator()->getPrevNode();
    if (!innerOp || innerOp->getNumResults() != 1) {
      std::cout << "[ERROR] Expected a single result before yield.\n";
      return failure();
    }

    auto yieldOp = cast<linalg::YieldOp>(block.getTerminator());
    if (yieldOp->getNumOperands() != 1 ||
        yieldOp->getOperand(0) != innerOp->getResult(0)) {
      std::cout << "[ERROR] Yield operand mismatch.\n";
      return failure();
    }
    std::cout << "[INFO] Inner operation and yield check passed.\n";

    // === 4. 检查输入数量匹配 ===
    if (innerOp->getNumOperands() != genericOp.getNumDpsInputs()) {
      std::cout << "[ERROR] Number of operands mismatch between inner op and inputs.\n";
      return failure();
    }

    // === 5. 构造 tensor-level 运算 ===
    Location loc = genericOp.getLoc();
    SmallVector<Value> inputValues;
    for (OpOperand *opOperand : genericOp.getDpsInputOperands()) {
      inputValues.push_back(opOperand->get());
    }

    Type resultTensorType = genericOp.getResult(0).getType();
    // Location loc = genericOp.getLoc();

    // === 6. 根据 innerOp 类型构造对应的 tensor-op ===
    Operation *newOp = nullptr;

    if (auto extfOp = dyn_cast<arith::ExtFOp>(innerOp)) {
      std::cout << "[INFO] Detected arith.extf -> converting to tensor-level extf\n";
      newOp = rewriter.create<arith::ExtFOp>(loc, resultTensorType, inputValues[0]);
    } else if (auto truncfOp = dyn_cast<arith::TruncFOp>(innerOp)) {
      std::cout << "[INFO] Detected arith.truncf -> converting to tensor.cast\n";
      newOp = rewriter.create<arith::TruncFOp>(loc, resultTensorType, inputValues[0]);

    } else if (auto addfOp = dyn_cast<arith::AddFOp>(innerOp)) {
      std::cout << "[INFO] Detected arith.addf -> converting to tensor-level addf\n";
      newOp = rewriter.create<arith::AddFOp>(loc, resultTensorType, inputValues[0], inputValues[1]);

    } else if (auto mulfOp = dyn_cast<arith::MulFOp>(innerOp)) {
      std::cout << "[INFO] Detected arith.mulf -> converting to tensor-level mulf\n";
      newOp = rewriter.create<arith::MulFOp>(loc, resultTensorType, inputValues[0], inputValues[1]);

    } else if (auto divfOp = dyn_cast<arith::DivFOp>(innerOp)) {
      std::cout << "[INFO] Detected arith.divf -> converting to tensor-level divf\n";
      newOp = rewriter.create<arith::DivFOp>(loc, resultTensorType, inputValues[0], inputValues[1]);

    } else if (auto subfOp = dyn_cast<arith::SubFOp>(innerOp)) {
      std::cout << "[INFO] Detected arith.subf -> converting to tensor-level subf\n";
      newOp = rewriter.create<arith::SubFOp>(loc, resultTensorType, inputValues[0], inputValues[1]);

    } else if (auto expOp = dyn_cast<math::ExpOp>(innerOp)) {
      std::cout << "[INFO] Detected math.exp -> converting to tensor-level exp\n";
      newOp = rewriter.create<math::ExpOp>(loc, resultTensorType, inputValues[0]);

    } else {
      std::cout << "[WARN] Unsupported arithmetic op: " << innerOp->getName().getStringRef().str() << "\n";
      return failure();
    }

    // === 7. 替换原操作 ===
    std::cout << "[INFO] Replacing linalg.generic with new op: " << newOp->getName().getStringRef().str() << "\n";
    rewriter.replaceOp(genericOp, newOp->getResults());

    return success();
  }

private:
  LogicalResult getIteratorTypeNames(linalg::GenericOp op,
                                    SmallVectorImpl<StringRef> &types) const {
    auto iteratorAttrs = op.getIteratorTypes().getValue();
    for (Attribute attr : iteratorAttrs) {
      if (auto iterTypeAttr = dyn_cast<linalg::IteratorTypeAttr>(attr)) {
        types.push_back(
            // linalg::stringifyIteratorType(iterTypeAttr.getValue()));
            mlir::utils::stringifyIteratorType(iterTypeAttr.getValue()));
      } else if (auto strAttr = dyn_cast<StringAttr>(attr)) {
        types.push_back(strAttr.getValue());
      } else {
        std::cout << "[ERROR] Unsupported iterator type attribute.\n";
        return failure();
      }
    }
    return success();
  }

  // 检查所有 indexing_map 是否为 identity 且一致
  bool hasIdentityIndexingMaps(linalg::GenericOp op) const {
    auto maps = op.getIndexingMaps();
    if (maps.empty())
      return false;

    AffineMap firstMap = cast<AffineMapAttr>(maps[0]).getValue();
    if (!firstMap.isIdentity())
      return false;

    for (auto mapAttr : maps) {
      if (!isa<AffineMapAttr>(mapAttr))
        return false;
      AffineMap map = cast<AffineMapAttr>(mapAttr).getValue();
      if (map != firstMap)
        return false;
    }
    return true;
  }
};


// ... 已有代码 ...


struct ConvertLinalgGenericToBroadcast : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    std::cout << "[ConvertLinalgGenericToBroadcast] Starting matchAndRewrite for LinalgGenericOp at location: ";
    genericOp.getLoc().print(llvm::outs());
    std::cout << "\n";

    // === Step 1: 检查是否是一个合法的 broadcast 模式 ===
    if (!isBroadcastPattern(genericOp)) {
      std::cout << "[INFO] Not a broadcast pattern\n";
      return failure();
    }
    std::cout << "[INFO] Detected broadcast pattern\n";

    // === Step 2: 获取输入输出 Tensor ===
    Value input = genericOp.getDpsInputOperand(0)->get();
    Value output = genericOp.getDpsInitOperand(0)->get();

    // 提取 broadcastDims 属性
    auto attr = genericOp->getAttr("broadcastDims");
    if (!attr) {
      std::cout << "[ERROR] Missing 'broadcastDims' attribute\n";
      return failure();
    }

    std::cout << "[DEBUG] Raw broadcastDims attribute value: ";
    attr.print(llvm::outs());
    std::cout << "\n";
    SmallVector<int64_t> broadcastDims;
    if (auto denseAttr = dyn_cast<DenseI64ArrayAttr>(attr)) {
      // broadcastDims = denseAttr.asArrayRef().vec();
      broadcastDims.assign(denseAttr.asArrayRef().begin(), denseAttr.asArrayRef().end());
    } else if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
        for (auto element : arrayAttr) {
            if (auto intAttr = dyn_cast<IntegerAttr>(element)) {
                broadcastDims.push_back(intAttr.getInt());
            } else {
                std::cout << "[ERROR] Invalid element in 'broadcastDims' array\n";
                return failure();
            }
        }
    } else {
        std::cout << "[ERROR] Invalid 'broadcastDims' attribute type\n";
        return failure();
    }

    std::cout << "[INFO] Detected broadcastDims = [";
    for (int64_t d : broadcastDims) {
      std::cout << d << " ";
    }
    std::cout << "]\n";

    // 创建 linalg.broadcast 操作
    rewriter.setInsertionPoint(genericOp);
    auto broadcastOp = rewriter.create<linalg::BroadcastOp>(
        genericOp.getLoc(),
        input,
        output,
        broadcastDims
    );

    // 替换原操作
    // 打印操作
    std::cout << "[DEBUG] Replacing linalg.generic with linalg.broadcast\n";
    std::cout << "[DEBUG] Before replacement:\n";
    genericOp.print(llvm::outs());
    std::cout << "\n";
    // 打印替换后的操作
    std::cout << "[DEBUG] After replacement:\n";
    broadcastOp.print(llvm::outs());
    std::cout << "\n";
    rewriter.replaceOp(genericOp, broadcastOp.getResult());
    std::cout << "[INFO] Replaced linalg.generic with linalg.broadcast\n";
    return success();
  }

private:
  // 判断是否为 broadcast 模式
    bool isBroadcastPattern(linalg::GenericOp op) const {
    std::cout << "[DEBUG] Checking broadcast pattern for linalg.generic at location: ";
      op.getLoc().print(llvm::outs());
      std::cout << "\n";

    // 1. 检查迭代器类型
      SmallVector<StringRef> iterTypes;
      if (failed(getIteratorTypeNames(op, iterTypes))) {
        std::cout << "[DEBUG] Failed to get iterator type names. Not a broadcast pattern.\n";
        return false;
      }
      if (!llvm::all_of(iterTypes, [](StringRef type) { return type == "parallel"; })) {
      std::cout << "[DEBUG] Iterator types are not all 'parallel'.\n";
        return false;
      }
      std::cout << "[DEBUG] All iterator types are 'parallel'.\n";

    // 2. 检查输入输出数量
      if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1) {
      std::cout << "[DEBUG] Expected 1 input and 1 output.\n";
        return false;
      }
    std::cout << "[DEBUG] Number of inputs and outputs matched.\n";

      // 检查 block 结构
      // Region &region = op->getRegion(0);
      if (op->getNumRegions() == 0) {
        std::cout << "[ERROR] Operation has no regions.\n";
        return false;
      }

      Region *region = &op->getRegion(0);
      if (!region || !region->hasOneBlock()) {
        std::cout << "[ERROR] Region not valid or does not have one block.\n";
        return false;
      }
      std::cout << "[DEBUG] Region has exactly one block.\n";

      Block &block = region->front();
    if (block.empty() || !isa<linalg::YieldOp>(block.back())) {
      std::cout << "[DEBUG] Block does not end with linalg.yield.\n";
        return false;
      }
      std::cout << "[DEBUG] Block is not empty and ends with linalg.yield.\n";

      Operation *innerOp = block.getTerminator()->getPrevNode();
    if (innerOp && !isa<linalg::YieldOp>(innerOp)) {
      std::cout << "[DEBUG] Expected only linalg.yield in block.\n";
        return false;
      }
      std::cout << "[DEBUG] No inner operation before yield.\n";

    auto yieldOp = cast<linalg::YieldOp>(block.getTerminator());
    if (yieldOp->getNumOperands() != 1 || yieldOp->getOperand(0) != block.getArgument(0)) {
      std::cout << "[DEBUG] Yield operand mismatch.\n";
        return false;
      }
      std::cout << "[DEBUG] Yield operand check passed.\n";

    std::cout << "[DEBUG] Inner operation and yield check passed.\n";

    // 4. 检查是否有 broadcastDims 属性
    if (!op->hasAttr("broadcastDims")) {
      std::cout << "[DEBUG] Missing 'broadcastDims' attribute.\n";
      return false;
    }

      std::cout << "[DEBUG] 'broadcastDims' attribute found. This is a broadcast pattern.\n";
      return true;
    }

  // 获取 Iterator 类型名称
  LogicalResult getIteratorTypeNames(linalg::GenericOp op,
                                    SmallVectorImpl<StringRef> &types) const {
    auto iteratorAttrs = op.getIteratorTypes().getValue();
    for (Attribute attr : iteratorAttrs) {
      if (auto iterTypeAttr = dyn_cast<linalg::IteratorTypeAttr>(attr)) {
        types.push_back(mlir::utils::stringifyIteratorType(iterTypeAttr.getValue()));
      } else if (auto strAttr = dyn_cast<StringAttr>(attr)) {
        types.push_back(strAttr.getValue());
      } else {
        std::cout << "[ERROR] Unsupported iterator type attribute.\n";
        return failure();
      }
    }
    return success();
  }
};

// ... 已有代码 ...



} // namespace
