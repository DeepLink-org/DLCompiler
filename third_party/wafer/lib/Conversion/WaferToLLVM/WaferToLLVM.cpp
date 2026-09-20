//===--------------------- WaferToLLVM.cpp ---------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file implements the patterns to convert operations from wafer dialect to
// LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#include "wafer/Conversion/WaferToLLVM/WaferToLLVM.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Utils/Utils.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "wafer/Dialect/IR/WaferDialect.h"
#include "llvm/ADT/TypeSwitch.h"

#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "wafer-to-llvm"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "wafer/Conversion/WaferToLLVM/Passes.h.inc"

namespace {
//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//
// Crt func name
const char rdma4dFuncName[] = "__Rdma4d";
const char wdma4dFuncName[] = "__Wdma4d";
const char rdma1dFuncName[] = "__Rdma1d";
const char wdma1dFuncName[] = "__Wdma1d";
const char rdmaFuncName[] = "__Rdma";
const char wdmaFuncName[] = "__Wdma";
const char memcpyFuncName[] = "__Memcpy";
const char recvFuncName[] = "__Recv";
const char sendFuncName[] = "__Send";
const char addVVFuncName[] = "__AddVV";
const char subVVFuncName[] = "__SubVV";
const char mulVVFuncName[] = "__MulVV";
const char divVVFuncName[] = "__DivVV";
const char absVVFuncName[] = "__AbsVV";
const char rsqrtVVFuncName[] = "__RsqrtVV";
const char sqrtVVFuncName[] = "__SqrtVV";
const char recipVVFuncName[] = "__RecipVV";
const char negVVFuncName[] = "__NegVV";
const char lnFuncName[] = "__Ln";
const char log2FuncName[] = "__Log2";
const char expFuncName[] = "__Exp";
const char pow2FuncName[] = "__Pow2";
const char sinFuncName[] = "__Sin";
const char cosFuncName[] = "__Cos";
const char addVSFuncName[] = "__AddVS";
const char subVSFuncName[] = "__SubVS";
const char mulVSFuncName[] = "__MulVS";
const char divVSFuncName[] = "__DivVS";
const char argMinFuncName[] = "__ArgMin";
const char argMaxFuncName[] = "__ArgMax";
const char reduceSumFuncName[] = "__ReduceSum";
const char reduceMaxFuncName[] = "__ReduceMax";
const char reduceMinFuncName[] = "__ReduceMin";
const char reduceMulFuncName[] = "__ReduceMul";
// Int8
const char int8ToBf16FuncName[] = "__INT8_BF16";
const char int8ToFp16FuncName[] = "__INT8_FP16";
const char int8ToFp32FuncName[] = "__INT8_FP32";
const char int8ToTf32FuncName[] = "__INT8_TF32";
// Int16
const char int16ToFp16FuncName[] = "__INT16_FP16";
const char int16ToBf16FuncName[] = "__INT16_BF16";
const char int16ToFp32FuncName[] = "__INT16_FP32";
const char int16ToTf32FuncName[] = "__INT16_TF32";
// Int32
const char int32ToFp16FuncName[] = "__INT32_FP16";
const char int32ToBf16FuncName[] = "__INT32_BF16";
const char int32ToFp32FuncName[] = "__INT32_FP32";
const char int32ToTf32FuncName[] = "__INT32_TF32";
// BF16
const char bf16ToInt8FuncName[] = "__BF16_INT8";
const char bf16ToInt16FuncName[] = "__BF16_INT16";
const char bf16ToInt32FuncName[] = "__BF16_INT32";
const char bf16ToFp16FuncName[] = "__BF16_FP16";
const char bf16ToFp32FuncName[] = "__BF16_FP32";
const char bf16ToTf32FuncName[] = "__BF16_TF32";
// FP16
const char fp16ToBf16FuncName[] = "__FP16_BF16";
const char fp16ToFp32FuncName[] = "__FP16_FP32";
const char fp16ToTf32FuncName[] = "__FP16_TF32";
const char fp16ToInt8FuncName[] = "__FP16_INT8";
const char fp16ToInt16FuncName[] = "__FP16_INT16";
const char fp16ToInt32FuncName[] = "__FP16_INT32";
// FP32
const char fp32ToInt8FuncName[] = "__FP32_INT8";
const char fp32ToInt16FuncName[] = "__FP32_INT16";
const char fp32ToInt32FuncName[] = "__FP32_INT32";
const char fp32ToFp16FuncName[] = "__FP32_FP16";
const char fp32ToBf16FuncName[] = "__FP32_BF16";
const char fp32ToTf32FuncName[] = "__FP32_TF32";
// TF32
const char tf32ToInt8FuncName[] = "__TF32_INT8";
const char tf32ToInt16FuncName[] = "__TF32_INT16";
const char tf32ToInt32FuncName[] = "__TF32_INT32";
const char tf32ToFp16FuncName[] = "__TF32_FP16";
const char tf32ToBf16FuncName[] = "__TF32_BF16";
const char tf32ToFp32FuncName[] = "__TF32_FP32";
// MXFP
const char fp8E4M3ToBF16FuncName[] = "__FP8E4M3_BF16";
const char fp8E4M3FNToBF16FuncName[] = "__FP8E4M3FN_BF16";
const char fp8E5M2ToBF16FuncName[] = "__FP8E5M2_BF16";
const char fp4E2M1ToBF16FuncName[] = "__FP4E2M1_BF16";
const char fp8E4M3ToFP16FuncName[] = "__FP8E4M3_FP16";
const char fp8E4M3FNToFP16FuncName[] = "__FP8E4M3FN_FP16";
const char fp8E5M2ToFP16FuncName[] = "__FP8E5M2_FP16";
const char fp4E2M1ToFP16FuncName[] = "__FP4E2M1_FP16";

const char boolEqualVVFuncName[] = "__BoolEqualVV";
const char boolUnEqualVVFuncName[] = "__BoolUnEqualVV";
const char boolGreaterEqualVVFuncName[] = "__BoolGreaterEqualVV";
const char boolGreaterVVFuncName[] = "__BoolGreaterVV";
const char boolLessEqualVVFuncName[] = "__BoolLessEqualVV";
const char boolLessThenVVFuncName[] = "__BoolLessThenVV";
const char equalVVFuncName[] = "__EqualVV";
const char unEqualVVFuncName[] = "__UnEqualVV";
const char greaterEqualVVFuncName[] = "__GreaterEqualVV";
const char greaterVVFuncName[] = "__GreaterVV";
const char lessEqualVVFuncName[] = "__LessEqualVV";
const char lessThenVVFuncName[] = "__LessThenVV";
const char boolEqualVSFuncName[] = "__BoolEqualVS";
const char boolUnEqualVSFuncName[] = "__BoolUnEqualVS";
const char boolGreaterEqualVSFuncName[] = "__BoolGreaterEqualVS";
const char boolGreaterVSFuncName[] = "__BoolGreaterVS";
const char boolLessEqualVSFuncName[] = "__BoolLessEqualVS";
const char boolLessThenVSFuncName[] = "__BoolLessThenVS";
const char equalVSFuncName[] = "__EqualVS";
const char unEqualVSFuncName[] = "__UnEqualVS";
const char greaterEqualVSFuncName[] = "__GreaterEqualVS";
const char greaterVSFuncName[] = "__GreaterVS";
const char lessEqualVSFuncName[] = "__LessEqualVS";
const char lessThenVSFuncName[] = "__LessThenVS";
const char andVVFuncName[] = "__AndVV";
const char orVVFuncName[] = "__OrVV";
const char xorVVFuncName[] = "__XorVV";
const char boolNotVFuncName[] = "__BoolNotV";
const char boolAndVFuncName[] = "__BoolAndV";
const char boolOrVFuncName[] = "__BoolOrV";
const char boolXorVFuncName[] = "__BoolXorV";
const char MaxVVFuncName[] = "__MaxVV";
const char MinVVFuncName[] = "__MinVV";
const char transposeFuncName[] = "__Transpose";
const char nchw2nhwcFuncName[] = "__Nchw2nhwc";
const char nhwc2nchwFuncName[] = "__Nhwc2nchw";
const char tanhFuncName[] = "__Tanh";
const char atomicBarrierInFuncName[] = "__AtomicBarrierIn";
const char atomicBarrierOutFuncName[] = "__AtomicBarrierOut";
const char MXFPScaleBF16FuncName[] = "__mxfpScaleBF16";
const char MXFPScaleFP16FuncName[] = "__mxfpScaleFP16";

static Value adjustElemCountType(ConversionPatternRewriter &rewriter,
                                 Location loc, Value elemCount) {
  Value newElemCount = elemCount;
  if (isa<IndexType>(elemCount.getType())) {
    newElemCount = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(), elemCount);
  } else if (isa<IntegerType>(elemCount.getType())) {
    auto elemCountType = dyn_cast<IntegerType>(elemCount.getType());
    if (elemCountType.isInteger(64))
      newElemCount = rewriter.create<arith::TruncIOp>(
          loc, rewriter.getI32Type(), elemCount);
  }
  return newElemCount;
}

static Value castIndexToInt32(ConversionPatternRewriter &rewriter, Location loc,
                              Value indexOp) {
  return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(),
                                             indexOp);
}

static Value createInt32ValueArray(ConversionPatternRewriter &rewriter,
                                   Location loc, SmallVector<Value> array,
                                   Operation *currentOp) {
  auto i32PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto i32Ty = rewriter.getI32Type();
  auto i64Ty = rewriter.getI64Type();

  // Find the parent function of the current operation
  Operation *parentFunc = currentOp->getParentOfType<LLVM::LLVMFuncOp>();
  LLVM::LLVMFuncOp funcOp = dyn_cast<LLVM::LLVMFuncOp>(parentFunc);
  assert(funcOp &&
         "Expected to find a parent function for the current operation\n");

  // Save current insertion point
  auto savedInsertionPoint = rewriter.saveInsertionPoint();

  // Insert alloca at the beginning of the function entry block
  Block &entryBlock = funcOp.getBody().front();
  rewriter.setInsertionPointToStart(&entryBlock);

  // Allocate memory for array
  Value rank = rewriter.create<LLVM::ConstantOp>(
      loc, i64Ty, rewriter.getI64IntegerAttr(array.size()));

  auto allocaOp = rewriter.create<LLVM::AllocaOp>(loc, i32PtrTy, i32Ty, rank);

  rewriter.restoreInsertionPoint(savedInsertionPoint);

  //   assert(moduleOp && moduleOp->hasAttr("triton_tsm.spm_use") &&
  //          "ModuleOp should not be null when creating an array");
  //   auto spmPointer =
  //   cast<IntegerAttr>(moduleOp->getAttr("triton_tsm.spm_use"))
  //                         .getValue()
  //                         .getZExtValue();
  //   Value spmOffsetOp = rewriter.create<LLVM::ConstantOp>(
  //       loc, rewriter.getI64Type(), rewriter.getI32IntegerAttr(spmPointer));
  //   auto elementPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  //   Value spmAddr = rewriter.create<LLVM::ZeroOp>(loc, elementPtrType);
  //   spmAddr =
  //       rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(),
  //       spmAddr);
  //   spmAddr = rewriter.create<LLVM::AddOp>(loc, rewriter.getI64Type(),
  //   spmAddr,
  //                                          spmOffsetOp);

  //   // Types for function declaration
  //   SmallVector<Type, 5> argTypes = {
  //       rewriter.getI64Type() // offset
  //   };

  // Declare the function
  //   Value funcPtr = triton::utils::declareWaferRuntimeFunction(
  //       moduleOp, rewriter, loc, "get_spm_memory_mapping_wrapper",
  //       elementPtrType, argTypes);

  // Create the call to __Rdma
  //   auto spmMemoryAddrPtr = rewriter.create<LLVM::CallOp>(
  //       loc, TypeRange{elementPtrType},
  //       "get_spm_memory_mapping_wrapper", // funcPtr,
  //       ValueRange{spmAddr});

  // Restore insertion point
  rewriter.restoreInsertionPoint(savedInsertionPoint);

  // Store each dimension in the array
  for (size_t i = 0; i < array.size(); i++) {
    // Create the index
    Value idx = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getI32IntegerAttr(i));

    // Create GEP to get pointer to array element
    Value elemPtr = rewriter.create<LLVM::GEPOp>(loc, i32PtrTy, i32Ty, allocaOp,
                                                 ArrayRef<Value>{idx});

    // Store the value
    rewriter.create<LLVM::StoreOp>(loc, array[i], elemPtr);
  }

  //   spmPointer += array.size() * sizeof(int32_t);
  //   // Record spm usage.
  //   moduleOp->setAttr(
  //       "triton_tsm.spm_use",
  //       mlir::IntegerAttr::get(mlir::IntegerType::get(moduleOp.getContext(),
  //       32),
  //                              spmPointer));

  return allocaOp;
}

static Value
indexValueArrayToInt32ValueArray(ConversionPatternRewriter &rewriter,
                                 Location loc, ValueRange array,
                                 Operation *currentOp) {

  SmallVector<Value> arrayValues;
  for (size_t i = 0; i < array.size(); i++) {
    // Create the dimension value
    arrayValues.push_back(castIndexToInt32(rewriter, loc, array[i]));
  }

  return createInt32ValueArray(rewriter, loc, arrayValues, currentOp);
}

static Value int32ArrayToInt32ValueArray(ConversionPatternRewriter &rewriter,
                                         Location loc, ArrayRef<int32_t> array,
                                         Operation *currentOp) {

  SmallVector<Value> arrayValues;
  auto i32Ty = rewriter.getI32Type();
  for (size_t i = 0; i < array.size(); i++) {
    // Create the dimension value
    arrayValues.push_back(rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(array[i])));
  }
  return createInt32ValueArray(rewriter, loc, arrayValues, currentOp);
}

//===----------------------------------------------------------------------===//
// Arith Operation Conversion Patterns
//===----------------------------------------------------------------------===//

// Convert constant operations to LLVM constants
struct ConstantOpConversion : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the constant value
    auto constAttr = op.getValue();

    // Get the result type
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());

    // Handle different attribute types
    if (auto intAttr = dyn_cast<IntegerAttr>(constAttr)) {
      // Convert integer attribute
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, resultType, intAttr);
      return success();
    } else if (auto floatAttr = dyn_cast<FloatAttr>(constAttr)) {
      // Convert float attribute
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, resultType, floatAttr);
      return success();
    } else if (auto boolAttr = dyn_cast<BoolAttr>(constAttr)) {
      // Convert bool attribute to i1
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, resultType,
          rewriter.getIntegerAttr(resultType, boolAttr.getValue()));
      return success();
    }

    return failure();
  }
};

// Convert arith.index_cast to appropriate LLVM conversions
struct IndexCastOpConversion : public OpConversionPattern<arith::IndexCastOp> {
  using OpConversionPattern<arith::IndexCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get source and result types
    auto srcType = adaptor.getIn().getType();
    auto dstType = getTypeConverter()->convertType(op.getResult().getType());

    // Convert from index to specific integer type
    if (isa<LLVM::LLVMPointerType>(srcType) && isa<IntegerType>(dstType)) {
      rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, dstType,
                                                    adaptor.getIn());
      return success();
    }

    // Convert from specific integer type to index
    if (isa<IntegerType>(srcType) && isa<LLVM::LLVMPointerType>(dstType)) {
      rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, dstType,
                                                    adaptor.getIn());
      return success();
    }

    // Handle integer to integer casts
    if (isa<IntegerType>(srcType) && isa<IntegerType>(dstType)) {
      unsigned srcWidth = cast<IntegerType>(srcType).getWidth();
      unsigned dstWidth = cast<IntegerType>(dstType).getWidth();

      if (srcWidth < dstWidth) {
        // Sign extend if source is signed, zero extend otherwise
        rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(op, dstType, adaptor.getIn());
      } else if (srcWidth > dstWidth) {
        // Truncate
        rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, dstType,
                                                   adaptor.getIn());
      } else {
        // Same width, just pass through
        rewriter.replaceOp(op, adaptor.getIn());
      }
      return success();
    }

    return failure();
  }
};

// Convert arith.addi to LLVM add
struct AddIOpConversion : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::AddOp>(op, adaptor.getLhs(),
                                             adaptor.getRhs());
    return success();
  }
};

// Convert arith.muli to LLVM mul
struct MulIOpConversion : public OpConversionPattern<arith::MulIOp> {
  using OpConversionPattern<arith::MulIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MulIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::MulOp>(op, adaptor.getLhs(),
                                             adaptor.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Wafer Operation Conversion Patterns
//===----------------------------------------------------------------------===//

struct BarrierConversion : public OpConversionPattern<wafer::BarrierOp> {
  using OpConversionPattern<wafer::BarrierOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wafer::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __Barrier runtime function if not already declared
    /*
    void __Barrier()
    */

    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Declare the function
    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                "__Barrier", i8PtrTy, {});

    // Create the call to __Barrier
    auto call = rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange{i8PtrTy},
                                              "__Barrier", // funcPtr,
                                              ValueRange{});

    // Replace the op with the call
    rewriter.eraseOp(op);

    return success();
  }
};

struct RandGenOpConversion : public OpConversionPattern<wafer::RandGenOp> {
  using OpConversionPattern<wafer::RandGenOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wafer::RandGenOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto i8PtrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // void __RandGen(uint64_t *src0, uint64_t *src1, uint64_t *dst0,
    //                uint64_t *dst1, uint64_t *dst2, uint32_t byte_count,
    //                uint16_t fmt);
    SmallVector<Type, 7> argTypes = {i8PtrTy, i8PtrTy, i8PtrTy, i8PtrTy,
                                     i8PtrTy, i32Ty,   i16Ty};
    (void)triton::declareWaferRuntimeFunction(module, rewriter, loc, "__RandGen",
                                      voidTy, argTypes);

    Value src0 =
        rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrTy, adaptor.getSrc0());
    Value src1 =
        rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrTy, adaptor.getSrc1());
    Value dst0 =
        rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrTy, adaptor.getDst0());
    Value dst1 =
        rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrTy, adaptor.getDst1());
    Value dst2 =
        rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrTy, adaptor.getDst2());
    Value byteCount = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(op.getElemNum()));
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        loc, i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, "__RandGen",
        ValueRange{src0, src1, dst0, dst1, dst2, byteCount, fmt});
    rewriter.eraseOp(op);
    return success();
  }
};


template <typename WaferOpT, const char *funcPrefix>
struct AtomicBarrierOpConversion : public OpConversionPattern<WaferOpT> {
  using OpConversionPattern<WaferOpT>::OpConversionPattern;
  using OpAdaptor = typename WaferOpT::Adaptor;

  LogicalResult
  matchAndRewrite(WaferOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->template getParentOfType<ModuleOp>();
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                funcPrefix, i8PtrTy, {});

    auto call = rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange{i8PtrTy},
                                              funcPrefix, // funcPtr,
                                              ValueRange{});

    // erase the op
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename WaferOpT, const char *funcPrefix>
struct Rdma4dOpConversion : public OpConversionPattern<WaferOpT> {
  using OpConversionPattern<WaferOpT>::OpConversionPattern;
  using OpAdaptor = typename WaferOpT::Adaptor;

  LogicalResult
  matchAndRewrite(WaferOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();
    auto voidType = rewriter.getType<LLVM::LLVMVoidType>();
    auto LLVMPtrType = rewriter.getType<LLVM::LLVMPointerType>();
    auto i32Type = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type> argTypes = {
        LLVMPtrType, // dest
        LLVMPtrType, // src
        i32Type,     // elem_count
        i32Type,     // stride0
        i32Type,     // iteration0
        i32Type,     // stride1
        i32Type,     // iteration1
        i32Type,     // stride2
        i32Type,     // iteration2
        i32Type      // fmt
    };

    // Declare the function
    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, loc,
                                                funcPrefix, voidType, argTypes);

    Value dstPtr = rewriter.create<LLVM::IntToPtrOp>(loc, LLVMPtrType,
                                                     adaptor.getTarget());
    Value srcPtr = rewriter.create<LLVM::IntToPtrOp>(loc, LLVMPtrType,
                                                     adaptor.getSource());
    Value elemCount = adaptor.getElemCount();
    Value strides[3] = {adaptor.getStride0(), adaptor.getStride1(),
                        adaptor.getStride2()};
    Value iterations[3] = {adaptor.getIteration0(), adaptor.getIteration1(),
                           adaptor.getIteration2()};
    Value fmt =
        rewriter.create<LLVM::ConstantOp>(loc, i32Type, op.getFmtAttr());

    // Create the call to __Rdma4d/__Wdma4d
    auto call = rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{}, funcPrefix,
        ValueRange{dstPtr, srcPtr, elemCount, strides[0], iterations[0],
                   strides[1], iterations[1], strides[2], iterations[2], fmt});

    return success();
  }
};

template <typename WaferOpT, const char *funcPrefix>
struct Rdma1dOpConversion : public OpConversionPattern<WaferOpT> {
  using OpConversionPattern<WaferOpT>::OpConversionPattern;
  using OpAdaptor = typename WaferOpT::Adaptor;

  LogicalResult
  matchAndRewrite(WaferOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();
    auto voidType = rewriter.getType<LLVM::LLVMVoidType>();
    auto LLVMPtrType = rewriter.getType<LLVM::LLVMPointerType>();
    auto i32Type = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type> argTypes = {
        LLVMPtrType, // dest
        LLVMPtrType, // src
        i32Type,     // elem_count
        i32Type      // fmt
    };

    // Declare the function
    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, loc,
                                                funcPrefix, voidType, argTypes);

    Value dstPtr = rewriter.create<LLVM::IntToPtrOp>(loc, LLVMPtrType,
                                                     adaptor.getTarget());
    Value srcPtr = rewriter.create<LLVM::IntToPtrOp>(loc, LLVMPtrType,
                                                     adaptor.getSource());
    Value elemCount = adaptor.getElemCount();
    Value fmt =
        rewriter.create<LLVM::ConstantOp>(loc, i32Type, op.getFmtAttr());

    // Create the call to __Rdma1d/__Wdma1d
    auto call = rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{}, funcPrefix,
        ValueRange{dstPtr, srcPtr, elemCount, fmt});

    return success();
  }
};

// Resolve wafer.remote_buffer to its destination address.
struct RemoteBufferOpConversion
    : public OpConversionPattern<wafer::RemoteBufferOp> {
  using OpConversionPattern<wafer::RemoteBufferOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wafer::RemoteBufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[4]);
    return success();
  }
};

// Convert wafer.remote_load to LLVM call to __Recv function
struct RemoteLoadOpConversion : public OpConversionPattern<wafer::RemoteLoadOp> {
  using OpConversionPattern<wafer::RemoteLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wafer::RemoteLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __Recv runtime function if not already declared
    // Signature:
    //   void __Recv(int64_t chip_x, int64_t chip_y, int64_t die_id,
    //              int64_t tile_id, void* dst,
    //              uint32_t elem_bytes, uint64_t data_size)
    auto i8PtrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = rewriter.getI64Type();
    auto i32Ty = rewriter.getI32Type();
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    // Types for function declaration
    SmallVector<Type, 7> argTypes = {
        i64Ty,   // remote_chip_id_x
        i64Ty,   // remote_chip_id_y
        i64Ty,   // remote_die_id
        i64Ty,   // remote_tile_id
        i8PtrTy, // dst
        i32Ty,   // elem_bytes
        i64Ty    // data_size
    };

    // Declare the function with void return type
    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, loc,
                                                recvFuncName, voidTy, argTypes);

    // Get the operands and convert dst to i8*
    Value chipX = adaptor.getOperands()[0];
    Value chipY = adaptor.getOperands()[1];
    Value dieId = adaptor.getOperands()[2];
    Value tileId = adaptor.getOperands()[3];
    Value dstAddr = adaptor.getOperands()[4];
    Value elemBytes = adaptor.getOperands()[5];
    Value dataSize = adaptor.getOperands()[6];

    // Convert destination address (i64) directly to pointer.
    Value dst = rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrTy, dstAddr);

    // Create the call to __Recv (void function, so empty TypeRange)
    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, recvFuncName,
        ValueRange{chipX, chipY, dieId, tileId, dst, elemBytes, dataSize});

    // wafer.remote_load has no results, just erase it
    rewriter.eraseOp(op);

    return success();
  }
};

// Convert wafer.remote_store to LLVM call to __Send function
struct RemoteStoreOpConversion : public OpConversionPattern<wafer::RemoteStoreOp> {
  using OpConversionPattern<wafer::RemoteStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wafer::RemoteStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __Send runtime function if not already declared
    // Signature:
    //   void __Send(int64_t chip_x, int64_t chip_y, int64_t die_id,
    //              int64_t tile_id, void* dst, void* src,
    //              uint32_t elem_bytes, uint64_t data_size)
    auto i8PtrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = rewriter.getI64Type();
    auto i32Ty = rewriter.getI32Type();
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    // Types for function declaration
    SmallVector<Type, 8> argTypes = {
        i64Ty,   // remote_chip_id_x
        i64Ty,   // remote_chip_id_y
        i64Ty,   // remote_die_id
        i64Ty,   // remote_tile_id
        i8PtrTy, // dst
        i8PtrTy, // src
        i32Ty,   // elem_bytes
        i64Ty    // data_size
    };

    // Declare the function with void return type
    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, loc,
                                                sendFuncName, voidTy, argTypes);

    // Get the operands and convert dst/src to i8*
    Value chipX = adaptor.getOperands()[0];
    Value chipY = adaptor.getOperands()[1];
    Value dieId = adaptor.getOperands()[2];
    Value tileId = adaptor.getOperands()[3];
    Value dstAddr = adaptor.getOperands()[4];
    Value src = adaptor.getOperands()[5];
    Value elemBytes = adaptor.getOperands()[6];
    Value dataSize = adaptor.getOperands()[7];

    // Convert destination and source addresses (i64) directly to pointers.
    Value dst = rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrTy, dstAddr);
    src = rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrTy, src);

    // Create the call to __Send (void function, so empty TypeRange)
    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, sendFuncName,
        ValueRange{chipX, chipY, dieId, tileId, dst, src, elemBytes, dataSize});

    // wafer.remote_store has no results, just erase it
    rewriter.eraseOp(op);

    return success();
  }
};

template <typename WaferOpT, const char *funcPrefix>
struct RdmaWdmaOpConversion : public OpConversionPattern<WaferOpT> {
  using OpConversionPattern<WaferOpT>::OpConversionPattern;
  using OpAdaptor = typename WaferOpT::Adaptor;

  LogicalResult
  matchAndRewrite(WaferOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the __Rdma runtime function if not already declared
    auto i8PtrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = rewriter.getI32Type();
    auto i32PtrTy = LLVM::LLVMPointerType::get(ctx);

    // Types for function declaration
    SmallVector<Type, 5> argTypes = {
        i8PtrTy,  // src
        i8PtrTy,  // target
        i32PtrTy, // src_shape array
        i32PtrTy, // src_strides array
        i32PtrTy, // dst_shape array
        i32PtrTy, // dst_strides array
        i32Ty,    // rank
        i32Ty,    // elemBytes
        i32Ty     // fmt
    };

    // Declare the function
    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, loc,
                                                funcPrefix, i8PtrTy, argTypes);

    // Get the operands
    Value src = adaptor.getSource();
    src = rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrTy, src);

    Value target = adaptor.getTarget();
    target = rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrTy, target);

    // Create arrays for shapes and strides

    // Create arrays for shapes and strides
    Value srcShapeArray = indexValueArrayToInt32ValueArray(
        rewriter, loc, adaptor.getSrcShape(), op);
    Value srcStridesArray = indexValueArrayToInt32ValueArray(
        rewriter, loc, adaptor.getSrcStrides(), op);
    Value dstShapeArray = indexValueArrayToInt32ValueArray(
        rewriter, loc, adaptor.getDstShape(), op);
    Value dstStridesArray = indexValueArrayToInt32ValueArray(
        rewriter, loc, adaptor.getDstStrides(), op);

    // Handle rank attribute
    Value rank = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(op.getRank()));

    // Handle elem byte attribute
    Value elemBytes = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(op.getElemBytes()));

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(op.getFmt()));

    // Create the call to __Rdma
    auto call = rewriter.create<LLVM::CallOp>(
        loc, TypeRange{i8PtrTy}, funcPrefix,
        ValueRange{src, target, srcShapeArray, srcStridesArray, dstShapeArray,
                   dstStridesArray, rank, elemBytes, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Convert wafer.mask_move to LLVM call to __MaskMove function
struct MaskMoveOpConversion : public OpConversionPattern<wafer::MaskMoveOp> {
  using OpConversionPattern<wafer::MaskMoveOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wafer::MaskMoveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __MaskMove runtime function if not already declared
    // Signature: void* __MaskMove(void* source, void* target, uint32_t
    // elem_count, int32_t* masks, uint32_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i32PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Types for function declaration
    SmallVector<Type, 5> argTypes = {
        i8PtrTy,  // source
        i8PtrTy,  // target
        i32Ty,    // elem_count
        i32PtrTy, // masks
        i32Ty     // fmt
    };

    // Declare the function
    Value funcPtr = triton::declareWaferRuntimeFunction(
        module, rewriter, op.getLoc(), "__MaskMove", i8PtrTy, argTypes);

    // Get the operands
    Value src = adaptor.getSource();

    // Need to bitcast src to i8*
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);

    Value target = adaptor.getTarget();

    // Need to bitcast src to i8*
    target = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, target);
    Value elemCount = adaptor.getElemCount();
    elemCount = castIndexToInt32(rewriter, op->getLoc(), elemCount);

    // Handle mask arrays
    Value mask = adaptor.getMask();

    // Need to bitcast src to i8*
    mask = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, mask);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getFmt()));

    // Create the call to __MaskMove
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__MaskMove", // funcPtr,
        ArrayRef<Value>{src, target, elemCount, mask, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

template <typename WaferOpT, const char *funcPrefix>
struct TransformOpConversion : public OpConversionPattern<WaferOpT> {
  using OpConversionPattern<WaferOpT>::OpConversionPattern;
  using OpAdaptor = typename WaferOpT::Adaptor;

  LogicalResult
  matchAndRewrite(WaferOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature:
    // __Transpose(uint64_t *src, uint64_t *dst, int32_t *src_shape, int32_t
    // *dst_shape, uint16_t fmt)

    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32PtrTy, i32PtrTy,
                                      i16Ty};

    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value src = adaptor.getSource();
    // Need to bitcast src to i8*
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);
    Value dst = adaptor.getTarget();
    // Need to bitcast src to i8*
    dst = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, dst);

    // Convert shape attribute to Value
    ArrayRef<int32_t> srcShape = adaptor.getSrcShape();
    ArrayRef<int32_t> dstShape = adaptor.getDstShape();

    // Get shape llvm array
    auto srcArray =
        int32ArrayToInt32ValueArray(rewriter, op.getLoc(), srcShape, op);
    auto dstArray =
        int32ArrayToInt32ValueArray(rewriter, op.getLoc(), dstShape, op);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{src, dst, srcArray, dstArray, fmt});

    // Erase the old op
    rewriter.eraseOp(op);

    return success();
  }
};

struct GatherScatterOpConversion
    : public OpConversionPattern<wafer::GatherScatter> {
  using OpConversionPattern<wafer::GatherScatter>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wafer::GatherScatter op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __GatherScatter runtime function if not already declared
    /*
    void __GatherScatter(uint64_t *src, uint64_t *dst, uint32_t bytes,
                     uint32_t src_strideN, uint32_t src_strideH,
                     uint32_t src_strideW, uint32_t src_iterN,
                     uint32_t src_iterH, uint32_t src_iterW,
                     uint32_t dst_strideN, uint32_t dst_strideH,
                     uint32_t dst_strideW, uint32_t dst_iterN,
                     uint32_t dst_iterH, uint32_t dst_ite_W)
    */
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i32PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Types for function declaration
    SmallVector<Type, 5> argTypes = {
        i8PtrTy, // src
        i8PtrTy, // dst
        i32Ty,   // bytes
        i32Ty,   // src_StrideN
        i32Ty,   // src_StrideH
        i32Ty,   // src_StrideW
        i32Ty,   // dst_StrideN
        i32Ty,   // dst_StrideH
        i32Ty,   // dst_StrideW
        i32Ty,   // src_IterN
        i32Ty,   // src_IterH
        i32Ty,   // src_IterW
        i32Ty,   // dst_IterN
        i32Ty,   // dst_IterH
        i32Ty    // dst_IterW
    };

    // Declare the function
    Value funcPtr = triton::declareWaferRuntimeFunction(
        module, rewriter, loc, "__GatherScatter", i8PtrTy, argTypes);

    // Get the operands
    Value src = adaptor.getSource();
    src = rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrTy, src);

    // Get the operands
    Value dst = adaptor.getTarget();
    dst = rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrTy, dst);

    // Get bytes
    auto bytes =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getBytes());

    // Get strides
    auto srcStrideN =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getSrcStrideN());
    auto srcStrideH =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getSrcStrideH());
    auto srcStrideW =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getSrcStrideW());
    auto dstStrideN =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getDstStrideN());
    auto dstStrideH =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getDstStrideH());
    auto dstStrideW =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getDstStrideW());

    // Get iterator
    auto srcIterN =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getSrcIterN());
    auto srcIterH =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getSrcIterH());
    auto srcIterW =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getSrcIterW());
    auto dstIterN =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getDstIterN());
    auto dstIterH =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getDstIterH());
    auto dstIterW =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getDstIterW());

    // Create the call to __GatherScatter
    auto call = rewriter.create<LLVM::CallOp>(
        loc, TypeRange{i8PtrTy}, "__GatherScatter", // funcPtr,
        ValueRange{src, dst, bytes, srcStrideN, srcStrideH, srcStrideW,
                   srcIterN, srcIterH, srcIterW, dstStrideN, dstStrideH,
                   dstStrideW, dstIterN, dstIterH, dstIterW});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

template <typename WaferOpT, const char *funcPrefix>
struct ArgMinMaxOpConversion : public OpConversionPattern<WaferOpT> {
  using OpConversionPattern<WaferOpT>::OpConversionPattern;
  using OpAdaptor = typename WaferOpT::Adaptor;

  LogicalResult
  matchAndRewrite(WaferOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature:

    // __ArgMinMax(uint64_t *src, uint64_t *dst0, uint64_t *dst1,
    //             uint32_t elem_count, uint16_t fmt)
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i8PtrTy, i32Ty, i16Ty};

    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value src = adaptor.getSrc();
    // Need to bitcast src to i8*
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);

    // Convert results
    Value value = adaptor.getValue();
    Value index = adaptor.getIndex();
    // Need to bitcast `value` and `index` to i8*
    value = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, value);
    index = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, index);

    // Get elem_count operand, convert Index to I32
    Value elemCount = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getElemCount()));

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{src, value, index, elemCount, fmt});

    // Erase the old op
    rewriter.eraseOp(op);

    return success();
  }
};

// Convert wafer.binary op to LLVM call
template <typename WaferOpT, const char *funcPrefix>
struct ReduceOpConversion : public OpConversionPattern<WaferOpT> {
  using OpConversionPattern<WaferOpT>::OpConversionPattern;
  using OpAdaptor = typename WaferOpT::Adaptor;

  LogicalResult
  matchAndRewrite(WaferOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature:
    // __ReduceSum(uint64_t *src, uint64_t *dst, uint32_t dim, uint16_t src_n,
    // uint16_t src_h, uint16_t src_w, uint16_t src_c, uint16_t fmt)
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32Ty, i16Ty,
                                      i16Ty,   i16Ty,   i16Ty, i16Ty};

    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value src = adaptor.getSrc();
    // Need to bitcast src to i8*
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);
    Value srcB = adaptor.getSrc();
    Value dst = adaptor.getDst();
    // Need to bitcast src to i8*
    dst = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, dst);

    // Convert dim attribute to Value
    Value dim = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getDim()));

    // Convert shape attribute to Value
    Value shape_n =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[0]);
    Value shape_h =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[1]);
    Value shape_w =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[2]);
    Value shape_c =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[3]);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{src, dst, dim, shape_n, shape_h, shape_w, shape_c,
                        fmt});

    // Erase the old op
    rewriter.eraseOp(op);

    return success();
  }
};

// Convert wafer.elementwise op to LLVM call
template <typename WaferOpT, const char *funcPrefix>
struct ElementWiseOpConversion : public OpConversionPattern<WaferOpT> {
  using OpConversionPattern<WaferOpT>::OpConversionPattern;
  using OpAdaptor = typename WaferOpT::Adaptor;
  // using OpConversionPattern<WaferOpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WaferOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void* __Add(void* a, void* b, void* out, uint32_t elem_count,
    // uint32_t rnd_mode, uint32_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i8PtrTy,

                                      i32Ty,   i32Ty,   i32Ty};

    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value srcA = adaptor.getInput0();
    // Need to bitcast src to i8*
    srcA = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcA);
    Value srcB = adaptor.getInput1();
    // Need to bitcast src to i8*
    srcB = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcB);
    Value out = adaptor.getOut();
    // Need to bitcast src to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand, convert Index to I32
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle round attribute
    Value rnd_mode = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getRndMode()));

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{srcA, srcB, out, elemCount, rnd_mode, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

template <typename WaferOpT, const char *funcPrefix>
struct UnaryOpConversion : public OpConversionPattern<WaferOpT> {
  using OpConversionPattern<WaferOpT>::OpConversionPattern;
  using OpAdaptor = typename WaferOpT::Adaptor;

  LogicalResult
  matchAndRewrite(WaferOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void* __Abs(void* src, void* dst, uint32_t elem_count,
    // uint16_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32Ty, i16Ty};

    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value input = adaptor.getInput();
    // Need to bitcast src to i8*
    input = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, input);
    Value out = adaptor.getOut();
    // Need to bitcast out to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand, convert Index to I32
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{input, out, elemCount, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// FIXME: Use trait to refactor the BinaryVSOpConversion and
// ElementWiseOpConversion
template <typename WaferOpT, const char *funcPrefix>
struct BinaryVSOpConversion : public OpConversionPattern<WaferOpT> {
  using OpConversionPattern<WaferOpT>::OpConversionPattern;
  using OpAdaptor = typename WaferOpT::Adaptor;

  LogicalResult
  matchAndRewrite(WaferOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void* __Add(void* a, void* b, void* out, uint32_t elem_count,
    // uint32_t rnd_mode, uint32_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i32Ty, i8PtrTy,
                                      i32Ty,   i32Ty, i32Ty};

    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value srcA = adaptor.getInput0();
    // Need to bitcast src to i8*
    srcA = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcA);

    Value srcB = adaptor.getValue();

    Value out = adaptor.getOut();
    // Need to bitcast src to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand, convert Index to I32
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle round attribute
    Value rnd_mode = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getRndMode()));

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{srcA, srcB, out, elemCount, rnd_mode, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

template <typename WaferOpT, const char *funcPrefix>
struct BinaryLogicVVOpConversion : public OpConversionPattern<WaferOpT> {
  using OpConversionPattern<WaferOpT>::OpConversionPattern;
  using OpAdaptor = typename WaferOpT::Adaptor;

  LogicalResult
  matchAndRewrite(WaferOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void* __XorVV(void* a, void* b, void* out, uint32_t
    // elem_count, uint32_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {
        i8PtrTy, // src0_addr
        i8PtrTy, // src1_addr
        i8PtrTy, // dst_addr
        i32Ty,   // elem_count
        i32Ty    // fmt
    };

    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value srcA = adaptor.getInput0();
    // Need to bitcast src to i8*
    srcA = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcA);
    Value srcB = adaptor.getInput1();
    // Need to bitcast src to i8*
    srcB = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcB);
    Value out = adaptor.getOut();
    // Need to bitcast src to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand, convert Index to I32
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{srcA, srcB, out, elemCount, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

template <typename WaferOpT, const char *funcPrefix>
struct UnaryBoolLogicVOpConversion : public OpConversionPattern<WaferOpT> {
  using OpConversionPattern<WaferOpT>::OpConversionPattern;
  using OpAdaptor = typename WaferOpT::Adaptor;

  LogicalResult
  matchAndRewrite(WaferOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void* __BoolNotV(void* src, void* dst, uint32_t elem_count);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {
        i8PtrTy, // src_addr
        i8PtrTy, // dst_addr
        i32Ty    // elem_count
    };

    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value src = adaptor.getInput();
    // Need to bitcast src to i8*
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);

    Value out = adaptor.getOut();
    // Need to bitcast dest to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand, convert Index to I32
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{src, out, elemCount});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

template <typename WaferOpT, const char *funcPrefix>
struct BinaryBoolLogicVOpConversion : public OpConversionPattern<WaferOpT> {
  using OpConversionPattern<WaferOpT>::OpConversionPattern;
  using OpAdaptor = typename WaferOpT::Adaptor;

  LogicalResult
  matchAndRewrite(WaferOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void* __BoolAndV(void* a, void* b, void* out, uint32_t
    // elem_count);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {
        i8PtrTy, // src0_addr
        i8PtrTy, // src1_addr
        i8PtrTy, // dst_addr
        i32Ty    // elem_count
    };

    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value srcA = adaptor.getInput0();
    // Need to bitcast src to i8*
    srcA = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcA);
    Value srcB = adaptor.getInput1();
    // Need to bitcast src to i8*
    srcB = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcB);
    Value out = adaptor.getOut();
    // Need to bitcast src to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand, convert Index to I32
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{srcA, srcB, out, elemCount});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

template <typename RelationVVOp, const char *funcPrefix>
struct RelationVVOpConversion : public OpConversionPattern<RelationVVOp> {
  using OpConversionPattern<RelationVVOp>::OpConversionPattern;
  using OpAdaptor = typename RelationVVOp::Adaptor;
  // using OpConversionPattern<WaferOpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RelationVVOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void __BoolLessEqualVV(uint64_t *src0, uint64_t *src1,
    // uint64_t *dst, uint32_t elem_count, uint16_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i8PtrTy, i32Ty, i16Ty};

    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value srcA = adaptor.getInput0();
    // Need to bitcast src to i8*
    srcA = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcA);
    Value srcB = adaptor.getInput1();
    // Need to bitcast src to i8*
    srcB = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcB);
    Value out = adaptor.getOut();
    // Need to bitcast src to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{srcA, srcB, out, elemCount, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// FIXME: Use trait to refactor the RelationVSOpConversion and
// ElementWiseOpConversion
template <typename WaferOpT, const char *funcPrefix>
struct RelationVSOpConversion : public OpConversionPattern<WaferOpT> {
  using OpConversionPattern<WaferOpT>::OpConversionPattern;
  using OpAdaptor = typename WaferOpT::Adaptor;

  LogicalResult
  matchAndRewrite(WaferOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void __BoolEqualVS(uint64_t *src0, uint32_t src1, uint64_t
    // *dst,uint32_t elem_count, uint16_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i32Ty, i8PtrTy, i32Ty, i32Ty};

    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value srcA = adaptor.getInput0();
    // Need to bitcast src to i8*
    srcA = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcA);

    Value srcB = adaptor.getValue();

    Value out = adaptor.getOut();
    // Need to bitcast src to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand, convert Index to I32
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{srcA, srcB, out, elemCount, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Convert wafer.ZeroPointConvertOp op to LLVM
template <typename ZeroPointConvertOp, const char *funcPrefix>
struct ZeroPointConvertOpConversion
    : public OpConversionPattern<ZeroPointConvertOp> {
  using OpConversionPattern<ZeroPointConvertOp>::OpConversionPattern;
  using OpAdaptor = typename ZeroPointConvertOp::Adaptor;

  LogicalResult
  matchAndRewrite(ZeroPointConvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void __INT8_FP32(uint64_t *src, uint64_t *dst, uint32_t
    // zero_point, uint32_t elem_count);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32Ty, i32Ty};

    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value input = adaptor.getSrc();
    Value output = adaptor.getDst();
    Value zeroPoint = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, adaptor.getZeroPointAttr());
    Value elemCount = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, adaptor.getElemCountAttr());

    // Bitcast all pointers to i8*
    input = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, input);
    output = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, output);

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{input, output, zeroPoint, elemCount});

    rewriter.eraseOp(op);
    return success();
  }
};

// Convert wafer.NormalConvertOp op to LLVM
template <typename NormalConvertOp, const char *funcPrefix>
struct NormalConvertOpConversion : public OpConversionPattern<NormalConvertOp> {
  using OpConversionPattern<NormalConvertOp>::OpConversionPattern;
  using OpAdaptor = typename NormalConvertOp::Adaptor;

  LogicalResult
  matchAndRewrite(NormalConvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void __FP16_FP32(uint64_t *src, uint64_t *dst, uint32_t
    // elem_count);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32Ty};

    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value input = adaptor.getInput();
    Value output = adaptor.getOutput();
    Value elemCount = adaptor.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Bitcast all pointers to i8*
    input = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, input);
    output = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, output);

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{input, output, elemCount});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Convert wafer.RoundConvertOp op to LLVM
template <typename RoundConvertOp, const char *funcPrefix>
struct RoundConvertOpConversion : public OpConversionPattern<RoundConvertOp> {
  using OpConversionPattern<RoundConvertOp>::OpConversionPattern;
  using OpAdaptor = typename RoundConvertOp::Adaptor;

  LogicalResult
  matchAndRewrite(RoundConvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void __INT16_FP32(uint64_t *src, uint64_t *dst, uint32_t
    // elem_count, RND_MODE round);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32Ty, i16Ty};

    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value input = adaptor.getInput();
    Value output = adaptor.getOutput();
    Value elemCount = adaptor.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);
    Value rnd_mode = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getRndMode()));

    // Bitcast all pointers to i8*
    input = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, input);
    output = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, output);

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{input, output, elemCount, rnd_mode});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

template <typename MXFPScaleOp, const char *funcPrefix>
struct MXFPScaleOpConversion : public OpConversionPattern<MXFPScaleOp> {
  using OpConversionPattern<MXFPScaleOp>::OpConversionPattern;
  using OpAdaptor = typename MXFPScaleOp::Adaptor;

  LogicalResult
  matchAndRewrite(MXFPScaleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 5> argTypes = {
        i8PtrTy, // src
        i8PtrTy, // scale
        i8PtrTy, // dst
        i32Ty,   // elem_count
    };

    // Declare the function
    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                funcPrefix, i8PtrTy, argTypes);

    // Get the operands
    Value src = adaptor.getSrc();
    Value scale = adaptor.getScale();
    Value dst = adaptor.getDst();

    // Need to bitcast src to i8*
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);
    scale = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, scale);
    dst = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, dst);

    Value elemCount = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, adaptor.getElemCountAttr());

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{src, scale, dst, elemCount});

    op->erase();
    return success();
  }
};

struct BitToFPOpConversion : public OpConversionPattern<wafer::Bit2FpOp> {
  using OpConversionPattern<wafer::Bit2FpOp>::OpConversionPattern;
  using OpAdaptor = wafer::Bit2FpOp::Adaptor;

  LogicalResult
  matchAndRewrite(wafer::Bit2FpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void __Bit2Fp(uint64_t *src, uint64_t *target, uint32_t
    // elem_count, uint16_t fmt)
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32Ty, i16Ty};

    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                "__Bit2Fp", i8PtrTy, argTypes);

    // Convert operands
    Value input = adaptor.getSrc();
    Value output = adaptor.getTarget();
    Value elemCount = adaptor.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Bitcast all pointers to i8*
    input = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, input);
    output = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, output);

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__Bit2Fp", // funcPtr,
        ArrayRef<Value>{input, output, elemCount, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Convert wafer.channel_norm op
struct ChannelNormOpConversion : public OpConversionPattern<wafer::ChannelNormOp> {
  using OpConversionPattern<wafer::ChannelNormOp>::OpConversionPattern;
  using OpAdaptor = typename wafer::ChannelNormOp::Adaptor;

  LogicalResult
  matchAndRewrite(wafer::ChannelNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature:
    // __ChannelNorm(uint64_t *src, uint64_t *dst, uint16_t n,
    // uint16_t h, uint16_t w, uint16_t c, uint16_t c0, uint16_t
    // dtype_size)
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i16Ty, i16Ty,
                                      i16Ty,   i16Ty,   i16Ty, i16Ty};

    Value funcPtr = triton::declareWaferRuntimeFunction(
        module, rewriter, op.getLoc(), "__ChannelNorm", i8PtrTy, argTypes);

    // Convert operands
    Value src = adaptor.getSrc();
    // Need to bitcast src to i8*
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);
    Value dst = adaptor.getDst();
    // Need to bitcast dst to i8*
    dst = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, dst);

    // Convert shape attribute to Value
    Value shape_n =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[0]);
    Value shape_h =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[1]);
    Value shape_w =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[2]);
    Value shape_c =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[3]);

    // Convert c0_align attribute to Value
    Value c0Align = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getC0Align()));

    // Convert dtype_size attribute to Value
    Value dtypeSize = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI32IntegerAttr(op.getDtypeSize()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__ChannelNorm", // funcPtr,
        ArrayRef<Value>{src, dst, shape_n, shape_h, shape_w, shape_c, c0Align,
                        dtypeSize});

    // Erase the old op
    rewriter.eraseOp(op);

    return success();
  }
};

struct DechannelNormOpConversion
    : public OpConversionPattern<wafer::DechannelNormOp> {
  using OpConversionPattern<wafer::DechannelNormOp>::OpConversionPattern;
  using OpAdaptor = typename wafer::DechannelNormOp::Adaptor;

  LogicalResult
  matchAndRewrite(wafer::DechannelNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature:
    // __DechannelNorm(uint64_t *src, uint64_t *dst, uint16_t n,
    // uint16_t h, uint16_t w, uint16_t c, uint16_t c0, uint16_t
    // dtype_size)
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i16Ty, i16Ty,
                                      i16Ty,   i16Ty,   i16Ty, i16Ty};

    Value funcPtr = triton::declareWaferRuntimeFunction(
        module, rewriter, op.getLoc(), "__DechannelNorm", i8PtrTy, argTypes);

    // Convert operands
    Value src = adaptor.getSrc();
    // Need to bitcast src to i8*
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);
    Value dst = adaptor.getDst();
    // Need to bitcast dst to i8*
    dst = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, dst);

    // Convert shape attribute to Value
    Value shape_n =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[0]);
    Value shape_h =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[1]);
    Value shape_w =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[2]);
    Value shape_c =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[3]);

    // Convert c0_align attribute to Value
    Value c0Align = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getC0Align()));

    // Convert dtype_size attribute to Value
    Value dtypeSize = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI32IntegerAttr(op.getDtypeSize()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__DechannelNorm", // funcPtr,
        ArrayRef<Value>{src, dst, shape_n, shape_h, shape_w, shape_c, c0Align,
                        dtypeSize});

    // Erase the old op
    rewriter.eraseOp(op);

    return success();
  }
};

// Convert wafer.gemm to LLVM call to __Gemm function
struct GemmOpConversion : public OpConversionPattern<wafer::GemmOp> {
  using OpConversionPattern<wafer::GemmOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wafer::GemmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __Gemm runtime function if not already declared
    // Signature: void __Gemm(int64_t* srcA, int64_t *srcB, int64_t * srcBias,
    // int64_t *dst, int32_t *dims, bool enPsum, int64_t *psum, bool enTransA,
    // bool enTransB, int64_t batchSizeA, int64_t batchSizeB, bool enLeakyRelu,
    // bool enBias,bool enNegScale, int64_t *negScale, bool enPosScale, int64_t
    // *posScale, int64_t srcFmt, int64_t dstFmt)
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i64Ty = rewriter.getI64Type();
    auto i32PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i1Ty = rewriter.getI1Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {
        i8PtrTy,  // srcA
        i8PtrTy,  // srcB
        i8PtrTy,  // srcBias
        i8PtrTy,  // dst
        i32PtrTy, // dims
        i1Ty,     // enPsum
        i8PtrTy,  // psum
        i1Ty,     // enTransA
        i1Ty,     // enTransB
        i32Ty,    // batchSizeA
        i32Ty,    // batchSizeB
        i32Ty,    // reluMode
        i1Ty,     // enBias
        i1Ty,     // enNegScale
        i8PtrTy,  // negScale
        i1Ty,     // enPosScale
        i8PtrTy,  // posScale
        i32Ty,    // srcFmt
        i32Ty     // dstFmt
    };

    // Declare the function
    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                "__Gemm", i8PtrTy, argTypes);

    // Convert operands
    Value srcA = adaptor.getSrcA();
    Value srcB = adaptor.getSrcB();
    Value srcBias = adaptor.getSrcBias();
    Value dst = adaptor.getDst();

    Value psumAddr = adaptor.getPsumAddr();
    Value srcNegScale = adaptor.getSrcNegScale();
    Value srcPosScale = adaptor.getSrcPosScale();

    // Bitcast all pointers to i8*
    srcA = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcA);
    srcB = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcB);
    srcBias = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcBias);
    dst = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, dst);
    psumAddr =
        rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, psumAddr);
    srcNegScale =
        rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcNegScale);
    srcPosScale =
        rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcPosScale);

    // Handle dims array - need to convert from attribute to runtime array
    auto dimsAttr = op.getDims();
    SmallVector<int32_t> dimsValues;
    for (auto dimAttr : dimsAttr)
      dimsValues.push_back(cast<IntegerAttr>(dimAttr).getInt());

    // Allocate memory for the dims array
    Value rank = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i64Ty, rewriter.getI64IntegerAttr(dimsValues.size()));

    auto dimsArrayI32Ptr =
        int32ArrayToInt32ValueArray(rewriter, op->getLoc(), dimsValues, op);

    // Convert boolean attributes
    Value transA = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i1Ty, rewriter.getBoolAttr(op.getTransSrcA()));
    Value transB = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i1Ty, rewriter.getBoolAttr(op.getTransSrcB()));
    Value enPSum = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i1Ty, rewriter.getBoolAttr(op.getEnPsum()));
    Value reluMode = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getReluMode()));
    Value enBias = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i1Ty, rewriter.getBoolAttr(op.getEnBias()));
    Value enNegScale = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i1Ty, rewriter.getBoolAttr(op.getEnNegScale()));
    Value enPosScale = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i1Ty, rewriter.getBoolAttr(op.getEnPosScale()));

    // Convert integer attributes
    Value batchA = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getBatchSrcA()));
    Value batchB = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getBatchSrcB()));
    Value srcFmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getSrcFmt()));
    Value dstFmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getDstFmt()));

    // Create the call to __Gemm
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__Gemm", // funcPtr,
        ArrayRef<Value>{srcA, srcB, srcBias, dst, dimsArrayI32Ptr, enPSum,
                        psumAddr, transA, transB, batchA, batchB, reluMode,
                        enBias, enNegScale, srcNegScale, enPosScale,
                        srcPosScale, srcFmt, dstFmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

struct SigmoidOpConversion : public OpConversionPattern<wafer::Sigmoid> {
  using OpConversionPattern<wafer::Sigmoid>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wafer::Sigmoid op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __Sigmoid runtime function if not already declared
    // Signature: void __Sigmoid(int64_t* src, int64_t *dst,
    // uint32_t elem_count, uint16_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32Ty, i16Ty};

    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                "__Sigmoid", i8PtrTy, argTypes);

    // Convert operands
    Value input = adaptor.getInput();
    Value output = adaptor.getOut();
    Value elemCount = adaptor.getElemCount();

    // Bitcast all pointers to i8*
    input = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, input);
    output = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, output);
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__Sigmoid", // funcPtr,
        ArrayRef<Value>{input, output, elemCount, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

struct GeluNoneOpConversion : public OpConversionPattern<wafer::GeluNone> {
  using OpConversionPattern<wafer::GeluNone>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wafer::GeluNone op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __GeluNone runtime function if not already declared
    // Signature: void __GeluNone(int64_t* src, int64_t *dst,
    // uint32_t elem_count, uint16_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32Ty, i16Ty};

    Value funcPtr = triton::declareWaferRuntimeFunction(
        module, rewriter, op.getLoc(), "__GeluNone", i8PtrTy, argTypes);

    // Convert operands
    Value input = adaptor.getInput();
    Value output = adaptor.getOut();
    Value elemCount = adaptor.getElemCount();

    // Bitcast all pointers to i8*
    input = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, input);
    output = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, output);
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__GeluNone", // funcPtr,
        ArrayRef<Value>{input, output, elemCount, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

struct GeluTanhOpConversion : public OpConversionPattern<wafer::GeluTanh> {
  using OpConversionPattern<wafer::GeluTanh>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wafer::GeluTanh op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __GeluTanh runtime function if not already declared
    // Signature: void __GeluTanh(int64_t* src, int64_t *imm, int64_t *dst,
    // uint32_t elem_count, uint16_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i8PtrTy, i32Ty, i16Ty};

    Value funcPtr = triton::declareWaferRuntimeFunction(
        module, rewriter, op.getLoc(), "__GeluTanh", i8PtrTy, argTypes);

    // Convert operands
    Value input = adaptor.getInput();
    Value imm = adaptor.getBuffer();
    Value output = adaptor.getOut();
    Value elemCount = adaptor.getElemCount();

    // Bitcast all pointers to i8*
    input = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, input);
    imm = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, imm);
    output = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, output);
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__GeluTanh", // funcPtr,
        ArrayRef<Value>{input, imm, output, elemCount, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Convert wafer.memset to LLVM call to __Memset function
struct MemsetOpConversion : public OpConversionPattern<wafer::MemsetOp> {
  using OpConversionPattern<wafer::MemsetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wafer::MemsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __Memset runtime function if not already declared
    // Signature: void* __Memset(void* dst, int64_t value, uint32_t rank,
    //                    int32_t* strides, int32_t* iterations, uint16_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i32PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 6> argTypes = {
        i8PtrTy,  // Spm addr
        i32Ty,    // value
        i32PtrTy, // src_shape array
        i32PtrTy, // src_strides array
        i32Ty,    // rank
        i16Ty     // fmt
    };

    // Declare the function
    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                "__Memset", i8PtrTy, argTypes);

    // Get operands
    Value dst = adaptor.getTarget();
    dst = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, dst);

    Value value = adaptor.getValue();

    // Handle strides and iterations arrays
    // Create arrays for shapes and strides
    Value dstShapeArray = indexValueArrayToInt32ValueArray(
        rewriter, loc, adaptor.getDstShape(), op);
    Value dstStridesArray = indexValueArrayToInt32ValueArray(
        rewriter, loc, adaptor.getDstStrides(), op);

    // Convert fmt attribute to Value
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    Value rank = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getRank()));

    // Create the call to __Memset
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__Memset", // funcPtr,
        ArrayRef<Value>{dst, value, dstShapeArray, dstStridesArray, rank, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Convert tt.get_program_id to LLVM call to __get_pid function
// Think this as Wafer special action. May can separate to a single pass or use
// wafer.get_program_id op
struct GetProgramIDConversion
    : public OpConversionPattern<triton::GetProgramIdOp> {
  using OpConversionPattern<triton::GetProgramIdOp>::OpConversionPattern;
  static uint32_t constexpr LAUNCH_GRID_RANK =
      mlir::triton::getMaxEnumValForProgramIDDim() + 1;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __Memset runtime function if not already declared
    // Signature: uint32_t __get_pid(uint32_t);
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 6> argTypes = {
        i32Ty, // x: 0/y: 1/z: 2,
    };

    // Declare the function
    Value funcPtr = triton::declareWaferRuntimeFunction(module, rewriter, op.getLoc(),
                                                "__get_pid", i32Ty, argTypes);

    // Get operands
    auto axis = (uint32_t)op.getAxis();

    assert(axis < LAUNCH_GRID_RANK && "program_id expects "
                                      "axis to be either 0, "
                                      "1, or 2");

    // Convert fmt attribute to Value
    Value src = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(axis));

    // Create the call to __Memset
    auto call = rewriter.create<LLVM::CallOp>(op.getLoc(), i32Ty,
                                              "__get_pid", // funcPtr,
                                              ArrayRef<Value>{src});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

struct AssertConversion : public OpConversionPattern<mk::AssertOp> {
  using OpConversionPattern<mk::AssertOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mk::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    auto i32Ty = rewriter.getI32Type();
    auto context = rewriter.getContext();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto assertRef = getOrInsertAssert(rewriter, parentModule);
    auto message = op.getMessage();

    StringRef file = "unknown";
    int line = 0;
    int col = 0;
    if (auto fileLineColLoc = dyn_cast<FileLineColLoc>(loc)) {
      file = fileLineColLoc.getFilename();
      line = fileLineColLoc.getLine();
      col = fileLineColLoc.getColumn();
    }

    SmallVector<Type, 6> argTypes = {
        i32Ty, // x: 0/y: 1/z: 2,
    };
    Value funcPtr = triton::declareWaferRuntimeFunction(parentModule, rewriter, loc,
                                                "__get_pid", i32Ty, argTypes);
    auto xDim = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, 0);
    auto yDim = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, 1);
    auto zDim = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, 2);
    auto pidX = rewriter
                    .create<LLVM::CallOp>(loc, i32Ty,
                                          "__get_pid", // funcPtr,
                                          ArrayRef<Value>{xDim})
                    ->getResult(0);
    auto pidY = rewriter
                    .create<LLVM::CallOp>(loc, i32Ty,
                                          "__get_pid", // funcPtr,
                                          ArrayRef<Value>{yDim})
                    ->getResult(0);
    auto pidZ = rewriter
                    .create<LLVM::CallOp>(loc, i32Ty,
                                          "__get_pid", // funcPtr,
                                          ArrayRef<Value>{zDim})
                    ->getResult(0);

    llvm::SmallString<64> messageString(message), fileString(file);
    messageString.push_back('\0');
    fileString.push_back('\0');
    Value messageStringVal =
        LLVM::addStringToModule(loc, rewriter, "assertMessage_", messageString);
    Value fileStringVal =
        LLVM::addStringToModule(loc, rewriter, "assertFile_", fileString);

    auto lineValue = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, line);
    auto colValue = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, col);

    rewriter.create<LLVM::CallOp>(loc, getAssertType(context), assertRef,
                                  ValueRange{messageStringVal, fileStringVal,
                                             lineValue, colValue, pidX, pidY,
                                             pidZ});
    rewriter.eraseOp(op);
    return success();
  }

private:
  static LLVM::LLVMFunctionType getAssertType(MLIRContext *context) {
    auto llvmPtr = LLVM::LLVMPointerType::get(context);
    // Match CRT's void __Assert(const char *, ...).
    return LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context), llvmPtr,
                                      true);
  }

  static FlatSymbolRefAttr getOrInsertAssert(PatternRewriter &rewriter,
                                             ModuleOp module,
                                             StringRef funcName = "__Assert") {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
      return SymbolRefAttr::get(context, funcName);

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName,
                                      getAssertType(context));
    return SymbolRefAttr::get(context, funcName);
  }
};

// The conversion pass
class WaferToLLVMPass : public WaferToLLVMBase<WaferToLLVMPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<LLVM::LLVMDialect, wafer::WaferDialect, arith::ArithDialect,
                func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    // Setup LLVM lowering options object which should live across the call to
    // applyFull/PartialConversion.
    LowerToLLVMOptions options(context);
    options.useBarePtrCallConv = false;

    // Setup conversion target
    target.addLegalDialect<LLVM::LLVMDialect, memref::MemRefDialect,
                           arith::ArithDialect, scf::SCFDialect,
                           func::FuncDialect, math::MathDialect>();
    // Handle the wafer op to llvm.call and support kcore load/store op's spm
    // offset
    target.addIllegalDialect<triton::TritonDialect, linalg::LinalgDialect,
                             tensor::TensorDialect, affine::AffineDialect,
                             wafer::WaferDialect>();

    // Setup rewrite patterns
    RewritePatternSet patterns(context);

    // NOTE: LLVMTypeConverter should be enough for MLIR core dialects.
    LLVMTypeConverter llvmTypeConverter(context, options);

    // Add the Wafer to LLVM conversion patterns
    // clang-format off
    patterns.add</* INT8 */
                ZeroPointConvertOpConversion<wafer::INT8ToFP16Op, int8ToFp16FuncName>,
                ZeroPointConvertOpConversion<wafer::INT8ToBF16Op, int8ToBf16FuncName>,
                ZeroPointConvertOpConversion<wafer::INT8ToFP32Op, int8ToFp32FuncName>,
                ZeroPointConvertOpConversion<wafer::INT8ToTF32Op, int8ToTf32FuncName>,
                /* INT16 */
                 NormalConvertOpConversion<wafer::INT16ToFP16Op, int16ToFp16FuncName>,
                 RoundConvertOpConversion<wafer::INT16ToBF16Op, int16ToBf16FuncName>,
                 RoundConvertOpConversion<wafer::INT16ToFP32Op, int16ToFp32FuncName>,
                 RoundConvertOpConversion<wafer::INT16ToTF32Op, int16ToTf32FuncName>,
                 /* INT32 */
                 RoundConvertOpConversion<wafer::INT32ToFP16Op, int32ToFp16FuncName>,
                 RoundConvertOpConversion<wafer::INT32ToBF16Op, int32ToBf16FuncName>,
                 RoundConvertOpConversion<wafer::INT32ToFP32Op, int32ToFp32FuncName>,
                 RoundConvertOpConversion<wafer::INT32ToTF32Op, int32ToTf32FuncName>,
                 /* BF16 */
                 NormalConvertOpConversion<wafer::BF16ToINT8Op, bf16ToInt8FuncName>,
                 RoundConvertOpConversion<wafer::BF16ToINT16Op, bf16ToInt16FuncName>,
                 RoundConvertOpConversion<wafer::BF16ToINT32Op, bf16ToInt32FuncName>,
                 NormalConvertOpConversion<wafer::BF16ToFP16Op, bf16ToFp16FuncName>,
                 NormalConvertOpConversion<wafer::BF16ToFP32Op, bf16ToFp32FuncName>,
                 NormalConvertOpConversion<wafer::BF16ToTF32Op, bf16ToTf32FuncName>,
                 /* FP16 */
                 RoundConvertOpConversion<wafer::FP16ToINT8Op, fp16ToInt8FuncName>,
                 RoundConvertOpConversion<wafer::FP16ToINT16Op,fp16ToInt16FuncName>,
                 RoundConvertOpConversion<wafer::FP16ToINT32Op, fp16ToInt32FuncName>,
                 RoundConvertOpConversion<wafer::FP16ToBF16Op, fp16ToBf16FuncName>,
                 NormalConvertOpConversion<wafer::FP16ToFP32Op, fp16ToFp32FuncName>,
                 NormalConvertOpConversion<wafer::FP16ToTF32Op, fp16ToTf32FuncName>,
                 /* FP32 */
                 RoundConvertOpConversion<wafer::FP32ToINT8Op, fp32ToInt8FuncName>,
                 RoundConvertOpConversion<wafer::FP32ToINT16Op, fp32ToInt16FuncName>,
                 RoundConvertOpConversion<wafer::FP32ToINT32Op, fp32ToInt32FuncName>,
                 RoundConvertOpConversion<wafer::FP32ToFP16Op, fp32ToFp16FuncName>,
                 RoundConvertOpConversion<wafer::FP32ToBF16Op,fp32ToBf16FuncName>,
                 RoundConvertOpConversion<wafer::FP32ToTF32Op, fp32ToTf32FuncName>, // NOTE: No op used
                 /* TF32 */
                 RoundConvertOpConversion<wafer::TF32ToINT8Op, tf32ToInt8FuncName>,
                 RoundConvertOpConversion<wafer::TF32ToINT16Op, tf32ToInt16FuncName>,
                 RoundConvertOpConversion<wafer::TF32ToINT32Op, tf32ToInt32FuncName>,
                 NormalConvertOpConversion<wafer::TF32ToFP16Op, tf32ToFp16FuncName>,
                 RoundConvertOpConversion<wafer::TF32ToBF16Op, tf32ToBf16FuncName>,
                 NormalConvertOpConversion<wafer::TF32ToFP32Op, tf32ToFp32FuncName>,
                 /* MXFP */
                 NormalConvertOpConversion<wafer::FP8E4M3ToBF16Op, fp8E4M3ToBF16FuncName>,
                 NormalConvertOpConversion<wafer::FP8E4M3FNToBF16Op, fp8E4M3FNToBF16FuncName>,
                 NormalConvertOpConversion<wafer::FP8E5M2ToBF16Op, fp8E5M2ToBF16FuncName>,
                 NormalConvertOpConversion<wafer::FP4E2M1ToBF16Op, fp4E2M1ToBF16FuncName>,
                 NormalConvertOpConversion<wafer::FP8E4M3ToFP16Op, fp8E4M3ToFP16FuncName>,
                 NormalConvertOpConversion<wafer::FP8E4M3FNToFP16Op, fp8E4M3FNToFP16FuncName>,
                 NormalConvertOpConversion<wafer::FP8E5M2ToFP16Op, fp8E5M2ToFP16FuncName>,
                 NormalConvertOpConversion<wafer::FP4E2M1ToFP16Op, fp4E2M1ToFP16FuncName>,
                 MXFPScaleOpConversion<wafer::MXFPScaleBF16Op, MXFPScaleBF16FuncName>,
                 MXFPScaleOpConversion<wafer::MXFPScaleFP16Op, MXFPScaleFP16FuncName>,
                 ArgMinMaxOpConversion<wafer::ArgMaxOp, argMaxFuncName>,
                 ArgMinMaxOpConversion<wafer::ArgMinOp, argMinFuncName>,
                 ReduceOpConversion<wafer::ReduceSumOp,reduceSumFuncName>,
                 ReduceOpConversion<wafer::ReduceMaxOp,reduceMaxFuncName>,
                 ReduceOpConversion<wafer::ReduceMinOp,reduceMinFuncName>,
                 ReduceOpConversion<wafer::ReduceMulOp,reduceMulFuncName>,
                 ElementWiseOpConversion<wafer::AddVVOp, addVVFuncName>,
                 ElementWiseOpConversion<wafer::SubVVOp, subVVFuncName>,
                 ElementWiseOpConversion<wafer::MulVVOp, mulVVFuncName>,
                 ElementWiseOpConversion<wafer::DivVVOp, divVVFuncName>,
                 ElementWiseOpConversion<wafer::MaxVVOp, MaxVVFuncName>,
                 ElementWiseOpConversion<wafer::MinVVOp, MinVVFuncName>,
                 UnaryOpConversion<wafer::AbsVVOp, absVVFuncName>,
                 UnaryOpConversion<wafer::RsqrtVVOp, rsqrtVVFuncName>,
                 UnaryOpConversion<wafer::SqrtVVOp, sqrtVVFuncName>,
                 UnaryOpConversion<wafer::RecipVVOp, recipVVFuncName>,
                 UnaryOpConversion<wafer::NegVVOp, negVVFuncName>,
                 UnaryOpConversion<wafer::LnOp, lnFuncName>,
                 UnaryOpConversion<wafer::Log2Op, log2FuncName>,
                 UnaryOpConversion<wafer::ExpOp, expFuncName>,
                 UnaryOpConversion<wafer::Pow2Op, pow2FuncName>,
                 UnaryOpConversion<wafer::SinOp, sinFuncName>,
                 UnaryOpConversion<wafer::CosOp, cosFuncName>,
                 UnaryOpConversion<wafer::Tanh, tanhFuncName>,
                 BinaryVSOpConversion<wafer::AddVSOp, addVSFuncName>,
                 BinaryVSOpConversion<wafer::SubVSOp, subVSFuncName>,
                 BinaryVSOpConversion<wafer::MulVSOp, mulVSFuncName>,
                 BinaryVSOpConversion<wafer::DivVSOp, divVSFuncName>,
                 RelationVVOpConversion<wafer::BoolEqualVV, boolEqualVVFuncName>,
                 RelationVVOpConversion<wafer::BoolUnEqualVV, boolUnEqualVVFuncName>,
                 RelationVVOpConversion<wafer::BoolGreaterEqualVV, boolGreaterEqualVVFuncName>,
                 RelationVVOpConversion<wafer::BoolGreaterVV, boolGreaterVVFuncName>,
                 RelationVVOpConversion<wafer::BoolLessEqualVV, boolLessEqualVVFuncName>,
                 RelationVVOpConversion<wafer::BoolLessThenVV, boolLessThenVVFuncName>,
                 RelationVVOpConversion<wafer::EqualVV, equalVVFuncName>,
                 RelationVVOpConversion<wafer::UnEqualVV, unEqualVVFuncName>,
                 RelationVVOpConversion<wafer::GreaterEqualVV, greaterEqualVVFuncName>,
                 RelationVVOpConversion<wafer::GreaterVV, greaterVVFuncName>,
                 RelationVVOpConversion<wafer::LessEqualVV, lessEqualVVFuncName>,
                 RelationVVOpConversion<wafer::LessThenVV, lessThenVSFuncName>,
                 RelationVSOpConversion<wafer::BoolEqualVS, boolEqualVSFuncName>,
                 RelationVSOpConversion<wafer::BoolUnEqualVS, boolUnEqualVSFuncName>,
                 RelationVSOpConversion<wafer::BoolGreaterEqualVS, boolGreaterEqualVSFuncName>,
                 RelationVSOpConversion<wafer::BoolGreaterVS, boolGreaterVSFuncName>,
                 RelationVSOpConversion<wafer::BoolLessEqualVS, boolLessEqualVSFuncName>,
                 RelationVSOpConversion<wafer::BoolLessThenVS, boolLessThenVSFuncName>,
                 RelationVSOpConversion<wafer::EqualVS,equalVSFuncName>,
                 RelationVSOpConversion<wafer::UnEqualVS,unEqualVSFuncName>,
                 RelationVSOpConversion<wafer::GreaterEqualVS,greaterEqualVSFuncName>,
                 RelationVSOpConversion<wafer::GreaterVS,greaterVSFuncName>,
                 RelationVSOpConversion<wafer::LessEqualVS,lessEqualVSFuncName>,
                 RelationVSOpConversion<wafer::LessThenVS,lessThenVSFuncName>,
                 BinaryLogicVVOpConversion<wafer::AndVV, andVVFuncName>,
                 BinaryLogicVVOpConversion<wafer::OrVV, orVVFuncName>,
                 BinaryLogicVVOpConversion<wafer::XorVV, xorVVFuncName>,
                 UnaryBoolLogicVOpConversion<wafer::BoolNotV, boolNotVFuncName>,
                 BinaryBoolLogicVOpConversion<wafer::BoolAndV, boolAndVFuncName>,
                 BinaryBoolLogicVOpConversion<wafer::BoolXorV, boolXorVFuncName>,
                 BinaryBoolLogicVOpConversion<wafer::BoolOrV, boolOrVFuncName>,
                 Rdma4dOpConversion<wafer::Rdma4dOp, rdma4dFuncName>,
                 Rdma4dOpConversion<wafer::Wdma4dOp, wdma4dFuncName>,
                 Rdma1dOpConversion<wafer::Rdma1dOp, rdma1dFuncName>,
                 Rdma1dOpConversion<wafer::Wdma1dOp, wdma1dFuncName>,
                 RdmaWdmaOpConversion<wafer::RdmaOp,rdmaFuncName>,
                 RdmaWdmaOpConversion<wafer::WdmaOp,wdmaFuncName>,
                 UnaryOpConversion<wafer::MemCopyOp, memcpyFuncName>,
                 TransformOpConversion<wafer::Transpose,transposeFuncName>,
                 TransformOpConversion<wafer::Nchw2nhwc,nchw2nhwcFuncName>,
                 TransformOpConversion<wafer::Nhwc2nchw,nhwc2nchwFuncName>,
                 AtomicBarrierOpConversion<wafer::AtomicBarrierInOp, atomicBarrierInFuncName>,
                 AtomicBarrierOpConversion<wafer::AtomicBarrierOutOp, atomicBarrierOutFuncName>,
                 MaskMoveOpConversion,
                 GatherScatterOpConversion,
                 BitToFPOpConversion,
                 ChannelNormOpConversion,   // NOTE: No op used
                 DechannelNormOpConversion, // NOTE: No op used
                 GemmOpConversion,
                 SigmoidOpConversion,
                 GeluNoneOpConversion,
                 GeluTanhOpConversion,
                 MemsetOpConversion,
                 GetProgramIDConversion,
                 BarrierConversion,
                 RemoteStoreOpConversion,
                 RemoteLoadOpConversion,
                 RandGenOpConversion,
                 AssertConversion>(
        context);
    // clang-format on

    // Add call op conversion
    populateCallOpTypeConversionPattern(patterns, llvmTypeConverter);

    // Add return op conversion
    populateReturnOpTypeConversionPattern(patterns, llvmTypeConverter);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createWaferToLLVMPass() {
  return std::make_unique<WaferToLLVMPass>();
}
