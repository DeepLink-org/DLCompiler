//===------------------- WaferMemrefToLLVMPass.cpp--------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "wafer/Conversion/WaferMemrefToLLVM/WaferMemrefToLLVM.h"
#include "wafer/Dialect/IR/WaferDialect.h"
#include "llvm/Support/Debug.h"
#include <memory>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Transforms/Passes.h>

#define DEBUG_TYPE "wafer-memref-to-llvm"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_CLASSES
#include "wafer/Conversion/WaferMemrefToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class WaferMemrefToLLVMPass
    : public mlir::triton::WaferMemrefToLLVMBase<WaferMemrefToLLVMPass> {
  using WaferMemrefToLLVMBase<WaferMemrefToLLVMPass>::WaferMemrefToLLVMBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<LLVM::LLVMDialect, wafer::WaferDialect, arith::ArithDialect,
                func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    target.addIllegalOp<
        memref::AllocOp, memref::LoadOp, memref::StoreOp,
        memref::ReinterpretCastOp, memref::ExtractStridedMetadataOp,
        memref::ExtractAlignedPointerAsIndexOp, memref::CastOp>();

    target.addLegalDialect<LLVM::LLVMDialect, memref::MemRefDialect,
                           func::FuncDialect, arith::ArithDialect,
                           math::MathDialect, arith::ArithDialect,
                           affine::AffineDialect, scf::SCFDialect,
                           cf::ControlFlowDialect, tensor::TensorDialect>();

    target.addLegalOp<ModuleOp>();

    LowerToLLVMOptions options(context);
    options.useBarePtrCallConv = false;
    LLVMTypeConverter llvmTypeConverter(context, options);
    triton::populateWaferMemrefToLLVMConversionPatterns(patterns,
                                                       llvmTypeConverter);
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createWaferMemrefToLLVMPass() {
  return std::make_unique<WaferMemrefToLLVMPass>();
}
