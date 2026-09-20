//===--------------------- WaferToLLVMPass.cpp -----------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "wafer/Conversion/WaferToLLVM/WaferToLLVM.h"
#include "wafer/Dialect/IR/WaferDialect.h"
#include "llvm/Support/Debug.h"
#include <memory>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Transforms/Passes.h>

#define DEBUG_TYPE "wafer-to-llvm"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "wafer/Conversion/WaferToLLVM/Passes.h.inc"

namespace {

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
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<arith::ArithDialect, memref::MemRefDialect,
                             linalg::LinalgDialect, tensor::TensorDialect,
                             wafer::WaferDialect>();

    // Setup rewrite patterns
    RewritePatternSet patterns(context);

    // NOTE: LLVMTypeConverter should be enough for MLIR core dialects.
    TensorToLLVMTypeConverter converter(context, options);

    triton::populateWaferToLLVMConversionPatterns(patterns, target, converter);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createWaferToLLVMPass() {
  return std::make_unique<WaferToLLVMPass>();
}
