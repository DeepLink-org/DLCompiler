//===------------------- Passes.td ----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef WAFER_TO_LLVM_CONVERSION_PASSES
#define WAFER_TO_LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def WaferToLLVM : Pass<"wafer-to-llvm", "ModuleOp"> {
  let summary = "Convert Wafer dialect to LLVM dialect";
  let description = [{
    This pass converts operations in the Wafer dialect to the LLVM IR dialect.

    It handles the conversion of Wafer-specific operations like wafer.rdma, wafer.wdma,
    wafer.gemm etc to appropriate LLVM calls to the Wafer runtime library.

    The pass also relies on existing conversion patterns for standard dialects
    like arith, func, memref, etc.
  }];

  let constructor = "triton::createWaferToLLVMPass()";

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "wafer::WaferDialect"
  ];
}

#endif
