//===------------------- Passes.td ----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef MK_TO_WAFER_CONVERSION_PASSES
#define MK_TO_WAFER_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def MKToWafer : Pass<"mk-to-wafer", "mlir::ModuleOp"> {
  let summary = "Convert magic kernel operations into Wafer Wafer operations";
  let constructor = "triton::createMKToWaferPass()";
}

#endif
