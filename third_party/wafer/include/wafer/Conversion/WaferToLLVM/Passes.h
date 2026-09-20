//===------------------- Passes.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef WAFER_TO_LLVM_CONVERSION_PASSES_H
#define WAFER_TO_LLVM_CONVERSION_PASSES_H

#include "wafer/Conversion/WaferToLLVM/WaferToLLVM.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "wafer/Conversion/WaferToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // WAFER_TO_LLVM_CONVERSION_PASSES_H
