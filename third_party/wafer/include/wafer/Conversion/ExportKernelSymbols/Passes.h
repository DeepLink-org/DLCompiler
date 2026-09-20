//===------------------- Passes.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Ludt) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef EXPORT_KERNEL_SYMBOLS_CONVERSION_PASSES_H
#define EXPORT_KERNEL_SYMBOLS_CONVERSION_PASSES_H

#include "wafer/Conversion/ExportKernelSymbols/ExportKernelSymbols.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "wafer/Conversion/ExportKernelSymbols/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif //  EXPORT_KERNEL_SYMBOLS_CONVERSION_PASSES_H
