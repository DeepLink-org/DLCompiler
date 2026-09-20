//===------------------- Passes.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef MK_TO_WAFER_CONVERSION_PASSES_H
#define MK_TO_WAFER_CONVERSION_PASSES_H

#include "wafer/Conversion/MKToWafer/MKToWafer.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "wafer/Conversion/MKToWafer/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif //  MK_TO_WAFER_CONVERSION_PASSES_H
