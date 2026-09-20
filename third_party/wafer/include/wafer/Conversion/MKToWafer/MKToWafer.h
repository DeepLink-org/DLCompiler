//===------------------- MKToWafer.h ---------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Lowering magic kernel ops to Wafer Wafer target.
//
//===----------------------------------------------------------------------===//

#ifndef ZTC_CONVERSION_MK_TO_WAFER_H
#define ZTC_CONVERSION_MK_TO_WAFER_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "wafer/Conversion/MKToWafer/Passes.h.inc"

void populateMKToWaferCanonicalizationPatterns(RewritePatternSet &patterns);

void populateMKToWaferConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<ModuleOp>> createMKToWaferPass();

} // namespace triton
} // namespace mlir

#endif // ZTC_CONVERSION_MK_TO_WAFER_H
