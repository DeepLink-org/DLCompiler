//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace deeplink {
namespace npu {
void populateLinalgToNPUConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<ModuleOp>> createLinalgToNPUPass();

} // namespace npu
} // namespace deeplink
} // namespace mlir
