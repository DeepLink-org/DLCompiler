
#pragma once

namespace mlir::dicp::npu {

std::unique_ptr<OperationPass<ModuleOp>> createCodegenPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "dicp/Dialect/NPU/Transforms/Passes.h.inc"

} // namespace mlir::dicp::npu
