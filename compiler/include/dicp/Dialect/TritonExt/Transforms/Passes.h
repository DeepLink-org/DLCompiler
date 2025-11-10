#ifndef TRITONEXT_DIALECT_TRITON_TRANSFORMS_PASSES_H_
#define TRITONEXT_DIALECT_TRITON_TRANSFORMS_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::dicp::trtion_ext {

std::unique_ptr<OperationPass<ModuleOp>> createBoolTritonPtrPromotionPass();

std::unique_ptr<OperationPass<ModuleOp>> createCanonicalizeCmpiPass();

#define GEN_PASS_REGISTRATION
#include "dicp/Dialect/TritonExt/Transforms/Passes.h.inc"

} // namespace mlir::dicp::trtion_ext

#endif
