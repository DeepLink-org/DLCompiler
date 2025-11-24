#ifndef TRITONEXT_DIALECT_TRITON_TRANSFORMS_PASSES_H_
#define TRITONEXT_DIALECT_TRITON_TRANSFORMS_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::dicp::trtion_ext {
const unsigned INT_BIT_WIDTH = 32;
const unsigned SET_INIT_SIZE = 16;

enum TensorKind { NONE = -1, INPUT = 0, OUTPUT = 1, INPUT_OUTPUT = 2 };

std::unique_ptr<OperationPass<ModuleOp>> createCanonicalizeTritonIRAscendPass();

std::unique_ptr<OperationPass<ModuleOp>> createCanonicalizeCmpiPass();

#define GEN_PASS_REGISTRATION
#include "dicp/Dialect/TritonExt/Transforms/Passes.h.inc"

} // namespace mlir::dicp::trtion_ext

#endif
