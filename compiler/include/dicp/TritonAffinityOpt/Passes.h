

#ifndef TRITON_ADAPTER_TRITON_AFFINITY_OPTIMIZATION_PASSES_H
#define TRITON_ADAPTER_TRITON_AFFINITY_OPTIMIZATION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
// Forward declarations.
class ModuleOp;

namespace triton {

/// Creates a pass to convert Triton dialect to Annotation dialect.
std::unique_ptr<OperationPass<ModuleOp>> createDAGSSBufferPass();

std::unique_ptr<OperationPass<ModuleOp>> createDAGSyncPass();

std::unique_ptr<OperationPass<ModuleOp>> createDAGScopePass();

#define GEN_PASS_REGISTRATION
#include "dicp/TritonAffinityOpt/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_TRITON_AFFINITY_OPTIMIZATION_PASSES_H