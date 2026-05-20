

#ifndef TRITON_ADAPTER_TRITON_TO_HFUSION_CONVERSION_PASSES_H
#define TRITON_ADAPTER_TRITON_TO_HFUSION_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
// Forward declarations.
class ModuleOp;

namespace triton {

/// Creates a pass to convert Triton dialect to HFusion dialect.
std::unique_ptr<OperationPass<ModuleOp>> createTritonToHFusionPass();

#define GEN_PASS_REGISTRATION
#include "dicp/TritonToHFusion/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_TRITON_TO_HFUSION_CONVERSION_PASSES_H
