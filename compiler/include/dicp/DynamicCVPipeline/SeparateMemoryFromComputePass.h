

#ifndef TRITON_ADAPTER_SEPARATE_MEMORY_FROM_COMPUTE_PASS_H
#define TRITON_ADAPTER_SEPARATE_MEMORY_FROM_COMPUTE_PASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

/// Sub-pipeline pass that separates memory from compute operations.
class SeparateMemoryFromComputePass
    : public PassWrapper<SeparateMemoryFromComputePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SeparateMemoryFromComputePass)

  SeparateMemoryFromComputePass() = default;

  void runOnOperation() override;
};

std::unique_ptr<OperationPass<ModuleOp>> createSeparateMemoryFromComputePass();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_SEPARATE_MEMORY_FROM_COMPUTE_PASS_H
