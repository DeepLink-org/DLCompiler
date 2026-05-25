

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMPUTE_BLOCK_PASS_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMPUTE_BLOCK_PASS_H

#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

// PlanComputeBlockPass for partitioning operations into compute blocks
class PlanComputeBlockPass
    : public PassWrapper<PlanComputeBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanComputeBlockPass)

  // Constructor
  PlanComputeBlockPass() = default;

  // Run the pass
  void runOnOperation() override;
};

// Create the pass
std::unique_ptr<OperationPass<ModuleOp>> createPlanComputeBlockPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMPUTE_BLOCK_PASS_H
