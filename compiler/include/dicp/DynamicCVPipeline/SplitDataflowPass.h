

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_SPLIT_DATAFLOW_PASS_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_SPLIT_DATAFLOW_PASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

class SplitDataflowPass
    : public PassWrapper<SplitDataflowPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitDataflowPass)

  // Constructor
  SplitDataflowPass() = default;

  // Run the pass
  void runOnOperation() override;
};

// Create the pass
std::unique_ptr<OperationPass<ModuleOp>> createSplitDataflowPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_SPLIT_DATAFLOW_PASS_H