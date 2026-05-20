

#ifndef TRITON_ADAPTER_ADD_DYNAMIC_CVPIPELINE_PASSES_H
#define TRITON_ADAPTER_ADD_DYNAMIC_CVPIPELINE_PASSES_H

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define GEN_PASS_DECL_ADDDYNAMICCVPIPELINE
#include "dicp/DynamicCVPipeline/Passes.h.inc"

#define GEN_PASS_DEF_ADDDYNAMICCVPIPELINE
#include "dicp/DynamicCVPipeline/Passes.h.inc"

extern bool compileOn91095Flag;

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>>
createAddDynamicCVPipelinePass(const AddDynamicCVPipelineOptions &options = {});
} // namespace triton
} // namespace mlir

namespace {
using namespace mlir;
using namespace triton;

class AddDynamicCVPipelinePass
    : public ::impl::AddDynamicCVPipelineBase<AddDynamicCVPipelinePass> {
public:
  explicit AddDynamicCVPipelinePass(const AddDynamicCVPipelineOptions &options);
  void runOnOperation() override;
};

} // namespace

#endif // TRITON_ADAPTER_ADD_DYNAMIC_CVPIPELINE_PASSES_H