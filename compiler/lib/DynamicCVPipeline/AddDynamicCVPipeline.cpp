

#include "dicp/DynamicCVPipeline/AddControlFlowCondition.h"
#include "dicp/DynamicCVPipeline/AllocMultiCache.h"
#include "dicp/DynamicCVPipeline/Passes.h"
#include "dicp/DynamicCVPipeline/PlanComputeBlockPass.h"
#include "dicp/DynamicCVPipeline/SeparateMemoryFromComputePass.h"
#include "dicp/DynamicCVPipeline/SplitDataflowPass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "AddDynamicCVPipeline";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ADDDYNAMICCVPIPELINE
#include "dicp/DynamicCVPipeline/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

AddDynamicCVPipelinePass::AddDynamicCVPipelinePass(
    const AddDynamicCVPipelineOptions &options)
    : AddDynamicCVPipelineBase(options) {}

void AddDynamicCVPipelinePass::runOnOperation() {
  auto moduleOp = getOperation();
  compileOn91095Flag = this->compileOn91095;

  LDBG("Enter pass");

  if (!compileOn91095Flag) {
    llvm::errs() << "Add-dynamic-cv-pipeline is only supported on 91095 now.\n";
    return;
  }

  PassManager pm(&getContext(), moduleOp.getOperationName());

  // todo: add related passes.
  pm.addPass(createPlanComputeBlockPass());
  pm.addPass(createSplitDataflowPass());
  pm.addPass(createSeparateMemoryFromComputePass());
  pm.addPass(createAllocMultiCachePass());
  pm.addPass(createAddControlFlowConditionPass());

  if (failed(runPipeline(pm, getOperation()))) {
    moduleOp->emitError() << "[" << DEBUG_TYPE << "] Pass failed!";
    signalPassFailure();
  }

  LDBG("Process successfully");
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createAddDynamicCVPipelinePass(
    const AddDynamicCVPipelineOptions &options) {
  return std::make_unique<AddDynamicCVPipelinePass>(options);
}