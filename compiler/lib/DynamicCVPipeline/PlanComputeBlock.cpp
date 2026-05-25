

#include "dicp/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "dicp/DynamicCVPipeline/PlanComputeBlock/OpClassifier.h"
#include "dicp/DynamicCVPipeline/PlanComputeBlock/Passes.h"
#include "dicp/DynamicCVPipeline/PlanComputeBlockPass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace triton;

static constexpr const char *DEBUG_TYPE = "plan-compute-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

// Run the pass
void PlanComputeBlockPass::runOnOperation() {
  ModuleOp module = getOperation();
  OpPassManager pm(module.getOperationName());
  CVPipeline::ComputeBlockIdManager::getInstance().reset();
  LOG_DEBUG("Enter pass.\n");

  // Step 1: Run OpClassifierPass to classify operations
  pm.addPass(createOpClassifierPass());

  // Step 2: Partition compute blocks for core_type=cube

  // Step 3: Partition compute blocks for core_type=vector
  pm.addPass(createPlanVectorBlockPass());

  // Step 4: Reorder

  if (failed(runPipeline(pm, module))) {
    module->emitError() << "[" << DEBUG_TYPE << "] Pass failed!";
    signalPassFailure();
  }

  LOG_DEBUG("Process successfully\n");
}
namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createPlanComputeBlockPass() {
  return std::make_unique<PlanComputeBlockPass>();
}
} // namespace triton
} // namespace mlir
