

#include "dicp/DynamicCVPipeline/SplitDataflowPass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "SplitDataflow";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace triton;

// Run the pass
void SplitDataflowPass::runOnOperation() {
  ModuleOp module = getOperation();
  OpPassManager pm(module.getOperationName());
  LDBG("Enter pass.");

  // Step 1: Run InterCoreTransferAndSync

  // Step 2: Run SeparateCVScope

  if (failed(runPipeline(pm, module))) {
    module->emitError() << "[" << DEBUG_TYPE << "] Pass failed!";
    signalPassFailure();
  }

  LDBG("Process successfully");
}
namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createSplitDataflowPass() {
  return std::make_unique<SplitDataflowPass>();
}
} // namespace triton
} // namespace mlir
