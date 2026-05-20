

#include "dicp/DynamicCVPipeline/AllocMultiCache.h"

#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "AllocMultiCache";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace triton;

// Run the pass
void AllocMultiCachePass::runOnOperation() {
  ModuleOp module = getOperation();
  OpPassManager pm(module.getOperationName());
  LDBG("Enter pass.");

  // Step 1: Walk scope operations

  // Step 2: Find main loop in each scope

  // Step 3: Apply inner multi-buffer optimization

  if (failed(runPipeline(pm, module))) {
    module->emitError() << "[" << DEBUG_TYPE << "] Pass failed!";
    signalPassFailure();
  }

  LDBG("Process successfully");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAllocMultiCachePass() {
  return std::make_unique<AllocMultiCachePass>();
}

} // namespace triton
} // namespace mlir
