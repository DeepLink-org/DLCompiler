

#include "dicp/DynamicCVPipeline/SeparateMemoryFromComputePass.h"
#include "dicp/DynamicCVPipeline/SeparateMemoryFromCompute/AddMultiBufferToGMLoadPass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "SeparateMemoryFromCompute";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace triton;

static constexpr int kDefaultBufferDepth = 1;

void SeparateMemoryFromComputePass::runOnOperation() {
  ModuleOp module = getOperation();

  int depth = kDefaultBufferDepth;

  if (depth <= 1) {
    LDBG("Buffer depth <= 1, skip multi-buffer transformation");
    return;
  }

  OpPassManager pm(module.getOperationName());
  LDBG("Enter SeparateMemoryFromCompute pass");

  // Step 1: Hoist memory operations out of compute blocks

  // Step 2: Apply multi-buffering to memory operations
  pm.addPass(createAddMultiBufferToGMLoadPass());

  if (failed(runPipeline(pm, module))) {
    module->emitError() << "[" << DEBUG_TYPE << "] Pass failed!";
    signalPassFailure();
  }

  LDBG("Process successfully");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createSeparateMemoryFromComputePass() {
  return std::make_unique<SeparateMemoryFromComputePass>();
}

} // namespace triton
} // namespace mlir