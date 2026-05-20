

#include "dicp/DynamicCVPipeline/AddControlFlowCondition.h"
#include "dicp/DynamicCVPipeline/AddControlFlowCondition/UpdateConditionInfo.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "AddControlFlowCondition";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace triton;

void AddControlFlowConditionPass::runOnOperation() {
  ModuleOp module = getOperation();
  LDBG("Enter add controlflow condition pass.");
  OpPassManager pm(module.getOperationName());
  ControlFlowConditionInfo info;

  // Step1:Fill in the intraCoreDependentMap and crossCoreDependentMap

  // Step2:Create an ifOp wrapper block based on the block_id

  // Step3:Fill in blockCounters innerDepConds and insertInterCorePipeS

  // Step4:Update the conditions of ifOp based on the intraCoreDependentMap and
  // crossCoreDependentMap
  auto updatePass = std::make_unique<UpdateConditionInfoPass>();
  updatePass->setConditionInfo(&info);
  pm.addPass(std::move(updatePass));
  // Step5:Update the iteration count of forOp
  LDBG("Exit add controlflow condition pass.");
}

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createAddControlFlowConditionPass() {
  return std::make_unique<AddControlFlowConditionPass>();
}
} // namespace triton
} // namespace mlir