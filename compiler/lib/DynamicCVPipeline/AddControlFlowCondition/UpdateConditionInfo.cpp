
#include "dicp/DynamicCVPipeline/AddControlFlowCondition/UpdateConditionInfo.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "dicp/DynamicCVPipeline/AddControlFlowCondition.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "UpdateConditionInfoPass";
static constexpr const char *SSBUFFER_Main_LOOP = "ssbuffer.main_loop";
static constexpr const char *SSBUFFER_IF = "ssbuffer.if";

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")
using namespace mlir;
using namespace triton;
using namespace hivm;

void UpdateConditionInfoPass::runOnOperation() {
  ModuleOp module = getOperation();

  LDBG("Enter UpdateConditionInfo pass.");
  // Update the conditions of ifOp based on the intraCoreDependentMap and
  // crossCoreDependentMap
  LDBG("Exit UpdateConditionInfo pass.");
}

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createUpdateConditionInfoPass() {
  return std::make_unique<UpdateConditionInfoPass>();
}
} // namespace triton
} // namespace mlir
