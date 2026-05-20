

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_ADD_CONTROLFLOW_CONDITION_PASS_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_ADD_CONTROLFLOW_CONDITION_PASS_H

#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
// The information that controlFlow sub-passes need to share
struct ControlFlowConditionInfo {
  DenseMap<Value, SmallVector<Value>> crossCoreDependentMap;
  DenseMap<scf::ForOp, DenseMap<Value, SmallVector<Value>>>
      intraCoreDependentMap;
};

class AddControlFlowConditionPass
    : public PassWrapper<AddControlFlowConditionPass, OperationPass<ModuleOp>> {
public:
  AddControlFlowConditionPass() = default;

  void runOnOperation() override;
};

std::unique_ptr<OperationPass<ModuleOp>> createAddControlFlowConditionPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_ADD_CONTROLFLOW_CONDITION_PASS_H
