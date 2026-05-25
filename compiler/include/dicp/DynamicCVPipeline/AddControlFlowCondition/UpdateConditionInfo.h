

#ifndef TRITON_ADAPTER_UPDATE_CONDITION_INFO_H
#define TRITON_ADAPTER_UPDATE_CONDITION_INFO_H

#include "dicp/DynamicCVPipeline/AddControlFlowCondition.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
class UpdateConditionInfoPass
    : public PassWrapper<UpdateConditionInfoPass, OperationPass<ModuleOp>> {
public:
  UpdateConditionInfoPass() = default;

  void runOnOperation() override;

  void setConditionInfo(ControlFlowConditionInfo *info) { this->info = info; }

private:
  ControlFlowConditionInfo *info = nullptr;
};

std::unique_ptr<OperationPass<ModuleOp>> createUpdateConditionInfoPass();
} // namespace triton
} // namespace mlir
#endif // TRITON_ADAPTER_UPDATE_CONDITION_INFO_H
