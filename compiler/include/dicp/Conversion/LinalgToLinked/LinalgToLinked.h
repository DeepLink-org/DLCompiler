#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::dicp::linked {

std::unique_ptr<OperationPass<ModuleOp>> createLinalgToLinkedPass();

std::unique_ptr<OperationPass<ModuleOp>> createBoolTritonPtrPromotionPass();

} // namespace mlir::dicp::linked
