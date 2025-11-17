#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::dicp::linked {

std::unique_ptr<OperationPass<ModuleOp>> createLinkedToHIVMPass();

#define GEN_PASS_REGISTRATION
#include "dicp/Conversion/LinkedToHIVM/Passes.h.inc"

} // namespace mlir::dicp::linked
