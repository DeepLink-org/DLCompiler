#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::dicp::linked {

std::unique_ptr<OperationPass<ModuleOp>>
createLinalgToLinkedPass(bool globalKernel = true, bool namedOps = true);

} // namespace mlir::dicp::linked
