#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::dicp::linked {

std::unique_ptr<OperationPass<ModuleOp>> createLinalgToLinkedPass();

} // namespace mlir::dicp::linked
