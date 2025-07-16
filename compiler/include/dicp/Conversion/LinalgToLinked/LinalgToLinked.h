#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::dicp::linked {
inline void populateLinalgToLinkedConversionPatterns(RewritePatternSet &patterns) {}

std::unique_ptr<OperationPass<ModuleOp>> createLinalgToLinkedPass();

} // namespace mlir::dicp::linked
