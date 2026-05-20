

#ifndef TRITON_TO_CFG_PASSES_H
#define TRITON_TO_CFG_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {
namespace cfg {

// 创建 BuildCFG pass 的工厂函数
std::unique_ptr<OperationPass<mlir::ModuleOp>> createBuildCFGPass();

// 注册所有 CFG 相关的 passes
#define GEN_PASS_REGISTRATION
#include "dicp/TritonToGraph/Passes.h.inc"

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_CFG_PASSES_H
