#ifndef TRITON_COMMONIR_PASSES_H_
#define TRITON_COMMONIR_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::func {
class FuncOp;
}

namespace mlir::dicp::CommonIR {

std::unique_ptr<OperationPass<func::FuncOp>> createVectorizeParallelLoopPass();

std::unique_ptr<OperationPass<mlir::ModuleOp>> createAnnotateKernelAttrsPass();

#define GEN_PASS_REGISTRATION
#include "dicp/Dialect/CommonIR/Passes.h.inc"

} // namespace mlir::dicp::CommonIR

#endif // TRITON_COMMONIR_PASSES_H_
