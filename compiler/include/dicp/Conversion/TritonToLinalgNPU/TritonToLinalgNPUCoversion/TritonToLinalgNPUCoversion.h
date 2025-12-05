#ifndef TRITON_CONVERSION_TRITONTOLINALG_NPU_TRITONTOLINALGH
#define TRITON_CONVERSION_TRITONTOLINALG_NPU_TRITONTOLINALGH

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::dicp::linked {

std::unique_ptr<OperationPass<ModuleOp>> createTritonToLinalgNPUCoversionPass();

} // namespace mlir::dicp::linked

#endif // TRITON_CONVERSION_TRITONTOLINALG_NPU_TRITONTOLINALGH
