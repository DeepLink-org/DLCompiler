#ifndef MEMREF_COPY_GATHER_TO_TENSOR_INSERT_PASSES_H
#define MEMREF_COPY_GATHER_TO_TENSOR_INSERT_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::dicp::linked {

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createMemRefCopyGatherToTensorInsertPass();

#define GEN_PASS_REGISTRATION
#include "dicp/Conversion/TritonToLinalgNPU/MemRefCopyGatherToTensorInsert/Passes.h.inc"

} // namespace mlir::dicp::linked

#endif
