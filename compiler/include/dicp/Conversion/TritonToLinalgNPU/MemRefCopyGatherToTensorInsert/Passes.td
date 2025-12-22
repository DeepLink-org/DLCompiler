#ifndef MEMREF_COPY_GATHER_TO_TENSOR_INSERT_PASSES
#define MEMREF_COPY_GATHER_TO_TENSOR_INSERT_PASSES

include "mlir/Pass/PassBase.td"

def MemRefCopyGatherToTensorInsert : Pass<"discrete-gather-to-direct-insert", "mlir::ModuleOp"> {
  let summary = "Converts alloc+copy based gather loops to direct tensor insertions";
  let description = [{
    Identifies patterns where a temporary memref is allocated, populated via 
    index-based gathers (subview + copy) in a loop, and then converted to a tensor.
    Replaces this with a tensor.empty + scf.for (iter_args) + tensor.insert 
    sequence to avoid stack allocation and enable register-level operation.
  }];
  let constructor = "::mlir::dicp::linked::createMemRefCopyGatherToTensorInsertPass()";
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect",
    "mlir::bufferization::BufferizationDialect"
  ];
}

#endif
