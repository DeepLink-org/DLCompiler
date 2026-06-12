#ifndef TRITON_COMMONIR_PASSES
#define TRITON_COMMONIR_PASSES

include "mlir/Pass/PassBase.td"

def VectorizeParallelLoop : Pass<"vectorize-parallel-loop", "func::FuncOp"> {
  let summary = "Convert single-element parallel loops to vectorized batch processing.";
  let constructor = "mlir::dicp::CommonIR::createVectorizeParallelLoopPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect", "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect", "mlir::scf::SCFDialect",
    "mlir::bufferization::BufferizationDialect", "mlir::func::FuncDialect"
  ];
}

def AnnotateKernelAttrs : Pass<"annotate-kernel-attrs", "mlir::ModuleOp"> {
  let summary = "Annotate kernel func.func with mix_mode/parallel_mode/global_kernel/SyncBlockLockArgIdx/WorkspaceArgIdx.";
  let description = [{
    For CommonIR entry kernels that arrive as plain `func.func` (without a
    `tt.func` upstream), the downstream TritonToLinalg lowering never gets a
    chance to attach the kernel-level attributes that the Ascend host wrapper
    parses (`mix_mode`, `parallel_mode`, `global_kernel`,
    `SyncBlockLockArgIdx`, `WorkspaceArgIdx`). This pass attaches those
    attributes and prepends the two stub arguments (syncBlockLock + workspace)
    expected by the NPU runtime.
  }];
  let constructor = "mlir::dicp::CommonIR::createAnnotateKernelAttrsPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect", "mlir::memref::MemRefDialect"
  ];
}

#endif // TRITON_COMMONIR_PASSES
