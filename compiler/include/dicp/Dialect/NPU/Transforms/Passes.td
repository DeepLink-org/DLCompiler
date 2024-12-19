#ifndef DICP_MLIR_NPU_DIALECT_PASSES
#define DICP_MLIR_NPU_DIALECT_PASSES

include "mlir/Pass/PassBase.td"

def Codegen : Pass<"codegen", "mlir::ModuleOp"> {
  let summary = "NPU dialect codegen";
  let constructor = "npu::createCodegenPass()";
}

#endif
