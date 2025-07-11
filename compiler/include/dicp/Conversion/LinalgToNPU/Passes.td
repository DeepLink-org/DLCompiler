#ifndef LINALG_TO_NPU_CONVERSION_PASSES
#define LINALG_TO_NPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def LinalgToNPU : Pass<"linalg-to-npu", "mlir::ModuleOp"> {
  let summary = "Convert Linalg to NPU dialect";
  let constructor = "npu::createLinalgToNPUPass()";
  let options = [
      Option<"globalKernel", "global-kernel",
          "bool", /*default*/"true",
          "generate a global kernel">,
      Option<"namedOps", "named-ops",
          "bool", /*default*/"false",
          "use linalg named ops instead of linalg.generic">
  ];
}

#endif
