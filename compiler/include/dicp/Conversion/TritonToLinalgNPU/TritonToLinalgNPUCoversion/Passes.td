#ifndef TRITON_TO_LINALG_NPU_CONVERSION_PASSES
#define TRITON_TO_LINALG_NPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToLinalgNPUCoversion : Pass<"triton-to-linalg-npu-conversion", "mlir::ModuleOp"> {
  let summary = "Convert Triton to Linalg dialect";
  let constructor = "::mlir::dicp::linked::createTritonToLinalgNPUCoversionPass()";
  let options = [
      Option<"enableMakeGatherScatterTensorPtr", "enable-make-gather-scatter", "bool", /*default*/"true",
             "Enable make_gather_scatter_tptr support">,
      Option<"enableCollapseShape", "enable-collapse-shape", "bool", /*default*/"false",
             "Enable collapse shape pass">,
  ];
}

#endif
