#ifndef LINALG_TO_LINKED_CONVERSION_PASSES
#define LINALG_TO_LINKED_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def LinalgToLinked : Pass<"linalg-to-linked", "mlir::ModuleOp"> {
  let summary = "Convert Linalg to Linked dialect";
  let constructor = "linked::createLinalgToLinkedPass()";
  let options = [
      Option<"globalKernel", "global-kernel",
          "bool", /*default*/"true",
          "Generate a global kernel">,
      Option<"namedOps", "named-ops",
          "bool", /*default*/"true",
          "Use linalg named ops instead of linalg.generic">,
      Option<"cpuVerify", "cpu-verify",
          "bool", /*default*/"false",
          "Skip NPU workspace args for CPU verification">
  ];
}

def DebugCPUVerify : Pass<"debug-cpu-verify", "mlir::ModuleOp"> {
  let summary = "Verify that only MLIR built-in dialects remain for CPU runner";
  let description = [{
    Verification pass that scans the module for operations belonging to
    non-MLIR-upstream (external) dialects. Any such operation is reported
    as an error, ensuring the IR has been fully lowered to standard MLIR
    constructs before being handed off to a CPU runner for correctness
    validation.
  }];
  let constructor = "linked::createDebugCPUVerifyPass()";
}

#endif
