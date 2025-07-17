#ifndef LINALG_TO_LINKED_CONVERSION_PASSES
#define LINALG_TO_LINKED_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def LinalgToLinked : Pass<"linalg-to-linked", "mlir::ModuleOp"> {
  let summary = "Convert Linalg to Linked dialect";
  let constructor = "linked::createLinalgToLinkedPass()";
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
