#ifndef LINKED_TO_HIVM_CONVERSION_PASSES
#define LINKED_TO_HIVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def LinkedToHIVM : Pass<"linked-to-hivm", "mlir::ModuleOp"> {
  let summary = "Convert Linked to HIVM dialect";
  let constructor = "linked::createLinkedToHIVMPass()";
  let dependentDialects = ["hivm::HIVMDialect"];
}

#endif // LINKED_TO_HIVM_CONVERSION_PASSES
