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
          "bool", /*default*/"true",
          "use linalg named ops instead of linalg.generic">
  ];
}

def BoolTritonPtrPromotion : Pass<"bool-triton-ptr-promotion", "mlir::ModuleOp"> {
  let summary = "Promote !tt.ptr<i1> function args (with tt.divisibility) to !tt.ptr<i8> and propagate";
  let constructor = "linked::createBoolTritonPtrPromotionPass()";
  let description = [{
    Finds function arguments of type !tt.ptr<i1> that have the 'tt.divisibility'
    argument attribute, changes the argument type to !tt.ptr<i8> and updates
    results/uses in the function body (e.g. tt.splat results, tensors carrying
    that pointer type, etc.) to use the i8 pointer type.
  }];
}

#endif
