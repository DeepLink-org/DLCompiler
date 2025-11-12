#ifndef TRITONEXT_PASSES
#define TRITONEXT_PASSES

include "mlir/Pass/PassBase.td"

def BoolTritonPtrPromotion : Pass<"bool-triton-ptr-promotion", "mlir::ModuleOp"> {
  let summary = "Promote !tt.ptr<i1> function args (with tt.divisibility) to !tt.ptr<i8> and propagate";
  let description = [{
    Finds function arguments of type !tt.ptr<i1> that have the 'tt.divisibility'
    argument attribute, changes the argument type to !tt.ptr<i8> and updates
    results/uses in the function body (e.g. tt.splat results, tensors carrying
    that pointer type, etc.) to use the i8 pointer type.
  }];
  let constructor = "mlir::dicp::trtion_ext::createBoolTritonPtrPromotionPass()";
  let dependentDialects = ["mlir::triton::TritonDialect"];
}


def CanonicalizeCmpi : Pass<"canonicalize-cmpi", "mlir::ModuleOp"> {
  let summary = "Canonicalize selected arith.cmpi predicates to stricter forms (with RHS+1).";
  let description = [{
    Canonicalize comparisons to equivalent forms using RHS+1:
      - signed less-or-equal (sle)  -> signed less-than (slt) with RHS + 1
      - signed greater-than (sgt)  -> signed greater-or-equal (sge) with RHS - 1
      - unsigned less-or-equal (ule)-> unsigned less-than (ult) with RHS + 1

    The pass only transforms when the RHS is an integer scalar or a ranked tensor of integer elements.
  }];
  let constructor = "mlir::dicp::trtion_ext::createCanonicalizeCmpiPass()";
  let dependentDialects = ["mlir::triton::TritonDialect","mlir::arith::ArithDialect"];
}

#endif
