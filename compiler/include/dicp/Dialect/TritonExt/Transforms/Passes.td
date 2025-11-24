#ifndef TRITONEXT_PASSES
#define TRITONEXT_PASSES

include "mlir/Pass/PassBase.td"

def CanonicalizeTritonIRAscend : Pass<"canonicalize-triton-ir-ascend", "mlir::ModuleOp"> {
  let summary = "Canonicalize Triton IR for NPU backend";

  let description = [{
    Perform canonicalization and Triton-specific pattern rewrites to prepare
    Triton IR for lowering to the NPU backend. This pass performs actions such
    as: lowering atomic RMW/CAS to linalg.generic forms, handling masked
    operations, moving/merging bitcasts, canonicalizing scalar/tensor store
    patterns, rewrites arith.remf, and other device-specific cleanups required by the backend.
  }];

  let constructor = "mlir::dicp::trtion_ext::createCanonicalizeTritonIRAscendPass()";

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect",
  ];
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
