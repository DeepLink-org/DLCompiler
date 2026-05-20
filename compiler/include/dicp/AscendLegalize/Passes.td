#ifndef ASCENDLEGALIZE_PASSES
#define ASCENDLEGALIZE_PASSES

include "mlir/Pass/PassBase.td"

def AscendLegalize : Pass<"ascend-legalize", "mlir::ModuleOp"> {
    let summary = "Legalize IR for Ascend backend (normalize cmpi predicates etc.)";
    let constructor = "triton::createAscendLegalizePass()";
    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::triton::TritonDialect"
    ];
}

#endif // ASCENDLEGALIZE_PASSES
