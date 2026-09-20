set(LLVM_TARGET_DEFINITIONS MathExtBase.td)
mlir_tablegen(MathExtDialect.h.inc -gen-dialect-decls)
mlir_tablegen(MathExtDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRMathExtDialectIncGen)

set(LLVM_TARGET_DEFINITIONS MathExtOps.td)
mlir_tablegen(MathExtOps.h.inc  -gen-op-decls)
mlir_tablegen(MathExtOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(MLIRMathExtOpsIncGen)
