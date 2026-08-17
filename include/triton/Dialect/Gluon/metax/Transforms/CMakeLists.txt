set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name GluonMetaX)
add_public_tablegen_target(GluonMetaXTransformsIncGen)
