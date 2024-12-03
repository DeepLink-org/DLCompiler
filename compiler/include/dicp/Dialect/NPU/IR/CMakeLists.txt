set(LLVM_TARGET_DEFINITIONS NPUOps.td)
mlir_tablegen(NPUDialect.h.inc -gen-dialect-decls -dialect=npu)
mlir_tablegen(NPUDialect.cpp.inc -gen-dialect-defs -dialect=npu)
mlir_tablegen(NPUOps.h.inc -gen-op-decls)
mlir_tablegen(NPUOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(NPUTableGen)


set(LLVM_TARGET_DEFINITIONS NPUTypes.td)
mlir_tablegen(NPUTypes.h.inc -gen-typedef-decls)
mlir_tablegen(NPUTypes.cpp.inc -gen-typedef-defs)