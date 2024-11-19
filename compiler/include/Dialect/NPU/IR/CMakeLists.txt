# add_dicp_compiler_dialect(DICPLinalgExt NPU)

set(LLVM_TARGET_DEFINITIONS NPUOps.td)
mlir_tablegen(NPUDialect.h.inc -gen-dialect-decls -dialect=NPU)
mlir_tablegen(NPUDialect.cpp.inc -gen-dialect-defs -dialect=NPU)
mlir_tablegen(NPUOps.h.inc -gen-op-decls)
mlir_tablegen(NPUOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(NPUTableGen)
