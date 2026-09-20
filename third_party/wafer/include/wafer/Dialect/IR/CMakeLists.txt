set(LLVM_TARGET_DEFINITIONS WaferOps.td)
mlir_tablegen(WaferDialect.h.inc -gen-dialect-decls -dialect=wafer)
mlir_tablegen(WaferDialect.cpp.inc -gen-dialect-defs -dialect=wafer)
mlir_tablegen(WaferOps.h.inc -gen-op-decls)
mlir_tablegen(WaferOps.cpp.inc -gen-op-defs)

mlir_tablegen(WaferEnums.h.inc -gen-enum-decls)
mlir_tablegen(WaferEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS WaferTypes.td)
mlir_tablegen(WaferTypes.h.inc -gen-typedef-decls)
mlir_tablegen(WaferTypes.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(WaferTableGen)
