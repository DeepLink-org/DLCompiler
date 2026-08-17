set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS GluonTypes.td)
mlir_tablegen(GluonTypes.h.inc -gen-typedef-decls)
mlir_tablegen(GluonTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(GluonTypesEnums.h.inc -gen-enum-decls)
mlir_tablegen(GluonTypesEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(GluonMetaXTableGen)
