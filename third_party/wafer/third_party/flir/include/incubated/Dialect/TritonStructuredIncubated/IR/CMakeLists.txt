set(LLVM_TARGET_DEFINITIONS TritonStructuredDialectIncubated.td)
mlir_tablegen(TritonStructuredDialectIncubated.h.inc -gen-dialect-decls -dialect=tts)
mlir_tablegen(TritonStructuredDialectIncubated.cpp.inc -gen-dialect-defs -dialect=tts)
mlir_tablegen(TritonStructuredOpsIncubated.h.inc -gen-op-decls)
mlir_tablegen(TritonStructuredOpsIncubated.cpp.inc -gen-op-defs)


add_public_tablegen_target(TritonStructuredIncubatedTableGen)
