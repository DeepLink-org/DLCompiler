get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(dicp_opt dicp_triton_opt.cpp PARTIAL_SOURCES_INTENDED)

# TODO: what's this?
llvm_update_compile_flags(dicp_opt)
target_link_libraries(dicp_opt PRIVATE
  TritonAnalysis
  TritonTransforms
  TritonGPUTransforms
  ${dialect_libs}
  ${conversion_libs}

  MLIROptLib
  MLIRPass
  MLIRTransforms
)

mlir_check_all_link_libraries(dicp_opt)