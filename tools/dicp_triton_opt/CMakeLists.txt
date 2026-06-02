get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(dicp_opt dicp_triton_opt.cpp PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(dicp_opt)
target_link_libraries(dicp_opt PRIVATE
  TritonAnalysis
  TritonTransforms
  ${dialect_libs}
  ${translation_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIRRegisterAllPasses

  # DICP NPU pass libraries
  TritonToLinalg
  TritonToStructured
  TritonToUnstructure
  TritonToHIVM
  TritonToAnnotation
  TritonToHFusion
  TritonToLLVM
  TritonToGraph
  AutoBlockify
  AscendLegalize
  CommonIRTransforms
  DiscreteMaskAccessConversion
  MLIRTritonNPUUtils
  TritonAffinityOpt
  TritonDicpIR
  TritonStructuredIR
  TritonNvidiaGPUTransforms

  MLIROptLib
  MLIRPass
  MLIRTransforms
)

mlir_check_all_link_libraries(dicp_opt)
