get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(dicp_opt dicp_triton_opt.cpp PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(dicp_opt)
target_link_libraries(dicp_opt PRIVATE
  TritonAnalysis
  TritonTransforms
  TritonGPUTransforms
  TritonNvidiaGPUTransforms
  TritonSharedAnalysis
  ${dialect_libs}
  ${translation_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIRRegisterAllPasses
  DICPNPU
  LinalgToNPU
  LinalgToLinked
  TritonExtTransforms
  LinalgExtTransforms
  LinkedToHIVM
  DICPLinalgExt
  DiscreteMaskAccessConversion
  DICPTransformOps
  MLIRLinalgTransformOps

  BiShengIRHIVMDialect
  LinalgExtAnalysis
  TritonToLinalg
  TritonTilingExtIR
  TritonToLinalgNPUCoversion
  MLIROptLib
  MLIRPass
  MLIRTransforms
)

mlir_check_all_link_libraries(dicp_opt)
