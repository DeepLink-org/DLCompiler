get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

# TODO: Workaround include path
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/wafer/include)
# include_directories(${CMAKE_CURRENT_BINARY_DIR}/../third_party/wafer/include)

add_llvm_executable(wafer-opt wafer-opt.cpp PARTIAL_SOURCES_INTENDED)

# TODO: what's this?
llvm_update_compile_flags(wafer-opt)
target_link_libraries(wafer-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  ${triton_libs}
  TritonSharedAnalysis
  TleDsaIR
  TleDsaToCore
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(wafer-opt)

# add_llvm_executable(wafer-reduce wafer-reduce.cpp PARTIAL_SOURCES_INTENDED)
# mlir_check_all_link_libraries(wafer-reduce)

# llvm_update_compile_flags(wafer-reduce)
# target_link_libraries(wafer-reduce PRIVATE
#   ${dialect_libs}
#   ${conversion_libs}
#   ${extension_libs}
#   ${triton_libs}
#   # tests
#   TritonTestAnalysis
#   TritonTestDialectTritonGPU
#   TritonAMDGPUTestAnalysis
#   # MLIR core
#   MLIRReduceLib
#   MLIRPass
#   MLIRTransforms
#   MLIRMathTestPasses
# )

# mlir_check_all_link_libraries(wafer-reduce)

# add_llvm_executable(wafer-lsp wafer-lsp.cpp PARTIAL_SOURCES_INTENDED)

# llvm_update_compile_flags(wafer-lsp)
# target_link_libraries(wafer-lsp PRIVATE
#   ${dialect_libs}
#   ${conversion_libs}
#   ${extension_libs}
#   ${triton_libs}
#   # tests
#   TritonTestAnalysis
#   TritonTestDialectTritonGPU
#   TritonAMDGPUTestAnalysis
#   # MLIR core
#   MLIRLspServerLib
#   MLIRPass
#   MLIRTransforms
#   MLIRMathTestPasses
# )

# mlir_check_all_link_libraries(wafer-lsp)


# add_llvm_executable(wafer-llvm-opt
#   wafer-llvm-opt.cpp

#   PARTIAL_SOURCES_INTENDED
#   DEPENDS
#   intrinsics_gen
#   SUPPORT_PLUGINS
#   )
# target_link_libraries(wafer-llvm-opt PRIVATE
#   TritonLLVMIR

#   LLVMAnalysis
#   LLVMCore
#   LLVMSupport
#   LLVMOption
#   LLVMCodeGen
#   )
# export_executable_symbols_for_plugins(wafer-llvm-opt)


# add_llvm_executable(wafer-tensor-layout wafer-tensor-layout.cpp PARTIAL_SOURCES_INTENDED)
# target_link_libraries(wafer-tensor-layout PRIVATE
#   ${triton_libs}
#   ${conversion_libs}
#   ${extension_libs}
#   ${dialect_libs}
#   TritonTestAnalysis
#   TritonTestDialectTritonGPU
#   TritonAMDGPUTestAnalysis
#   MLIRMathTestPasses
#   )

install(TARGETS wafer-opt
  RUNTIME DESTINATION ${INSTALL_WAFER_DIR}/bin
)
