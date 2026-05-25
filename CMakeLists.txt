set(DC_TRITON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DC_TRITON_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(DC_TRITON_INCLUDE_DIR "")
set(DC_TRITON_LINK_DIR "")

include_directories(${CMAKE_CURRENT_BINARY_DIR}/compiler/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/compiler/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/include)

# ASCENDNPU_IR编译选项
set(LLVM_MAJOR_VERSION_22_COMPATIBLE ON CACHE BOOL "NPUIR build with LLVM 22" FORCE)
add_definitions(-D__LLVM_MAJOR_VERSION_22_COMPATIBLE__)
add_compile_options(-Wno-switch)

set(ASCENDNPU_IR_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ascendnpu-ir)
set(ASCENDNPU_IR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/third_party/ascendnpu-ir)
set(BISHENGIR_BUILD_STANDALONE_IR_ONLY ON)

add_subdirectory(${ASCENDNPU_IR_SRC_DIR} ${ASCENDNPU_IR_BINARY_DIR})
include_directories(${ASCENDNPU_IR_SRC_DIR}/bishengir/include)
include_directories(${ASCENDNPU_IR_BINARY_DIR}/bishengir/include) # Tablegen'd files

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/triton_linalg/triton)
    file(REMOVE_RECURSE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/triton_linalg/triton")
endif()

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
${dialect_libs}
${conversion_libs}
)

list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/cmake/
)

include(utils)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tools)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/compiler)

if (TRITON_BUILD_PYTHON_MODULE)
    add_triton_plugin(tritonDicpTriton ${CMAKE_CURRENT_SOURCE_DIR}/triton_dicp_triton.cc
    LINK_LIBS

    MLIRAffineToStandard
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRSupport
    MLIRBytecodeWriter
    MLIRFuncInlinerExtension
    MLIRBufferizationDialect
    MLIRMemRefDialect

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
    DynamicCVPipeline
    TritonAffinityOpt
    TritonDicpIR
    TritonStructuredIR
    )
    target_link_libraries(tritonDicpTriton PRIVATE Python3::Module pybind11::headers)
endif()
