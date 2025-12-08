set(DC_TRITON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DC_TRITON_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(TRTION_SHARED_NPU_SPECIFIC_SOURCES  "${DC_TRITON_SOURCE_DIR}/compiler/lib/Conversion/TritonToLinalgNPU")

set(DC_TRITON_INCLUDE_DIR "")
set(DC_TRITON_LINK_DIR "")

include_directories(${CMAKE_CURRENT_BINARY_DIR}/compiler/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/compiler/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party/triton_shared/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/triton_shared/include)

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
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/triton_shared)

if (TRITON_BUILD_PYTHON_MODULE)
    add_triton_plugin(tritonDicpTriton ${CMAKE_CURRENT_SOURCE_DIR}/triton_dicp_triton.cc 
    LINK_LIBS 

    MLIRAffineToStandard
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRSupport
    MLIRBytecodeWriter
    
    TritonToLinalgNPUCoversion

    LinalgExtTransforms
    TritonExtTransforms

    LinalgToLinked
    LinkedToHIVM
    )
    target_link_libraries(tritonDicpTriton PRIVATE Python3::Module pybind11::headers)
endif()