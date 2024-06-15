set(DC_TRITON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DC_TRITON_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(DC_TRITON_INCLUDE_DIR "")
set(DC_TRITON_LINK_DIR "")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/triton_linalg/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/triton_linalg)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

set(LIBS
${dialect_libs}
${conversion_libs}
)


add_triton_plugin(DICPTriton ${CMAKE_CURRENT_SOURCE_DIR}/dicp_triton.cc LINK_LIBS ${LIBS})
target_include_directories(DICPTriton PUBLIC "${pybind11_INCLUDE_DIRS}")
