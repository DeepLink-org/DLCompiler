option(TRITON_SHARED_BUILD_CPU_BACKEND "Build triton-shared CPU backend" ON)
option(FLIR_BUILD_INCUBATED "Build FLIR incubated dialects/conversions" ${FLIR_BUILD_INCUBATED})
set(TRITON_SHARED_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TRITON_SHARED_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIR})

if(FLAGTREE_BACKEND STREQUAL "wafer")
  add_compile_definitions(FLAGTREE_BACKEND_WAFER)
endif()

add_subdirectory(include)
add_subdirectory(lib)

if(NOT FLAGTREE_BACKEND STREQUAL "wafer")
  add_subdirectory(test)
  add_subdirectory(tools)
endif()

if (TRITON_SHARED_BUILD_CPU_BACKEND)
    add_triton_plugin(TritonShared ${CMAKE_CURRENT_SOURCE_DIR}/triton_shared.cc LINK_LIBS TritonSharedAnalysis TritonToLinalg TritonTilingExtIR)
endif()


