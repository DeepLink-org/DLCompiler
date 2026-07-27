include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(
    TritonMETAX
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_metax.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/Gluon/GluonPythonBindings.cpp
    LINK_LIBS TritonMETAXGPUToLLVM TritonMETAXGPUTransforms
              TritonMETAXGPUGluon)
  target_link_libraries(TritonMETAX PRIVATE Python3::Module pybind11::headers)
endif()
