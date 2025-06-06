add_subdirectory(triton-adapter triton-adapter)

add_triton_plugin(TritonAscend ${CMAKE_CURRENT_SOURCE_DIR}/triton_ascend.cpp)