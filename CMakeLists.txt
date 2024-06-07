
set(DC_TRITON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DC_TRITON_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(DC_TRITON_INCLUDE_DIR "")
set(DC_TRITON_LINK_DIR "")

include(cmake/utils.cmake)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/triton_linalg/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/triton_linalg)

