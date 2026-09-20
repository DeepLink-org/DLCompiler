set(WAFER_BUILD_ROLE "tools" CACHE STRING "Wafer component: frontend or tools")
set_property(CACHE WAFER_BUILD_ROLE PROPERTY STRINGS frontend tools)

# The Wafer Python extension contains only the TTIR/DSA frontend. FLIR's
# dialects and lowering live in the separate wafer-opt build. Neither build
# links the original DLCompiler C++ plugin.
if(WAFER_BUILD_ROLE STREQUAL "frontend")
  if(NOT TRITON_BUILD_PYTHON_MODULE)
    message(FATAL_ERROR "Wafer frontend requires TRITON_BUILD_PYTHON_MODULE")
  endif()
  set(WAFER_TLE_BUILD_CONVERSIONS OFF)
  add_subdirectory(third_party/tle)
  add_triton_plugin(TritonWafer
    ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_wafer_frontend.cc
    LINK_LIBS TleDsaIR MLIRArithDialect MLIRLinalgDialect MLIRTensorDialect
              MLIRVectorDialect MLIRFuncDialect MLIRFuncAllExtensions
              MLIRArithTransforms MLIRBufferizationTransforms
              MLIRLinalgTransforms MLIRSCFTransforms MLIRTensorTransforms)
  target_link_libraries(TritonWafer PRIVATE Python3::Module pybind11::headers)
  return()
elseif(NOT WAFER_BUILD_ROLE STREQUAL "tools")
  message(FATAL_ERROR "WAFER_BUILD_ROLE must be frontend or tools")
endif()

if(TARGET TritonToLinalg OR TARGET TritonStructuredIR)
  message(FATAL_ERROR
    "Wafer tools require a separate build tree without the DICP plugin. "
    "Build the original DICP plugin and Wafer in separate build trees.")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/WaferConfig.cmake")

if(NOT DEFINED WAFER_DEPS_ROOT)
  if(DEFINED ENV{WAFER_DEPS_ROOT})
    set(WAFER_DEPS_ROOT $ENV{WAFER_DEPS_ROOT})
  else()
    message(STATUS "WAFER_DEPS_ROOT not set, CRT/Profiler will be disabled")
    set(WAFER_ENABLE_CRT OFF)
  endif()
endif()

# Check if WAFER_DEPS_ROOT exists to enable CRT/Profiler
if(DEFINED WAFER_DEPS_ROOT AND EXISTS "${WAFER_DEPS_ROOT}")
  set(WAFER_ENABLE_CRT ON)
else()
  set(WAFER_ENABLE_CRT OFF)
  if(DEFINED WAFER_DEPS_ROOT)
    message(WARNING "WAFER_DEPS_ROOT set but path does not exist: ${WAFER_DEPS_ROOT}")
  endif()
endif()
# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

if(NOT DEFINED USE_HOST_PROFILE)
  if(DEFINED ENV{USE_HOST_PROFILE})
    set(USE_HOST_PROFILE $ENV{USE_HOST_PROFILE})
  endif()
endif()

if(NOT DEFINED ENABLE_PROFILING)
  if(DEFINED ENV{ENABLE_PROFILING})
    set(ENABLE_PROFILING $ENV{ENABLE_PROFILING})
  endif()
endif()

if(NOT DEFINED NO_INTRNISIC_RUN)
  if(DEFINED ENV{NO_INTRNISIC_RUN})
    set(NO_INTRNISIC_RUN $ENV{NO_INTRNISIC_RUN})
  endif()
endif()

if(NOT DEFINED ENABLE_SYNCHRONOUS_INTRINSIC)
  if(DEFINED ENV{ENABLE_SYNCHRONOUS_INTRINSIC})
    set(ENABLE_SYNCHRONOUS_INTRINSIC $ENV{ENABLE_SYNCHRONOUS_INTRINSIC})
  endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set FLAGTREE_BACKEND for FLIR to skip test/tools directories
set(FLAGTREE_BACKEND "wafer")

set(XUANTIE_NAME Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2)
set(INSTALL_WAFER_DIR ${CMAKE_INSTALL_PREFIX}/triton/backends/wafer/)
install(CODE "file(MAKE_DIRECTORY \"${INSTALL_WAFER_DIR}\")")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/crt/include)

if(NOT DEFINED WAFER_SDK_INCLUDE_DIR OR
   NOT EXISTS "${WAFER_SDK_INCLUDE_DIR}/instr_def.h")
  message(FATAL_ERROR
    "WAFER_SDK_INCLUDE_DIR must contain instr_def.h for Wafer lowering")
endif()
include_directories(${WAFER_SDK_INCLUDE_DIR})

# WAFER_DEPS_ROOT include only if CRT enabled
if(WAFER_ENABLE_CRT)
  include_directories(${WAFER_DEPS_ROOT}/include)
endif()

# FLIR (FlagTree Intermediate Representation) as local submodule
# FLIR provides TritonTilingExt and TritonStructured dialects
set(FLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flir)
set(FLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/third_party/flir)
include_directories(${FLIR_SOURCE_DIR}/include)
include_directories(${FLIR_BINARY_DIR}/include)

# TLE (Wafer Language Extension) - DSA dialect needed by TLEToMK pass
# Bundled in third_party/tle so wafer is independent of FlagTree
set(TLE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tle)
set(TLE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/third_party/tle)
# Make "tle/include/..." includes work (for lib/Conversion/TLEToMK/TLEToMK.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party)
# Make "third_party/tle/include/..." includes work (for bin/RegisterTritonDialects.h)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# When built via FlagTree (cmake source), FlagTree builds TLE itself.
# Pass -DWAFER_USE_EXTERNAL_TLE=ON to let the parent cmake provide TLE.
option(WAFER_USE_EXTERNAL_TLE "Use TLE provided by the parent cmake (e.g. FlagTree)" OFF)
if(NOT WAFER_USE_EXTERNAL_TLE)
    add_subdirectory(third_party/tle)
endif()
add_subdirectory(lib)
add_subdirectory(include)
add_subdirectory(third_party/flir)
add_subdirectory(bin)

# Conditionally add CRT and Profiler
if(WAFER_ENABLE_CRT)
  add_subdirectory(crt)
  add_subdirectory(profiler)
else()
  message(STATUS "Building wafer without CRT/Profiler")
endif()

if(TRITON_BUILD_PYTHON_MODULE)
  # FIXME: Unify the libraries for Wafer into fewer ones
  # FLIR libraries (from third_party/flir):
  # - TritonTilingExtIR, TritonStructuredIR: Dialect libraries
  # - TritonToLinalg, TritonToStructured: Conversion passes
  add_triton_plugin(TritonWafer ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_wafer.cc
                    LINK_LIBS TritonSharedAnalysis TritonSharedAnalysisStructured MagicKernelIR
                    WaferIR TritonTilingExtIR TritonStructuredIR TritonToCoreDialects
                    TritonToLinalg WaferTritonToStructured StructuredToMemref LinalgToMagicKernel
                    TritonArithToLinalg CoreDialectsToMK WaferToLLVM AllocateSharedMemory ExportKernelSymbols
                    WaferMemrefToLLVM MKToWafer
                    MLIRFuncAllExtensions)
  target_link_libraries(TritonWafer PRIVATE Python3::Module pybind11::headers)
endif()
#if(TRITON_BUILD_UT)
#  add_subdirectory(unittest)
#endif()
#add_subdirectory(test)
