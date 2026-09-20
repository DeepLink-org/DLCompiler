add_triton_library(ZTCAnalysis
  Allocation.cpp
  Alias.cpp
  Membar.cpp

  DEPENDS
  TritonTableGen
  WaferTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
)
