if(FLAGTREE_BACKEND STREQUAL "wafer")
add_triton_library(TritonSharedAnalysisStructured
  PtrAnalysisTS.cpp

  DEPENDS
  TritonTableGen
  TritonStructuredTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonStructuredIR
  MLIRAnalysis
)
else()
add_triton_library(TritonSharedAnalysisStructured
  PtrAnalysis.cpp

  DEPENDS
  TritonTableGen
  TritonStructuredTableGen

  LINK_LIBS PUBLIC
  TritonStructuredIR
  MLIRAnalysis
)
endif()