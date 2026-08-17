add_triton_library(GluonMetaXAnalysis
  LayoutPropagation.cpp

  DEPENDS
  GluonMetaXTableGen
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  GluonIR
  TritonIR
  TritonGPUIR
  MLIRAnalysis
)
