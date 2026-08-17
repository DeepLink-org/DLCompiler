add_triton_library(GluonMetaXTransforms
  AccelerateMatmul.cpp
  AlignMmaConsumers.cpp
  BlockedLayoutCandidates.cpp
  BufferOffsetCalculation.cpp
  ExpandLayoutCandidates.cpp
  GluonToTritonGPU.cpp
  InsertRequireLayout.cpp
  LayoutCandidates.cpp
  MmaLayoutCandidates.cpp
  PropagateLayout.cpp
  RewriteLocalAlias.cpp
  SharedLayoutCandidates.cpp
  StorageAliasAllocation.cpp
  StorageAliasLowering.cpp
  StorageAliasSizeDefinition.cpp

  DEPENDS
  GluonMetaXTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonAnalysis
  TritonGPUTransforms
  GluonMetaXAnalysis
  GluonIR
  MLIRAnalysis
  MLIRTransformUtils
)

if(TRITON_USE_MACA)
  target_link_libraries(GluonMetaXTransforms PUBLIC TritonMETAXGPUTransforms)
endif()
