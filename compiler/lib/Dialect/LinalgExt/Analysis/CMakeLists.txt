add_triton_library(LinalgExtAnalysis
  BufferEscapeAnalysis.cpp
  ContractionClosureAnalysis.cpp
  DimAnalyzer.cpp
  StagePartition.cpp
  StageDependencyAnalyzer.cpp
  StageUtils.cpp
  SliceParallelAnalysis.cpp

  LINK_LIBS PUBLIC

  MLIRAffineDialect
  MLIRArithDialect
  MLIRAnalysis
  MLIRBufferizationDialect
  MLIRCallInterfaces
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRMemRefDialect
  MLIRPass
  MLIRShapeDialect
  MLIRTensorDialect
  MLIRTensorUtils
  MLIRViewLikeInterface
  DICPUtils
)
