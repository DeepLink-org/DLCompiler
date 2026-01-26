add_triton_library(LinalgExtAnalysis
  DimAnalyzer.cpp
  StageDependencyAnalyzer.cpp

  LINK_LIBS PUBLIC

  MLIRAffineDialect
  MLIRArithDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRMemRefDialect
  MLIRPass
  MLIRShapeDialect
  MLIRTensorDialect
  MLIRTensorUtils
)