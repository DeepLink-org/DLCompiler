add_triton_library(TritonToLinalg
  TritonToLinalg.cpp
  TritonToLinalgPass.cpp

  DEPENDS
  TritonToLinalgConversionPassIncGen
  MLIRMathExtDialectIncGen
  MLIRMathExtOpsIncGen

  LINK_LIBS PUBLIC
  TritonTilingExtIR
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonSharedAnalysis
)
