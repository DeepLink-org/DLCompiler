add_triton_library(LinalgToLinked
  LinalgToLinkedPass.cpp
  VerifyNoLinalgGenericPass.cpp

  DEPENDS
  DICPNPUIncGen
  LinalgToLinkedConversionPassIncGen

  LINK_LIBS PUBLIC
  BiShengIRAnnotationDialect
  TritonTilingExtIR
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonAnalysis
  TritonIR
  TritonTransforms
  TritonSharedAnalysis
  DICPNPU

  TritonArithToLinalg
  StructuredToMemref
  TritonToStructured
)
