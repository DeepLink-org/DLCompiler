add_triton_library(LinalgToLinked
  LinalgToLinkedPass.cpp
  DebugCPUVerifyPass.cpp
  VerifyNoLinalgGenericPass.cpp
  TritonOpConverter.cpp

  DEPENDS
  DICPNPUIncGen
  LinalgToLinkedConversionPassIncGen

  LINK_LIBS PUBLIC
  BiShengIRAnnotationDialect
  BiShengIRHIVMDialect
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
