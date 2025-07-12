add_triton_library(LinalgToNPU
  LinalgToNPUPass.cpp
  LinalgToNPU.cpp
  ConvertRankedToUnrankedPass.cpp

  DEPENDS
  DICPNPUIncGen
  LinalgToNPUConversionPassIncGen

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
  TritonAnalysis
  TritonIR
  TritonTransforms
  TritonSharedAnalysis
  DICPNPU

  TritonArithToLinalg
  StructuredToMemref
  TritonToStructured
)
