add_triton_library(LinalgToNPU
  LinalgToNPUPass.cpp
  LinalgToNPU.cpp

  DEPENDS
  NPUTableGen
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
  NPUIR

  TritonArithToLinalg
  StructuredToMemref
  TritonToStructured
)
