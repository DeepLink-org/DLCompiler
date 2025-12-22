add_triton_library(TritonToLinalgNPUCoversion
  TritonToLinalgNPUConversionPass.cpp


  DEPENDS
  TritonToLinalgNPUCoversionPassIncGen

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
  TPtrIR
  TritonIR
  TritonTransforms
  TritonSharedAnalysis
  TritonSharedUtils
  TritonToLinalgExperimental

  TritonArithToLinalg
  StructuredToMemref
  TritonToStructured
  TritonToUnstructured
  TritonPtrToMemref
  UnstructuredToMemref
  MemRefCopyGatherToTensorInsert
  TritonToUnstructure
)
