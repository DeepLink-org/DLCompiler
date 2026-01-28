add_triton_library(LinalgExtTransforms
  LinalgIfToSelect.cpp
  LinalgGenericToSCF.cpp
  ScalarTo1DTensorPass.cpp
  RemoveSingleIterationLoop.cpp
  TensorTransform.cpp
  AnnotateTransposePass.cpp

  DEPENDS
  LinalgExtTransformsIncGen

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

  TritonArithToLinalg
  StructuredToMemref
  TritonToStructured
)