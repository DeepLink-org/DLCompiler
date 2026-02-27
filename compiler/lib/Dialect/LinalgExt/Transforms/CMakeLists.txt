add_triton_library(LinalgExtTransforms
  LinalgIfToSelect.cpp
  LinalgGenericToSCF.cpp
  ScalarTo1DTensorPass.cpp
  RemoveSingleIterationLoop.cpp
  TensorTransform.cpp
  NPUUnroolPipeline.cpp
  NPUVectorTileTransform.cpp
  NPUTileLoopTagging.cpp
  DeLinalgize.cpp
  FuseLoop.cpp
  LoopUnrollStage.cpp
  ShrinkBuffers.cpp

  DEPENDS
  LinalgExtTransformsIncGen
  LinalgExtAnalysis

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
  MLIRAnalysis
  MLIRSCFUtils
  MLIRSCFTransforms
  MLIRLinalgTransformOps
  MLIRLinalgUtils
  MLIRTransformUtils
  MLIRTransformDialectTransforms
  MLIRTensorTransforms
  TritonAnalysis
  TritonIR
  TritonTransforms
  TritonSharedAnalysis

  DICPTransformOps

  TritonArithToLinalg
  StructuredToMemref
  TritonToStructured
)
