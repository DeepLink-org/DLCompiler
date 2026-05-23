add_triton_library(LinalgExtTransforms
  LinalgIfToSelect.cpp
  LinalgGenericToSCF.cpp
  ScalarTo1DTensorPass.cpp
  RemoveSingleIterationLoop.cpp
  TensorTransform.cpp
  VectorizeParallelLoopPass.cpp
  PipelineLoopUnroll.cpp
  CanonicalizeTileableOps.cpp
  DuplicateStageProducers.cpp
  ApplyTilingPlan.cpp
  StageTilingPlan.cpp
  DeLinalgize.cpp
  LowerForallToFor.cpp
  UnrollStageLoops.cpp
  ShrinkBuffers.cpp
  FuseLoop.cpp
  PrepareVectorTiling.cpp
  VerifyNoLocalBufferCopy.cpp
  PrepareInputBundleScheduling.cpp
  ScheduleInputBundles.cpp
  VectorTilingPipeline.cpp

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
  DICPUtils
)
