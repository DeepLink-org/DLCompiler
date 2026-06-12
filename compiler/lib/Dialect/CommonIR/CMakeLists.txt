add_triton_library(CommonIRTransforms
  VectorizeParallelLoopPass.cpp
  AnnotateKernelAttrsPass.cpp

  DEPENDS
  CommonIRPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
)
