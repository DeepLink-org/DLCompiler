add_triton_library(DICPTransformOps
  TransformsUtils.cpp
  DialectExtension.cpp
  DicpTransformOps.cpp

  DEPENDS
  DICPTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRIndexDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgTransformOps
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  MLIRTransformDialect
  MLIRTransformDialectUtils
  MLIRTransformUtils
  )