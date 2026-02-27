add_triton_library(DICPTransformOps
  TransformsUtils.cpp
  DialectExtension.cpp
  DicpTransformOps.cpp

  DEPENDS
  DICPTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIndexDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgTransformOps
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  MLIRTransformDialect
  MLIRTransformDialectUtils
  MLIRTransformUtils
  )