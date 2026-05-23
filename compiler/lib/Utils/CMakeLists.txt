add_triton_library(DICPUtils
  Utils.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  TritonIR
  MLIRTransformDialect
  MLIRTransformDialectUtils
)
