add_triton_library(DICPUtils
  Utils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  TritonIR
  MLIRTransformDialect
  MLIRTransformDialectUtils
)