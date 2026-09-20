add_triton_library(MKToWafer
  MKToWafer.cpp
  MKToWaferPass.cpp

  DEPENDS
  WaferTableGen
  MKToWaferConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
)
