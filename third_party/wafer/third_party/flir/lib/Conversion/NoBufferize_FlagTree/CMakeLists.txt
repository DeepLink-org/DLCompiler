add_triton_library(NoBufferizeFlagTree
  NoBufferizeFlagTree.cpp
  NoBufferizeFlagTreePass.cpp

  DEPENDS
  NoBufferizeFlagTreeConversionPassIncGen
  Registrar
  Common

  LINK_LIBS PUBLIC
  MLIRSCFTransforms
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonTilingExtIR
  TritonStructuredIR
)
