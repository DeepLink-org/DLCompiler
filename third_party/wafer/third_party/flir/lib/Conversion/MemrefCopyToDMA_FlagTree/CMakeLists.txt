add_triton_library(MemrefCopyToDMAFlagTree
  MemrefCopyToDMAFlagTree.cpp
  MemrefCopyToDMAFlagTreePass.cpp

  DEPENDS
  MemrefCopyToDMAFlagTreeConversionPassIncGen
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
