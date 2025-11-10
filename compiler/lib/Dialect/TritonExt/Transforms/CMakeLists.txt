add_triton_library(TritonExtTransforms
  BoolTritonPtrPromotionPass.cpp
  CanonicalizeCmpiPass.cpp

  DEPENDS
  TritonExtTransformsIncGen

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
  TritonAnalysis
  TritonIR
  TritonTransforms
  TritonSharedAnalysis

  TritonArithToLinalg
  StructuredToMemref
  TritonToStructured
)
