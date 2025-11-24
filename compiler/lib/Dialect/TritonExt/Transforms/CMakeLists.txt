add_triton_library(TritonExtTransforms
  CanonicalizeTritonIRAscend.cpp
  CanonicalizeCmpiPass.cpp
  CanonicalizerPattern.cpp

  DEPENDS
  TritonExtTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport

  TritonIR
  TritonSharedAnalysis
  TritonArithToLinalg
  TritonToStructured
)
