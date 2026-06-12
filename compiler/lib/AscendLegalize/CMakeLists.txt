add_triton_library(
  AscendLegalize
  AscendLegalizePass.cpp

  DEPENDS
  AscendLegalizePassIncGen

  LINK_LIBS PUBLIC
  BiShengIRHACCDialect
  MLIRArithDialect
  MLIRIR
  MLIRPass
  TritonIR
  MLIRTransforms
)
