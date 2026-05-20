add_triton_library(AscendLegalize
        AscendLegalizePass.cpp

  DEPENDS
  AscendLegalizePassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  TritonIR
  MLIRTransforms
)
