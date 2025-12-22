add_triton_library(DiscreteMaskAccessConversion
  DiscreteMaskAccessConversionPass.cpp
  MaskAnalysis.cpp

  DEPENDS
  DiscreteMaskAccessConversionPassIncGen

  LINK_LIBS
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
  MLIRAnalysis
)
