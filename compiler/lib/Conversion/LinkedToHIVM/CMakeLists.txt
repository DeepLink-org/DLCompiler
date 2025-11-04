add_triton_library(LinkedToHIVM
  LinkedToHIVM.cpp

  DEPENDS
  LinkedToHIVMConversionPassIncGen

  LINK_LIBS
  BiShengIRHIVMDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
)
