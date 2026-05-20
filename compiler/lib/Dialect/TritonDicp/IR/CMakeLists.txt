add_triton_library(TritonDicpIR
  TritonDicpAttrs.cpp
  TritonDicpDialect.cpp
  TritonDicpOps.cpp

  DEPENDS
  TritonDicpTableGen
  TritonDicpAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  TritonIR
)
