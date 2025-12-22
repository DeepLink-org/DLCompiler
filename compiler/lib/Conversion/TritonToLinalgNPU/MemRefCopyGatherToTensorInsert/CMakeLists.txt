add_triton_library(MemRefCopyGatherToTensorInsert
  MemRefCopyGatherToTensorInsert.cpp

  DEPENDS
  MemRefCopyGatherToTensorInsertPassIncGen

  LINK_LIBS
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
)
