add_triton_library(WaferMemrefToLLVM
  WaferMemrefToLLVM.cpp
  WaferMemrefToLLVMPass.cpp

  DEPENDS
  WaferMemrefToLLVMConversionPassIncGen
  TritonSharedUtils

  LINK_LIBS PUBLIC
  TritonSharedUtils
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
)
