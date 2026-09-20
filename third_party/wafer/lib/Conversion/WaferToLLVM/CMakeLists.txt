add_triton_library(WaferToLLVM
  WaferToLLVM.cpp
  KernelArgBufferPass.cpp

  DEPENDS
  WaferToLLVMConversionPassIncGen
  KernelArgBufferPassIncGen
  MLIRMemRefToLLVM

  LINK_LIBS PUBLIC
  MLIRMemRefToLLVM
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIRArithToLLVM
  MLIRAffineToStandard
  MLIRLinalgToStandard
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRTransforms
)
