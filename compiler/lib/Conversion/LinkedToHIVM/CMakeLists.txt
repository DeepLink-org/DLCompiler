add_triton_library(LinkedToHIVM
  LinkedToHIVM.cpp

  DEPENDS
  LinkedToHIVMConversionPassIncGen
  # HIVMSynchronizationOpsIncGen      # 将依赖的sync功能，单独依赖起来。

  LINK_LIBS
  BiShengIRHIVMDialect          # 依赖llvm具体实现。会导致版本不一致。
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
)
