add_dicp_triton_library(LinalgToNPU
  LinalgToNPUPass.cpp
  LinalgToNPU.cpp

  DEPENDS
  NPUTableGen
  LinalgToNPUConversionPassIncGen

  LINK_LIBS PUBLIC
  
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  NPUIR
)
