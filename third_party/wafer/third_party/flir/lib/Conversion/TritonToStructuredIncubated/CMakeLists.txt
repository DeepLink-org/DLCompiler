add_triton_library(TritonToStructuredIncubated
        TritonToStructuredIncubatedPass.cpp
        PtrAnalysis.cpp
        CannonicalizerConverter.cpp
        MemOpConverter.cpp
        MaskAnalysis.cpp

  DEPENDS
  TritonToStructuredIncubatedConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonAnalysis
  MLIRTritonNPUUtils
  MLIRSCFTransforms
  MLIRLinalgTransforms
  BiShengIRHIVMDialect
)
