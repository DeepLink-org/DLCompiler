add_triton_library(TritonToUnstructureIncubated
        UnstructureConversionPass.cpp
        OffsetAnalysis.cpp
        BubbleUpOperation.cpp

  DEPENDS
  TritonToUnstructureConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonAnalysis
  MLIRSCFTransforms
)
