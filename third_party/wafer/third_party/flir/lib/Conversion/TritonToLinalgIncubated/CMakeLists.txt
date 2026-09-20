add_triton_library(TritonToLinalgIncubated
        TritonToLinalgIncubatedPass.cpp
        LoadStoreConverter.cpp
        FunctionConverter.cpp
        ArgMinMaxConverter.cpp
        TritonOpConverter.cpp
        HoistBroadcast.cpp
        BlockPtrAnalysis.cpp
        MaskAnalysis.cpp
        UseAnalysis.cpp
        DescriptorConverter.cpp

  DEPENDS
  TritonToLinalgIncubatedConversionPassIncGen

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
  TleToLinalg
)
