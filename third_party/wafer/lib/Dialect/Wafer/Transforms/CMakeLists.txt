add_triton_library(WaferTransforms
  InsertBarrierPass.cpp
  DEPENDS WaferTransformsPassIncGen WaferTableGen
  LINK_LIBS PUBLIC ZTCAnalysis WaferIR MLIRSCFDialect MLIRFuncDialect MLIRMemRefDialect MLIRPass
)
