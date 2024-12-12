add_dicp_triton_library(DICPNPU
  NPUOps.cpp
  NPUDialect.cpp
  NPUTypes.cpp

  DEPENDS
  DICPNPUIncGen

  LINK_LIBS PUBLIC
  TritonIR
  MLIRIR
  )
