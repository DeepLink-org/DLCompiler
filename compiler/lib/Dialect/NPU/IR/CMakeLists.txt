add_dicp_triton_library(NPUIR
  NPUOps.cpp
  NPUDialect.cpp

  DEPENDS
  NPUTableGen

  LINK_LIBS PUBLIC
  TritonIR
  MLIRIR
  )
