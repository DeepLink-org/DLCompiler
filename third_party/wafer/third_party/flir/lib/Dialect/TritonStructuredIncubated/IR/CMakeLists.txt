add_triton_library(TritonStructuredIncubatedIR
  TritonStructuredOpsIncubated.cpp
  TritonStructuredDialectIncubated.cpp

  DEPENDS
  TritonStructuredIncubatedTableGen

  LINK_LIBS PUBLIC
  TritonIR
  MLIRIR
  )
