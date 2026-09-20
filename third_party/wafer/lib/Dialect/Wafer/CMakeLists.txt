add_triton_library(WaferIR
  IR/WaferDialect.cpp
  IR/WaferOps.cpp

  DEPENDS
  WaferTableGen

  LINK_LIBS PUBLIC
  MLIRIR
)

add_subdirectory(Transforms)
