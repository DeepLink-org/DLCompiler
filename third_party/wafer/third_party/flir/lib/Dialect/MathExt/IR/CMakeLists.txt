add_mlir_dialect_library(MLIRMathExtDialect
  MathExtOps.cpp
  MathExtDialect.cpp

  DEPENDS
  MLIRMathExtDialectIncGen
  MLIRMathExtOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
