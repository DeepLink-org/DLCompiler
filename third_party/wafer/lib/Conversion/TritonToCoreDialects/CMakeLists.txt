#===------------------------------------------------------------------------===#
#
# Copyright (c) Triton Project Contributors.
#
#===------------------------------------------------------------------------===#

add_triton_library(TritonToCoreDialects
  TritonToCoreDialectsPass.cpp

  DEPENDS
  TritonToCoreDialectsConversionPassIncGen
  TLEToMKConversionPassIncGen

  LINK_LIBS PUBLIC
  TritonTilingExtIR
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  MLIRMathExtDialect
  TritonIR
  TritonTransforms
  ZTCAnalysis

  TritonArithToLinalg
  StructuredToMemref
  UnstructuredToMemref
  WaferTritonToStructured
  TritonToUnstructured
  TLEToMagicKernel
  ConvertTritonPtr
  ReconcilePtrCasts
  TritonPtrToMemref
)
