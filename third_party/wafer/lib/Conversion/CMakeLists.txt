add_subdirectory(TritonArithToLinalg)
add_subdirectory(StructuredToMemref)
add_subdirectory(ConvertTritonPtr)
# TritonPtrToMemref is provided by FLIR
add_subdirectory(WaferMemrefToLLVM)
add_subdirectory(LinalgToMK)
add_subdirectory(MKToWafer)
add_subdirectory(WaferToLLVM)
add_subdirectory(TritonToCoreDialects)
add_subdirectory(CoreDialectsToMK)
add_subdirectory(LegalizeTensorFormLoops)
add_subdirectory(LinalgTiling)
add_subdirectory(LinalgFusion)
add_subdirectory(AllocateSharedMemory)
add_subdirectory(ExportKernelSymbols)
add_subdirectory(TLEToMK)
add_subdirectory(UnstructuredToMK)
add_subdirectory(ReconcilePtrCasts)

# FLIR conversions are built from third_party/flir/:
# - TritonToLinalg
# - TritonToStructured
# - TritonToUnstructured
# - UnstructuredToMemref

add_subdirectory(MKPipeline)
