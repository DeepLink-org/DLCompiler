#ifndef MEMREF_COPY_TO_DMA_FLAGTREE_CONVERSION_PASSES
#define MEMREF_COPY_TO_DMA_FLAGTREE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def MemrefCopyToDMAFlagTree : Pass<"memref-copy-to-dma-flagtree", "mlir::ModuleOp"> {
  let summary = "Convert memrefcopy to DMA";
}

#endif
