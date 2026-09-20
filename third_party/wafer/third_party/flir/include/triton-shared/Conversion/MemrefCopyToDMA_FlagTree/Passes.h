#ifndef TRITON_MEMREF_COPY_TO_DMA_FLAGTREE_CONVERSION_PASSES_H
#define TRITON_MEMREF_COPY_TO_DMA_FLAGTREE_CONVERSION_PASSES_H

#include "triton-shared/Conversion/MemrefCopyToDMA_FlagTree/MemrefCopyToDMAFlagTree.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/MemrefCopyToDMA_FlagTree/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
