#ifndef TRITON_NO_BUFFERIZE_FLAGTREE_CONVERSION_PASSES_H
#define TRITON_NO_BUFFERIZE_FLAGTREE_CONVERSION_PASSES_H

#include "triton-shared/Conversion/NoBufferize_FlagTree/NoBufferizeFlagTree.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/NoBufferize_FlagTree/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
