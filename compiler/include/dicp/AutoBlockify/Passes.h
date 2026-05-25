

#ifndef TRITON_ADAPTER_AUTO_BLOCKIFY_PASSES_H
#define TRITON_ADAPTER_AUTO_BLOCKIFY_PASSES_H

#include "dicp/AutoBlockify/AutoBlockify.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "dicp/AutoBlockify/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_AUTO_BLOCKIFY_PASSES_H
