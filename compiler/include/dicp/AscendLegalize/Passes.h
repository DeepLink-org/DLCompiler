#ifndef TRITON_ADAPTER_ASCENDLEGALIZE_PASSES_H
#define TRITON_ADAPTER_ASCENDLEGALIZE_PASSES_H

#include "dicp/AscendLegalize/AscendLegalizePass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "dicp/AscendLegalize/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_ASCENDLEGALIZE_PASSES_H
