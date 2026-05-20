

#ifndef TRITON_ADAPTER_DISCRETE_MASK_ACCESS_CONVERSION_PASSES_H
#define TRITON_ADAPTER_DISCRETE_MASK_ACCESS_CONVERSION_PASSES_H

#include "dicp/DiscreteMaskAccessConversion/DiscreteMaskAccessConversionPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "dicp/DiscreteMaskAccessConversion/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_DISCRETE_MASK_ACCESS_CONVERSION_PASSES_H
