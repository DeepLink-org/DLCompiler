

#ifndef TRITON_ADAPTER_TRITON_TO_STRUCTURE_CONVERSION_PASSES_H
#define TRITON_ADAPTER_TRITON_TO_STRUCTURE_CONVERSION_PASSES_H

#include "dicp/TritonToStructured/TritonToStructuredPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "dicp/TritonToStructured/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_TRITON_TO_UNSTRUCTURE_CONVERSION_PASSES_H
