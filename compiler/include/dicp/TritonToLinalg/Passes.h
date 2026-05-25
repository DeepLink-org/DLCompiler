

#ifndef TRITON_ADAPTER_TRITON_TO_LINALG_CONVERSION_PASSES_H
#define TRITON_ADAPTER_TRITON_TO_LINALG_CONVERSION_PASSES_H

#include "dicp/TritonToLinalg/AscendNPUIRLegalizePass.h"
#include "dicp/TritonToLinalg/MarkTensorKindPass.h"
#include "dicp/TritonToLinalg/TritonToLinalgPass.h"

namespace mlir::triton {

#define GEN_PASS_REGISTRATION
#include "dicp/TritonToLinalg/Passes.h.inc"

} // namespace mlir::triton

#endif // TRITON_ADAPTER_TRITON_TO_LINALG_CONVERSION_PASSES_H
