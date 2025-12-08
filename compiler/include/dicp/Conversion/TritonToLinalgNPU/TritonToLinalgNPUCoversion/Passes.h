#ifndef TRITON_TO_LINALG_NPU_CONVERSION_PASSES_H
#define TRITON_TO_LINALG_NPU_CONVERSION_PASSES_H

#include "dicp/Conversion/TritonToLinalgNPU/TritonToLinalgNPUCoversion/TritonToLinalgNPUCoversion.h"

namespace mlir::dicp::linked {
#define GEN_PASS_REGISTRATION
#include "dicp/Conversion/TritonToLinalgNPU/TritonToLinalgNPUCoversion/Passes.h.inc"

} // namespace mlir::dicp::linked

#endif
