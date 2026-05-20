

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PASSES_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PASSES_H

#include "dicp/DynamicCVPipeline/AddDynamicCVPipeline.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "dicp/DynamicCVPipeline/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PASSES_H