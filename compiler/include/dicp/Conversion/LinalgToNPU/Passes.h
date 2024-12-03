
#pragma once
#include "dicp/Conversion/LinalgToNPU/LinalgToNPU.h"

namespace mlir::dicp::npu {

#define GEN_PASS_REGISTRATION
#include "dicp/Conversion/LinalgToNPU/Passes.h.inc"

} // namespace mlir::dicp::npu
