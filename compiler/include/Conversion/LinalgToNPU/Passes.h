
#pragma once
#include "compiler/include/Conversion/LinalgToNPU/LinalgToNPU.h"

namespace mlir {
namespace npu {

#define GEN_PASS_REGISTRATION
#include "compiler/include/Conversion/LinalgToNPU/Passes.h.inc"

} // namespace npu
} // namespace mlir
