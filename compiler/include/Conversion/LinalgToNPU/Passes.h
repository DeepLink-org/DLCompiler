
#pragma once
#include "compiler/Conversion/LinalgToNPU/LinalgToNPU.h"

namespace mlir {
namespace deeplink {
namespace npu {

#define GEN_PASS_REGISTRATION
#include "compiler/Conversion/LinalgToNPU/Passes.h.inc"

} // namespace npu
} // namespace deeplink
} // namespace mlir
