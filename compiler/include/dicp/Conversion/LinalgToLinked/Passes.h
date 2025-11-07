
#pragma once
#include "dicp/Conversion/LinalgToLinked/LinalgToLinked.h"

namespace mlir::dicp::linked {

#define GEN_PASS_REGISTRATION
#include "dicp/Conversion/LinalgToLinked/Passes.h.inc"

} // namespace mlir::dicp::linked
