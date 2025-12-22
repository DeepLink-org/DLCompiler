#ifndef TRITON_DLC_DISCRETE_MASK_ACCESS_CONVERSION_PASSES_H
#define TRITON_DLC_DISCRETE_MASK_ACCESS_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/PatternMatch.h"

#define GEN_PASS_DECL_DISCRETEMASKACCESSCONVERSION
#include "dicp/Conversion//DiscreteMaskAccessConversion/Passes.h.inc"

#define GEN_PASS_DEF_DISCRETEMASKACCESSCONVERSION
#include "dicp/Conversion//DiscreteMaskAccessConversion/Passes.h.inc"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createDiscreteMaskAccessConversionPass(
    const DiscreteMaskAccessConversionOptions &options = {});

} // namespace triton
} // namespace mlir

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "dicp/Conversion//DiscreteMaskAccessConversion/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_DLC_DISCRETE_MASK_ACCESS_CONVERSION_PASSES_H
