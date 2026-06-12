

#ifndef TRITON_ADAPTER_DISCRETEMASKACCESSCONVERSION_H
#define TRITON_ADAPTER_DISCRETEMASKACCESSCONVERSION_H

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/PatternMatch.h"

#define GEN_PASS_DECL_DISCRETEMASKACCESSCONVERSION
#include "dicp/DiscreteMaskAccessConversion/Passes.h.inc"

#define GEN_PASS_DEF_DISCRETEMASKACCESSCONVERSION
#include "dicp/DiscreteMaskAccessConversion/Passes.h.inc"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createDiscreteMaskAccessConversionPass(
    const DiscreteMaskAccessConversionOptions &options = {});

} // namespace triton
} // namespace mlir

namespace {

using namespace mlir;
using namespace triton;

class DiscreteMaskAccessConversionPass
    : public ::impl::DiscreteMaskAccessConversionBase<
          DiscreteMaskAccessConversionPass> {
public:
  explicit DiscreteMaskAccessConversionPass(
      const DiscreteMaskAccessConversionOptions &options);
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};

} // namespace

#endif // DISCRETE_MASK_ACCESS_CONVERSION_H
