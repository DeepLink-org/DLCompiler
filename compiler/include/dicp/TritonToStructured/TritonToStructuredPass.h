
#ifndef TRITON_ADAPTER_CONVERSION_TRITONTOSTRUCTURED_H
#define TRITON_ADAPTER_CONVERSION_TRITONTOSTRUCTURED_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define GEN_PASS_CLASSES
#include "dicp/TritonToStructured/Passes.h.inc"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createTritonToStructuredPass();

std::unique_ptr<OperationPass<ModuleOp>> createTritonToStructuredPass(bool,
                                                                      bool);

} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace triton;

class TritonToStructuredPass
    : public TritonToStructuredBase<TritonToStructuredPass> {
public:
  TritonToStructuredPass() = default;

  TritonToStructuredPass(bool enableMaskFallbackConversion,
                         bool optimizeDynamicOffset) {
    this->enableMaskFallbackConversion = enableMaskFallbackConversion;
    this->optimizeDynamicOffset = optimizeDynamicOffset;
  };
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;

private:
  void populateTritonToStructuredCanonicalizationPatterns(
      RewritePatternSet &patterns);

  void populateTritonToStructuredPatterns(RewritePatternSet &patterns,
                                          bool optimizeDynamicOffset,
                                          bool enableMaskFallbackConversion);

  LogicalResult processSplatBinaryOperations(ModuleOp moduleOp);
};

#endif // TRITON_ADAPTER_CONVERSION_TRITONTOSTRUCTURED_H
