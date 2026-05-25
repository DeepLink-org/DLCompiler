

#ifndef TRITON_ADAPTER_CONVERSION_MARKTENSORKINDPASS_H
#define TRITON_ADAPTER_CONVERSION_MARKTENSORKINDPASS_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define GEN_PASS_DEF_MARKTENSORKIND
#include "dicp/TritonToLinalg/Passes.h.inc"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createMarkTensorKindPass();

enum TensorKind { NONE = -1, INPUT = 0, OUTPUT = 1, INPUT_OUTPUT = 2 };

} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace triton;

class MarkTensorKindPass
    : public ::impl::MarkTensorKindBase<MarkTensorKindPass> {
public:
  MarkTensorKindPass() = default;

  void runOnOperation() override;
};

#endif // TRITON_ADAPTER_CONVERSION_MARKTENSORKINDPASS_H