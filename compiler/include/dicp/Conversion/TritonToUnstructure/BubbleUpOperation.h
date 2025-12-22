#pragma once

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/PatternMatch.h"

#define GEN_PASS_DECL_BUBBLEUPOPERATION
#include "dicp/Conversion/TritonToUnstructure/Passes.h.inc"

#define GEN_PASS_DEF_BUBBLEUPOPERATION
#include "dicp/Conversion/TritonToUnstructure/Passes.h.inc"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createBubbleUpOperationPass(const BubbleUpOperationOptions &options = {});

} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace triton;

class BubbleUpOperationPass
    : public ::impl::BubbleUpOperationBase<BubbleUpOperationPass> {
public:
  explicit BubbleUpOperationPass(const BubbleUpOperationOptions &options);
  void runOnOperation() override;
};
