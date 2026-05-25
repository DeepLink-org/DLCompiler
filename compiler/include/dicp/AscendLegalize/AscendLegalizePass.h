#ifndef TRITON_ADAPTER_ASCENDLEGALIZE_H
#define TRITON_ADAPTER_ASCENDLEGALIZE_H

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/PatternMatch.h"

#define GEN_PASS_DECL_ASCENDLEGALIZE
#include "dicp/AscendLegalize/Passes.h.inc"

#define GEN_PASS_DEF_ASCENDLEGALIZE
#include "dicp/AscendLegalize/Passes.h.inc"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAscendLegalizePass();

} // namespace triton
} // namespace mlir

namespace {

using namespace mlir;
using namespace triton;

struct FlipCmpiPredicatePattern : public OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern<arith::CmpIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp cmpOp,
                                PatternRewriter &rewriter) const override;
};

class AscendLegalizePass
    : public ::impl::AscendLegalizeBase<AscendLegalizePass> {
public:
  using AscendLegalizeBase::AscendLegalizeBase;
  void runOnOperation() override;
};

} // namespace

#endif // TRITON_ADAPTER_ASCENDLEGALIZE_H
