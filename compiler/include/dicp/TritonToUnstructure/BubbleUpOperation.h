

#pragma once

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/PatternMatch.h"

#define GEN_PASS_DECL_BUBBLEUPOPERATION
#include "dicp/TritonToUnstructure/Passes.h.inc"

#define GEN_PASS_DEF_BUBBLEUPOPERATION
#include "dicp/TritonToUnstructure/Passes.h.inc"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createBubbleUpOperationPass(const BubbleUpOperationOptions &options = {});

} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace triton;

template <typename ExtractOpTy>
class BubbleUpExtract : public OpRewritePattern<ExtractOpTy> {
  static_assert(std::is_same_v<ExtractOpTy, tensor::ExtractOp> ||
                std::is_same_v<ExtractOpTy, tensor::ExtractSliceOp>);

public:
  using OpRewritePattern<ExtractOpTy>::OpRewritePattern;

  explicit BubbleUpExtract(MLIRContext *context, bool enableAggressiveMode);

  LogicalResult matchAndRewrite(ExtractOpTy op,
                                PatternRewriter &rewriter) const override;

private:
  Value createExtractOp(ExtractOpTy op, Value value, Location loc,
                        PatternRewriter &rewriter) const;
  template <typename BinOpTy>
  void bubbleUpIntBinaryOp(ExtractOpTy op, BinOpTy binOp, Location loc,
                           PatternRewriter &rewriter) const;
  template <typename BinOpTy>
  void bubbleUpFloatBinaryOp(ExtractOpTy op, BinOpTy binOp, Location loc,
                             PatternRewriter &rewriter) const;

  void bubbleUpOperation(ExtractOpTy op, arith::ExtSIOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, arith::CmpIOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, arith::TruncFOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, arith::ExtFOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, arith::FPToSIOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, arith::SIToFPOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, triton::ClampFOp parentOp,
                         Location loc, PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, arith::CmpFOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, triton::BroadcastOp parentOp,
                         Location loc, PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, triton::ExpandDimsOp parentOp,
                         Location loc, PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, triton::SplatOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, triton::MakeRangeOp parentOp,
                         Location loc, PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, triton::AddPtrOp parentOp,
                         Location loc, PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, math::FloorOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, math::CeilOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, tensor::ExtractSliceOp parentOp,
                         Location loc, PatternRewriter &rewriter) const;

  bool enableAggressiveMode;
};

class BubbleUpOperationPass
    : public ::impl::BubbleUpOperationBase<BubbleUpOperationPass> {
public:
  explicit BubbleUpOperationPass(const BubbleUpOperationOptions &options);
  void runOnOperation() override;
};
