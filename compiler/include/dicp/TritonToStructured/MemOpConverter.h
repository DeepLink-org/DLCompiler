
#ifndef TRITON_ADAPTER_MEMOPCONVERTER_H
#define TRITON_ADAPTER_MEMOPCONVERTER_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "dicp/TritonToStructured/MaskAnalysis.h"
#include "dicp/TritonToStructured/PtrAnalysis.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace MemOpConverter {

using namespace mlir;
using namespace triton;

class LoadConverter : public OpRewritePattern<triton::LoadOp> {
public:
  explicit LoadConverter(MLIRContext *context,
                         bool optimizeDynamicOffset = false,
                         bool enableMaskFallbackConversion = false)
      : OpRewritePattern<triton::LoadOp>(context),
        optimizeDynamicOffset(optimizeDynamicOffset),
        enableMaskFallbackConversion(enableMaskFallbackConversion){};

  using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::LoadOp op,
                                PatternRewriter &rewriter) const override;

private:
  bool optimizeDynamicOffset;
  bool enableMaskFallbackConversion;
};

class StoreConverter : public OpRewritePattern<triton::StoreOp> {
public:
  explicit StoreConverter(MLIRContext *context,
                          bool optimizeDynamicOffset = false,
                          bool enableMaskFallbackConversion = false)
      : OpRewritePattern<triton::StoreOp>(context),
        optimizeDynamicOffset(optimizeDynamicOffset),
        enableMaskFallbackConversion(enableMaskFallbackConversion){};

  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::StoreOp op,
                                PatternRewriter &rewriter) const override;

private:
  bool optimizeDynamicOffset;
  bool enableMaskFallbackConversion;
};

class MemOpTransformer {
public:
  TritonToStructured::PtrState ptrState;
  TritonToStructured::MaskState maskState;

  enum class MemType { load, store, deafaultType };

  bool optimizeDynamicOffset;

  MemType currentType = MemType::deafaultType;

  MemOpTransformer(MemType memType, bool optimizeDynamicOffset = false)
      : currentType(memType), optimizeDynamicOffset(optimizeDynamicOffset) {}

  Value materializeImplicitBroadcast(Value srcTensor, const Location loc,
                                     PatternRewriter &rewriter);

  Value materializeImplicitReshape(Value srcTensor, const Location loc,
                                   PatternRewriter &rewriter);

  Value materializeImplicitSelect(Value srcTensor, Value mask, Value other,
                                  const Location loc,
                                  PatternRewriter &rewriter);

  Value materializeImplicitPermute(Value srcTensor, const Location loc,
                                   PatternRewriter &rewriter);

  Value createNewPtr(Value oldPtr, const Location loc,
                     PatternRewriter &rewriter);

  Value createNewMask(Value oldPtr, const Location loc,
                      PatternRewriter &rewriter);

  Value createNewOther(Value oldOther, const Location loc,
                       PatternRewriter &rewriter);

  bool applyPermuteOnMask();
};

// Create local lock var
hivm::CreateSyncBlockLockOp createSyncBlockLockVar(OpBuilder &builder,
                                                   Location loc);

} // namespace MemOpConverter

#endif