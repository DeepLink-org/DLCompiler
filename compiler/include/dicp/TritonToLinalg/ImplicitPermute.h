

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

namespace ImplicitPermute {

using namespace mlir;
using namespace triton;

class LoadConverter : public OpRewritePattern<triton::LoadOp> {
public:
  explicit LoadConverter(MLIRContext *context)
      : OpRewritePattern<triton::LoadOp>(context){};

  using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::LoadOp op,
                                PatternRewriter &rewriter) const override;
};

class StoreConverter : public OpRewritePattern<triton::StoreOp> {
public:
  explicit StoreConverter(MLIRContext *context)
      : OpRewritePattern<triton::StoreOp>(context){};

  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::StoreOp op,
                                PatternRewriter &rewriter) const override;
};

class AtomicRMWConverter : public OpRewritePattern<triton::AtomicRMWOp> {
public:
  explicit AtomicRMWConverter(MLIRContext *context)
      : OpRewritePattern<triton::AtomicRMWOp>(context){};

  using OpRewritePattern<triton::AtomicRMWOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override;
};

class AtomicCASConverter : public OpRewritePattern<triton::AtomicCASOp> {
public:
  explicit AtomicCASConverter(MLIRContext *context)
      : OpRewritePattern<triton::AtomicCASOp>(context){};

  using OpRewritePattern<triton::AtomicCASOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::AtomicCASOp op,
                                PatternRewriter &rewriter) const override;
};

class MemOpTransformer {
public:
  TritonToStructured::PtrState ptrState;
  TritonToStructured::MaskState maskState;

  enum class MemType { load, store, deafaultType };

  MemType currentType = MemType::deafaultType;

  MemOpTransformer(MemType memType) : currentType(memType) {}

  Value materializeImplicitPermute(Value srcTensor, const Location loc,
                                   PatternRewriter &rewriter);

  Value createNewAddPtr(Value oldPtr, const Location loc,
                        PatternRewriter &rewriter);

  Value createNewAdvancePtr(Value oldPtr, const Location loc,
                            PatternRewriter &rewriter);

  Value createNewTensorPtr(Value oldPtr, const Location loc,
                           PatternRewriter &rewriter);

  Value createNewMask(Value oldPtr, const Location loc,
                      PatternRewriter &rewriter);

  Value createNewOther(Value oldOther, const Location loc,
                       PatternRewriter &rewriter);

  SmallVector<int32_t>
  getBoundaryCheck(ArrayRef<int32_t> oldBoundaryCheck) const;

  bool applyPermuteOnMask();
};

} // namespace ImplicitPermute
