#ifndef TRITON_DLC_LOADSTORECONVERTER_H
#define TRITON_DLC_LOADSTORECONVERTER_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace triton;

namespace mlir::dicp::trtion_ext {

// tempate class's impl must in header file
template <typename OpTy>
class LoadStoreCanonicalizer : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Value ptrVal = op.getPtr();
    Type ptrTy = ptrVal.getType();
    auto ptrDefOp = ptrVal.getDefiningOp();

    bool shouldAddZeros = false;
    if (!isa<BlockArgument>(ptrVal))
      shouldAddZeros = !isTensorPointerType(ptrTy) &&
                       !isa_and_nonnull<triton::AddPtrOp>(ptrDefOp);
    else if (auto ptrType = dyn_cast<triton::PointerType>(ptrTy))
      shouldAddZeros = ptrType.getPointeeType().isIntOrIndexOrFloat();

    if (shouldAddZeros) {
      if (isa_and_nonnull<triton::BitcastOp>(ptrDefOp)) {
        auto castOp = cast<triton::BitcastOp>(ptrDefOp);
        auto castSrc = castOp.getSrc();
        if (!isa<BlockArgument>(castSrc)) {
          auto castSrcDefOp = castSrc.getDefiningOp();
          if (isa<triton::AddPtrOp>(castSrcDefOp)) {
            return rewriter.notifyMatchFailure(
                op, "BitcastCanonicalizer handles addptr->bitcast->load!");
          }
        }
      }

      Type zeroTy = getI32SameShape(ptrTy);
      Value zeroVal =
          createScalarOrSplatConstant(rewriter, op.getLoc(), zeroTy, 0);
      Value addptrVal = rewriter.create<triton::AddPtrOp>(op.getLoc(), ptrTy,
                                                          ptrVal, zeroVal);
      rewriter.modifyOpInPlace(
          op, [&]() { op->replaceUsesOfWith(ptrVal, addptrVal); });
      return success();
    }
    return failure();
  }
};

class ScalarStoreCanonicalizer : public OpRewritePattern<triton::StoreOp> {
public:
  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::StoreOp op,
                                PatternRewriter &rewriter) const override;
};

class ScalarAtomicRMWCanonicalizer
    : public OpRewritePattern<triton::AtomicRMWOp> {
  using OpRewritePattern<triton::AtomicRMWOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override;
};

class ScalarAtomicCASCanonicalizer
    : public OpRewritePattern<triton::AtomicCASOp> {
  using OpRewritePattern<triton::AtomicCASOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::AtomicCASOp op,
                                PatternRewriter &rewriter) const override;
};

class AtomicCASConverter : public OpConversionPattern<triton::AtomicCASOp> {
public:
  explicit AtomicCASConverter(MLIRContext *context)
      : OpConversionPattern<triton::AtomicCASOp>(context) {}
  using OpConversionPattern<triton::AtomicCASOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class AtomicRMWConverter : public OpConversionPattern<triton::AtomicRMWOp> {
private:
  Value createAtomicBinaryOps(OpBuilder &builder, Location loc,
                              triton::AtomicRMWOp op, Type elementType,
                              Value lhs, Value rhs) const {
    auto rmwOp = op.getAtomicRmwOp();

    // it has been confirmed in AtomicRMWConverter::matchAndRewrite
    // that the ptr of op is of MemRefType
    Value binaryOp;
    if (rmwOp == triton::RMWOp::FADD) {
      binaryOp = builder.create<arith::AddFOp>(loc, lhs, rhs);
    } else if (rmwOp == triton::RMWOp::ADD) {
      binaryOp = builder.create<arith::AddIOp>(loc, lhs, rhs);
    } else if (rmwOp == triton::RMWOp::XOR) {
      binaryOp = builder.create<arith::XOrIOp>(loc, lhs, rhs);
    } else if (rmwOp == triton::RMWOp::OR) {
      binaryOp = builder.create<arith::OrIOp>(loc, lhs, rhs);
    } else if (rmwOp == triton::RMWOp::AND) {
      binaryOp = builder.create<arith::AndIOp>(loc, lhs, rhs);
    } else if (rmwOp == triton::RMWOp::MAX) {
      // Max/Min only support f32/i32 for now
      // Other type is not supported because of semantic.py
      if (isa<FloatType>(elementType)) {
        binaryOp = builder.create<arith::MaxNumFOp>(loc, lhs, rhs);
      } else {
        binaryOp = builder.create<arith::MaxSIOp>(loc, lhs, rhs);
      }
    } else if (rmwOp == triton::RMWOp::MIN) {
      if (isa<FloatType>(elementType)) {
        binaryOp = builder.create<arith::MinNumFOp>(loc, lhs, rhs);
      } else {
        binaryOp = builder.create<arith::MinSIOp>(loc, lhs, rhs);
      }
    } else if (rmwOp == triton::RMWOp::XCHG) {
      binaryOp = rhs;
    } else {
      op.emitOpError("unsupported atomic RMW operation: ");
      llvm_unreachable(
          "Not implemented. Support fadd, add, max, min for now !");
    }
    return binaryOp;
  }

  // used when handling scalar
  // to verify whether we need to handle this scalar
  bool isConstantMaskTrue(Value mask) const {
    if (auto denseAttr =
            mask.getDefiningOp()->getAttrOfType<DenseElementsAttr>("value")) {
      auto eleType = denseAttr.getType().getElementType();
      if (isa<IntegerType>(eleType) &&
          cast<IntegerType>(eleType).getWidth() == 1) {
        auto values = denseAttr.getValues<bool>();
        return values[0];
      }
    }
    return false;
  }

  DenseSet<triton::RMWOp> softwareAtomicKinds = {
      triton::RMWOp::AND, triton::RMWOp::OR, triton::RMWOp::XOR};

public:
  explicit AtomicRMWConverter(MLIRContext *context);
  using OpConversionPattern<triton::AtomicRMWOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class AtomicMaxMinCanonicalizer : public OpRewritePattern<triton::AtomicRMWOp> {
  using OpRewritePattern<triton::AtomicRMWOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override;
};

class SelectCanonicalizer : public OpRewritePattern<arith::SelectOp> {
public:
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override;
};

/*
 * Move tt.bitcast to a previous location if tt.bitcast is not directly applied
 * on function arguments
 */
class BitcastCanonicalizer : public OpRewritePattern<triton::BitcastOp> {
public:
  using OpRewritePattern<triton::BitcastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::BitcastOp bitcastOp,
                                PatternRewriter &rewriter) const override;
};

template <typename MathOp>
class ScalarMathCanonicalizer : public OpRewritePattern<MathOp> {
public:
  using OpRewritePattern<MathOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MathOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "ScalarMathCanonicalizer expects single scalar output.");
    }
    if (!op->getResult(0).getType().isIntOrIndexOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "ScalarMathCanonicalizer handles scalar load scene.");
    }
    if (auto linalgOp = op->template getParentOfType<triton::ReduceOp>()) {
      return rewriter.notifyMatchFailure(
          op, "ScalarMathCanonicalizer handles op not within tt.reduce.");
    }
    if (auto linalgOp = op->template getParentOfType<triton::ScanOp>()) {
      return rewriter.notifyMatchFailure(
          op, "ScalarMathCanonicalizer handles op not within tt.scan.");
    }
    auto loc = op.getLoc();
    llvm::SmallVector<Value> inputs;
    for (auto input : op->getOperands()) {
      auto blkTy = RankedTensorType::get({(int64_t)1}, input.getType());
      auto inputSplat = rewriter.create<triton::SplatOp>(loc, blkTy, input);
      inputs.push_back(inputSplat.getResult());
    }
    auto blkOp = rewriter.create<MathOp>(loc, inputs);
    Value offset =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    auto extractOp =
        rewriter.create<tensor::ExtractOp>(loc, blkOp.getResult(), offset);
    rewriter.replaceOp(op, extractOp);
    return success();
  }
};

/*
 * Rewrite tt.make_tensor_ptr with non-contiguous order to
 * tt.make_tensor_ptr + tt.load + tt.trans.
 */
class MakeTensorPtrCanonicalizer
    : public OpRewritePattern<triton::MakeTensorPtrOp> {
public:
  using OpRewritePattern<triton::MakeTensorPtrOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::MakeTensorPtrOp op,
                                PatternRewriter &rewriter) const override;
};

class ReduceSingleCanonicalizer : public OpRewritePattern<triton::ReduceOp> {
public:
  using OpRewritePattern<triton::ReduceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override;
};

/**
 * @brief Rewrites arith.remf: remf(a, b) = a - b * floor(a / b)
 */
class RemfToBasicArithmetic final : public OpRewritePattern<arith::RemFOp> {
public:
  using OpRewritePattern<arith::RemFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::RemFOp op,
                                PatternRewriter &rewriter) const override;
};

/**
 * @brief Rewrites arith.remsi: remsi(a, b) = a - b * (a / b)
 */
class RemSIToBasicArithmetic final : public OpRewritePattern<arith::RemSIOp> {
public:
  using OpRewritePattern<arith::RemSIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::RemSIOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::dicp::trtion_ext
#endif
