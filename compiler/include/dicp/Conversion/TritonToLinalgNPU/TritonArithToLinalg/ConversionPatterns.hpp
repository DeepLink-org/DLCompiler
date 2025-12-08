#ifndef TRITON_NPU_CONVERSION_PATTERNS
#define TRITON_NPU_CONVERSION_PATTERNS

#include "dicp/Utils/Utils.h"

#include "triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>
#include <optional>
#include <type_traits>

using namespace mlir;
using namespace triton;
using namespace mlir::dicp::linked;

namespace {

enum class InputPrecision : uint32_t {
  TF32 = 0,
  TF32x3 = 1,
  IEEE = 2,
  HF32 = 3,
};
static ::llvm::StringRef stringifyInputPrecision(InputPrecision val) {
  switch (val) {
  case InputPrecision::TF32:
    return "tf32";
  case InputPrecision::TF32x3:
    return "tf32x3";
  case InputPrecision::IEEE:
    return "ieee";
  case InputPrecision::HF32:
    return "hf32";
  }
  return "";
}

struct BroadcastNPUConverter : public OpConversionPattern<triton::BroadcastOp> {
private:
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;

  SmallVector<int64_t> getBroadcastDims(RankedTensorType src,
                                        RankedTensorType dst) const {
    SmallVector<int64_t> broadcastDims;
    auto srcShape = src.getShape();
    auto dstShape = dst.getShape();

    for (size_t i = 0; i < srcShape.size(); i++) {
      if (dstShape[i] != srcShape[i]) {
        assert(srcShape[i] == 1);
        broadcastDims.push_back(i);
      }
    }
    assert(!broadcastDims.empty() && "cannot identify broadcast dimension");
    return broadcastDims;
  }

  // Broadcasts input tensor based on TosaToLinalg's broadcastToShape
  AffineMap getBroadcastAffineMap(MLIRContext *context,
                                  ArrayRef<int64_t> inputShape,
                                  ArrayRef<int64_t> broadcastToShape) const {

    assert(broadcastToShape.size() >= inputShape.size());

    // Create affine map and shapes for tensor initialization.
    SmallVector<AffineExpr> outExpr;

    size_t diff = broadcastToShape.size() - inputShape.size();
    for (size_t i = 0; i < broadcastToShape.size(); i++) {
      if (i < diff) {
        continue;
      }
      size_t j = i - diff;
      if (inputShape[j] == 1) {
        // Broadcast singleton dimension
        outExpr.push_back(mlir::getAffineConstantExpr(0, context));
        continue;
      }
      // Non-broadcast case
      outExpr.push_back(mlir::getAffineDimExpr(i, context));
    }
    return AffineMap::get(broadcastToShape.size(), 0, outExpr, context);
  }

public:
  // Dimensions of collapesd tensor is all unbroadcast dims
  SmallVector<int64_t> getUnbroadcastDims(RankedTensorType src,
                                          RankedTensorType dst) const {
    SmallVector<int64_t> unbroadcastDims;
    auto srcShape = src.getShape();
    auto dstShape = dst.getShape();

    for (size_t i = 0; i < srcShape.size(); ++i) {
      if (dstShape[i] == srcShape[i]) {
        unbroadcastDims.emplace_back(srcShape[i]);
      }
    }
    return unbroadcastDims;
  }
  // Here convert tt.broadcast to linalg.broadcast
  //
  // before
  // %out = tt.broadcast %in : tensor<1x4x8xf32> -> tensor<128x4x8xf32>
  //
  // after
  // %collpased = tensor.collapse_shape %in [[0, 1], [2]] :
  //                                    tensor<1x4x8xf32> into tensor<4x8xf32>
  // %out = linalg.broadcast ins(%collpased : tensor<4x8xf32>)
  //                         outs(%empty : tensor<128x4x8xf32>) dimensions = [0]
  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumResults() == 1 && "BroadcastOp assumes single result");

    RankedTensorType sourceType =
        cast<RankedTensorType>(adaptor.getSrc().getType());
    RankedTensorType resultType = cast<RankedTensorType>(op.getType());
    auto elementType = resultType.getElementType();
    size_t resultRank = resultType.getRank();
    auto loc = op.getLoc();

    auto initEmpty = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), elementType);

    SmallVector<int64_t> broadcastDims =
        getBroadcastDims(sourceType, resultType);
    SmallVector<int64_t> unbroadcastDims =
        getUnbroadcastDims(sourceType, resultType);

    SmallVector<ReassociationIndices> collapseReassociationIndices;
    auto collapseReassociationIndicesOptional =
        getReassociationIndicesForCollapse(sourceType.getShape(),
                                           unbroadcastDims);
    if (!collapseReassociationIndicesOptional.has_value()) {
      return rewriter.notifyMatchFailure(
          op, "Failure with getReassociationIndicesForCollapse call");
    }
    collapseReassociationIndices = collapseReassociationIndicesOptional.value();

    RankedTensorType collapseResultType =
        RankedTensorType::get(unbroadcastDims, sourceType.getElementType());

    auto collpasedOp = rewriter.create<tensor::CollapseShapeOp>(
        loc, collapseResultType, adaptor.getSrc(),
        collapseReassociationIndices);

    auto broadcastOp = rewriter.create<linalg::BroadcastOp>(
        loc, collpasedOp, initEmpty,
        rewriter.getDenseI64ArrayAttr(broadcastDims));

    rewriter.replaceOp(op, broadcastOp.getResults());
    return success();
  }
};

struct MatmulNPUConverter : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  // true means tensor elements are zeros
  // false means not zero or it cannot be determined
  bool isZeroTensor(Value &v, bool integers) const {
    if (auto splatOp = v.getDefiningOp<triton::SplatOp>()) {
      if (auto constOp = splatOp.getSrc().getDefiningOp<arith::ConstantOp>()) {
        if (auto val = dyn_cast<FloatAttr>(constOp.getValue())) {
          return val.getValueAsDouble() == 0.;
        }
        if (auto val = dyn_cast<IntegerAttr>(constOp.getValue())) {
          return val.getValue() == 0;
        }
      }
      return false;
    }

    if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
        if (denseAttr.isSplat()) {
          if (integers)
            return denseAttr.getSplatValue<APInt>().isZero();
          return denseAttr.getSplatValue<APFloat>().isZero();
        }
      }
    }

    return false;
  }

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opa = adaptor.getA();
    auto opb = adaptor.getB();
    auto opc = adaptor.getC();
    auto dstType = cast<RankedTensorType>(op.getType());
    auto inputPrec = op.getInputPrecision();

    if (dstType.getRank() == 2) {
      auto matmulOp = rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
          op, ValueRange{opa, opb}, ValueRange{opc});
      matmulOp->setAttr(
          "input_precison",
          rewriter.getStringAttr(stringifyInputPrecision(inputPrec)));
    } else if (dstType.getRank() == 3) {
      auto matmulOp = rewriter.replaceOpWithNewOp<linalg::BatchMatmulOp>(
          op, ValueRange{opa, opb}, ValueRange{opc});
      matmulOp->setAttr(
          "input_precison",
          rewriter.getStringAttr(stringifyInputPrecision(inputPrec)));
    } else {
      llvm_unreachable("Datatype of DotOp operands could only be 2D or 3D");
    }
    return success();
  }
};

struct ReduceNPUConverter : public OpConversionPattern<triton::ReduceOp> {

  ReduceNPUConverter(MLIRContext *context, bool transposeToRank0 = false,
                     PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit),
        transposeToRank0(transposeToRank0) {}

private:
  bool transposeToRank0;

  llvm::SmallVector<Operation *> getRedOps(triton::ReduceOp redOp) const {
    auto reduceBlock = redOp.getBody();
    return llvm::map_to_vector(reduceBlock->without_terminator(),
                               [](Operation &op) { return &op; });
  }

  bool isReductionOpSupported(Operation *redOp) const {
    return isa<arith::AddFOp, arith::AddIOp, arith::AndIOp, arith::MaximumFOp,
               arith::MulFOp, arith::MulIOp, arith::MaxNumFOp,
               arith::MinimumFOp, arith::MinNumFOp, arith::MinSIOp,
               arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp, arith::OrIOp,
               arith::XOrIOp>(redOp);
  }

  arith::ConstantOp getRedBaseConstOp(ConversionPatternRewriter &rewriter,
                                      Operation *redOp,
                                      Type constantType) const {
    const int64_t bitWidth = constantType.getIntOrFloatBitWidth();

    auto attr =
        llvm::TypeSwitch<Operation *, TypedAttr>(redOp)
            .Case([&](arith::AddFOp) {
              return rewriter.getFloatAttr(constantType, 0.f);
            })
            .Case([&](arith::AddIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Case<arith::MaximumFOp, arith::MaxNumFOp>([&](auto) {
              return rewriter.getFloatAttr(
                  constantType, -std::numeric_limits<float>::infinity());
            })
            .Case<arith::MinimumFOp, arith::MinNumFOp>([&](auto) {
              return rewriter.getFloatAttr(
                  constantType, std::numeric_limits<float>::infinity());
            })
            .Case([&](arith::MinSIOp) {
              return rewriter.getIntegerAttr(constantType,
                                             llvm::maxIntN(bitWidth));
            })
            .Case([&](arith::MinUIOp) {
              return rewriter.getIntegerAttr(constantType,
                                             llvm::maxUIntN(bitWidth));
            })
            .Case([&](arith::MaxSIOp) {
              return rewriter.getIntegerAttr(constantType,
                                             llvm::minIntN(bitWidth));
            })
            .Case<arith::MaxUIOp, arith::XOrIOp>(
                [&](auto) { return rewriter.getIntegerAttr(constantType, 0); })
            .Case([&](arith::MulFOp) {
              return rewriter.getFloatAttr(constantType, 1.f);
            })
            .Case<arith::MulIOp, arith::AndIOp>(
                [&](auto) { return rewriter.getIntegerAttr(constantType, 1); })
            .Case([&](arith::OrIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Default([](Operation *op) {
              op->dump();
              llvm_unreachable("Reduction op not yet supported");
              return nullptr;
            });

    return rewriter.create<arith::ConstantOp>(redOp->getLoc(), constantType,
                                              attr);
  }

  bool requiresF32Conversion(const Type elemType, Operation *redOp) const {
    unsigned width =
        cast<FloatType>(Float32Type::get(elemType.getContext())).getWidth();
    return isa<FloatType>(elemType) &&
           elemType.getIntOrFloatBitWidth() < width &&
           isa<arith::AddFOp>(redOp);
  }

  Value getRedElement(Value lhs, Value rhs, const Location loc,
                      Operation *redOp, OpBuilder &b,
                      const bool convertLhsToF32Precision) const {
    return llvm::TypeSwitch<Operation *, Value>(redOp)
        .Case<arith::AddFOp, arith::MulFOp>([&](auto redOp) {
          if (convertLhsToF32Precision) {
            lhs = b.create<arith::ExtFOp>(loc, Float32Type::get(b.getContext()),
                                          lhs);
          }
          return b.create<decltype(redOp)>(loc, lhs, rhs);
        })
        .Case<arith::AddIOp, arith::AndIOp, arith::XOrIOp, arith::MaximumFOp,
              arith::MaxNumFOp, arith::MulIOp, arith::MinimumFOp,
              arith::MinNumFOp, arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp,
              arith::MaxUIOp, arith::OrIOp>([&](auto redOp) {
          return b.create<decltype(redOp)>(loc, lhs, rhs);
        })
        .Default([](Operation *op) {
          op->dump();
          llvm_unreachable("Reduction op not yet supported");
          return nullptr;
        });
  }

  LogicalResult
  convertToLinalgReduce(triton::ReduceOp op,
                        typename triton::ReduceOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    auto source = adaptor.getOperands().front();
    auto sourceType = cast<RankedTensorType>(source.getType());
    auto elemType = sourceType.getElementType();
    auto resType = op.getResult().front().getType();
    auto loc = op.getLoc();
    auto reductionOps = getRedOps(op);

    // Reduction of arbitrary operations isn't supported because using the first
    // element across the reduction dimension requires us to iterate over a
    // subview that skips over each first element.
    if (reductionOps.size() != 1 ||
        !isReductionOpSupported(reductionOps.front())) {
      return rewriter.notifyMatchFailure(
          op, "Only support lowering reduction with body "
              "containing 1 max(i/f), addf, ori, or mulf.");
    }

    auto rop = reductionOps.front();
    auto axis = op.getAxis();
    auto rank = sourceType.getRank();
    auto isVectorReduce = (rank == 1);

    // For now we are transposing reductions from Triton Shared as an
    // optimization. This should not be the job of Triton Shared so moving
    // forward this will be removed. Doing the transpose here lacks a wider
    // scope of analysis that might indicate that the transpose to a given axis
    // is not optimal.
    if (transposeToRank0) {
      // if it is not a vector reduce, we can transpose the source
      // so that the reduction axis is the first dimension.
      if (!isVectorReduce && axis != 0) {
        SmallVector<int32_t> order;
        order.reserve(rank);
        order.push_back(axis);
        for (int i = 0; i < rank; ++i) {
          if (i != axis) {
            order.push_back(i);
          }
        }
        source = getTransposedValue(source, op.getLoc(), rewriter, order);
        axis = 0;
      }
    }

    bool convertToF32Precision = requiresF32Conversion(resType, rop);

    auto constantType = convertToF32Precision
                            ? Float32Type::get(rewriter.getContext())
                            : elemType;

    auto accBaseConstOp = getRedBaseConstOp(rewriter, rop, constantType);
    Value initTensor;

    if (isVectorReduce) {
      // The affine vectorizer cannot vectorize affine loops generated from
      // linalg.reduce for the vector reduce case, so we must rewrite the
      // linalg.reduce to affine loops manually. Here we lower to AllocTensor
      // directly instead of EmptyOp so that the subsequent pass can recognize
      // the patterns (EmptyOp is susceptible to being CSE'd away, making it
      // harder to match the patterns correctly).
      initTensor = rewriter.create<bufferization::AllocTensorOp>(
          loc, RankedTensorType::get({}, constantType), ValueRange{});
      initTensor = rewriter.create<tensor::InsertOp>(loc, accBaseConstOp,
                                                     initTensor, ValueRange{});
    } else {
      Value init = rewriter.create<tensor::EmptyOp>(
          loc, cast<RankedTensorType>(resType).getShape(), constantType);
      initTensor = rewriter
                       .create<linalg::FillOp>(loc, ValueRange{accBaseConstOp},
                                               ValueRange{init})
                       .result();
    }

    Value finalResult =
        rewriter
            .create<linalg::ReduceOp>(
                loc, ValueRange{source}, ValueRange{initTensor},
                SmallVector<int64_t>{axis},
                [&](OpBuilder &opBuilder, Location loc, ValueRange inputs) {
                  assert(inputs.size() == 2);
                  Value result =
                      getRedElement(inputs[0], inputs[1], loc, rop, opBuilder,
                                    convertToF32Precision);
                  opBuilder.create<linalg::YieldOp>(loc, result);
                })
            .getResult(0);

    if (isVectorReduce) {
      finalResult =
          rewriter.create<tensor::ExtractOp>(loc, constantType, finalResult);
    }

    if (convertToF32Precision) {
      finalResult = rewriter.create<arith::TruncFOp>(loc, resType, finalResult);
    }

    rewriter.replaceOp(op, finalResult);
    return success();
  }

public:
  LogicalResult
  matchAndRewrite(triton::ReduceOp op,
                  typename triton::ReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType =
        cast<RankedTensorType>(adaptor.getOperands().front().getType());
    assert(sourceType.hasRank() && "Expected input is "
                                   "ranked");

    int64_t axis = op.getAxis();
    assert(axis >= 0 && axis < sourceType.getRank() &&
           "Expected reduction "
           "axis is within "
           "operand's rank");

    return convertToLinalgReduce(op, adaptor, rewriter);
  }
};

class BitcastNPUConverter : public OpRewritePattern<triton::BitcastOp> {
public:
  using OpRewritePattern<triton::BitcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::BitcastOp op,
                                PatternRewriter &rewriter) const {
    if (op->hasAttr("Input_Arg_i1_Bitcast_To_i8")) {
      return failure();
    }

    Value result;
    if (auto resPointerType = dyn_cast<triton::PointerType>(op.getType())) {
      // TODO: use typeconverter
      auto srcPointerType = cast<triton::PointerType>(op.getSrc().getType());
      auto resType = MemRefType::get({ShapedType::kDynamic},
                                     resPointerType.getPointeeType());
      // Handling special case
      // %0 = tt.bitcast %arg0 {MixUse} : !tt.ptr<i1> -> !tt.ptr<i8>
      if (isa<BlockArgument>(op.getSrc()) &&
          srcPointerType.getPointeeType() == rewriter.getIntegerType(1) &&
          resPointerType.getPointeeType() == rewriter.getIntegerType(8)) {
        rewriter.modifyOpInPlace(op, [&]() {
          op->setAttr("Input_Arg_i1_Bitcast_To_i8", rewriter.getUnitAttr());
        });
        return success();
      }
      result =
          rewriter.create<arith::BitcastOp>(op.getLoc(), resType, op.getSrc());
    } else {
      result = rewriter.create<arith::BitcastOp>(op.getLoc(), op.getType(),
                                                 op.getSrc());
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

#endif
