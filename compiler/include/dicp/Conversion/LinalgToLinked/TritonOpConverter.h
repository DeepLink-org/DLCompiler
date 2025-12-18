#ifndef TRITON_DLC_TRITONOPCONVERTER_H
#define TRITON_DLC_TRITONOPCONVERTER_H

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace mlir::dicp::linked {

template <typename OpTy>
class ReductionOpBaseConverter : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto sourceType =
        cast<RankedTensorType>(adaptor.getOperands().front().getType());
    assert(sourceType.hasRank() && "Expected input is ranked");

    int64_t axis = op.getAxis();
    assert(axis >= 0 && axis < sourceType.getRank() &&
           "Expected reduction axis is within operand's rank");

    auto reductionOps = this->getRedOps(op);
    if (reductionOps.size() == 1) {
      return this->convertToTargetOp(op, adaptor, rewriter);
    }
    return this->convertToTargetOpExtended(op, adaptor, rewriter);
  }

protected:
  llvm::SmallVector<Operation *> getRedOps(OpTy redOp) const {
    auto redBody = redOp.getBody();
    return llvm::map_to_vector(redBody->without_terminator(),
                               [](Operation &op) { return &op; });
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
            .Case([&](arith::MulFOp) {
              return rewriter.getFloatAttr(constantType, 1.f);
            })
            .template Case<arith::MaximumFOp, arith::MaxNumFOp>([&](auto) {
              return rewriter.getFloatAttr(
                  constantType, -std::numeric_limits<float>::infinity());
            })
            .template Case<arith::MinimumFOp, arith::MinNumFOp>([&](auto) {
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
            .Case([&](arith::MaxUIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Case([&](arith::OrIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Case([&](arith::AndIOp) {
              return rewriter.getIntegerAttr(constantType, 1);
            })
            .Case([&](arith::XOrIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Default([](Operation *op) {
              op->dump();
              llvm_unreachable("Reduction op not supported yet");
              return nullptr;
            });

    return rewriter.create<arith::ConstantOp>(redOp->getLoc(), constantType,
                                              attr);
  }

  bool requiresF32Conversion(const Type elemType, Operation *redOp) const {
    return isa<FloatType>(elemType) &&
           elemType.getIntOrFloatBitWidth() <
               Float32Type::get(elemType.getContext())
                   .getIntOrFloatBitWidth() &&
           (isa<arith::AddFOp>(redOp) || isa<arith::MulFOp>(redOp));
  }

  Value getRedElement(Value lhs, Value rhs, const Location loc,
                      Operation *redOp, OpBuilder &b,
                      const bool convertLhsToF32Precision) const {
    return llvm::TypeSwitch<Operation *, Value>(redOp)
        .template Case<arith::AddFOp, arith::MulFOp>([&](auto redOp) {
          if (convertLhsToF32Precision) {
            lhs = b.create<arith::ExtFOp>(loc, Float32Type::get(b.getContext()),
                                          lhs);
          }
          return b.create<decltype(redOp)>(loc, lhs, rhs);
        })
        .template Case<arith::AddIOp, arith::MaximumFOp, arith::MaxNumFOp,
                       arith::MinimumFOp, arith::MinNumFOp, arith::MinSIOp,
                       arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp,
                       arith::AndIOp, arith::OrIOp, arith::XOrIOp>(
            [&](auto redOp) {
              return b.create<decltype(redOp)>(loc, lhs, rhs);
            })
        .Default([](Operation *op) {
          op->dump();
          llvm_unreachable("Reduction op not yet supported");
          return nullptr;
        });
  }

  virtual bool isReductionOpSupported(Operation *redOp) const = 0;

  virtual LogicalResult
  convertToTargetOp(OpTy op, typename OpTy::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const = 0;

  virtual LogicalResult
  convertToTargetOpExtended(OpTy op, typename OpTy::Adaptor adaptor,
                            ConversionPatternRewriter &rewriter) const = 0;
};

class ReduceConverter : public ReductionOpBaseConverter<triton::ReduceOp> {
public:
  explicit ReduceConverter(MLIRContext *context)
      : ReductionOpBaseConverter<triton::ReduceOp>(context) {}

  using ReductionOpBaseConverter<triton::ReduceOp>::ReductionOpBaseConverter;

protected:
  bool isReductionOpSupported(Operation *redOp) const override;

  LogicalResult
  convertToTargetOp(triton::ReduceOp op,
                    typename triton::ReduceOp::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override;

  LogicalResult
  convertToTargetOpExtended(triton::ReduceOp op,
                            typename triton::ReduceOp::Adaptor adaptor,
                            ConversionPatternRewriter &rewriter) const override;
};

class ScanConverter : public ReductionOpBaseConverter<triton::ScanOp> {
public:
  explicit ScanConverter(MLIRContext *context)
      : ReductionOpBaseConverter<triton::ScanOp>(context) {}

  using ReductionOpBaseConverter<triton::ScanOp>::ReductionOpBaseConverter;

protected:
  bool isReductionOpSupported(Operation *redOp) const override;

  LogicalResult
  convertToTargetOp(triton::ScanOp op, typename triton::ScanOp::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override;

  LogicalResult
  convertToTargetOpExtended(triton::ScanOp op,
                            typename triton::ScanOp::Adaptor adaptor,
                            ConversionPatternRewriter &rewriter) const override;
};

class DeviceAssertConverter : public OpConversionPattern<triton::AssertOp> {
  using OpConversionPattern<triton::AssertOp>::OpConversionPattern;

private:
  static constexpr llvm::StringRef printFuncNameBase = "triton_assert";
  static constexpr llvm::StringRef msgAttrName = "msg";

public:
  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class DevicePrintConverter : public OpConversionPattern<triton::PrintOp> {
  using OpConversionPattern<triton::PrintOp>::OpConversionPattern;

private:
  static constexpr llvm::StringRef printFuncNameBase = "triton_print";
  static constexpr llvm::StringRef prefixAttrName = "prefix";
  static constexpr llvm::StringRef hexAttrName = "hex";

public:
  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace mlir::dicp::linked

#endif
