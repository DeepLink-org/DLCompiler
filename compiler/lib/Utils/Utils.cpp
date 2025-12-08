#include "dicp/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <unordered_set>

#define DEBUG_TYPE "Dicp-Utils"
using namespace mlir;

namespace mlir::dicp {

llvm::StringRef getBackend(ModuleOp module) {
  if (!module)
    return llvm::StringRef();

  if (auto strAttr = module->getAttrOfType<StringAttr>("dicp.backend"))
    return strAttr.getValue(); // StringRef，适配 StringSwitch

  return llvm::StringRef(); // 空字符串
}

bool isAscendBackend(ModuleOp module) { return getBackend(module) == "ascend"; }

static Value createConstIndexValueOp(const Location &loc, OpBuilder &b,
                                     int64_t value) {
  return b.create<arith::ConstantOp>(loc, b.getIndexAttr(value)).getResult();
}

static std::optional<int64_t> getConstantOfAttr(const OpFoldResult &arg) {
  if (isa<Attribute>(arg)) {
    return getConstantIntValue(arg);
  }

  return std::nullopt;
}

std::optional<int64_t>
getLastStrideOfReinterpretCastOp(memref::ReinterpretCastOp op) {
  SmallVector<OpFoldResult> mixedStrides = op.getMixedStrides();
  if (mixedStrides.empty()) {
    op->emitError("ReinterpretCastOp has no strides");
    return std::nullopt;
  }

  OpFoldResult lastStride = mixedStrides.back();
  if (auto attr = dyn_cast<Attribute>(lastStride)) {
    return getConstantOfAttr(lastStride);
  } else if (auto value = dyn_cast<Value>(lastStride)) {
    auto defOp = value.getDefiningOp();
    if (auto constIndexOp = dyn_cast<arith::ConstantIndexOp>(defOp)) {
      int64_t constValue = constIndexOp.value();
      return constValue;
    } else if (auto constIntOp = dyn_cast<arith::ConstantIntOp>(defOp)) {
      int64_t constValue = constIntOp.value();
      return constValue;
    }
  }
  return std::nullopt;
}

Value getTransposedValue(Value source, const Location loc,
                         ConversionPatternRewriter &rewriter,
                         llvm::ArrayRef<int> order) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto sourceRank = sourceType.getRank();

  SmallVector<int64_t> perm(order);
  SmallVector<int64_t> originalShape(sourceType.getShape());
  SmallVector<int64_t> transposedShape(sourceRank);
  for (size_t i = 0; i < sourceRank; i++) {
    transposedShape[i] = originalShape[perm[i]];
  }

  Value transposeInit = rewriter.create<tensor::EmptyOp>(
      loc, transposedShape, sourceType.getElementType());

  Value transpose =
      rewriter.create<linalg::TransposeOp>(loc, source, transposeInit, perm)
          .getResults()[0];

  return transpose;
}

SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n) {
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}

Value getScalarValue(Value operand, Location loc,
                     ConversionPatternRewriter &rewriter) {
  SmallVector<Operation *> ops;
  auto reconstructScalarValue = [&](Value src) {
    for (auto op = ops.rbegin(); op != ops.rend(); ++op) {
      src = mlir::TypeSwitch<Operation *, Value>(*op)
                .Case<arith::SIToFPOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return rewriter.create<arith::SIToFPOp>(loc, resType, src);
                })
                .Case<arith::TruncFOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return rewriter.create<arith::TruncFOp>(loc, resType, src);
                })
                .Default([](Operation *op) {
                  llvm_unreachable("unsupported op in generating ");
                  return nullptr;
                });
    }
    return src;
  };

  while (true) {
    if (!dyn_cast<ShapedType>(operand.getType())) {
      return reconstructScalarValue(operand);
    } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<DenseElementsAttr>(op.getValue())) {
        if (!attr.isSplat()) {
          InFlightDiagnostic diag = emitError(loc)
                                    << "other value used in masked load "
                                       "produced by unsupported instruction";
          return nullptr;
        }
        auto elemValue = attr.getSplatValue<Attribute>();
        auto constOp = arith::ConstantOp::materialize(
            rewriter, elemValue, attr.getElementType(), op.getLoc());
        return reconstructScalarValue(constOp.getResult());
      }
      InFlightDiagnostic diag = emitError(loc)
                                << "other value used in masked load produced "
                                   "by unsupported instruction";
      return nullptr;
    } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
      operand = op.getSrc();
    } else if (auto op = operand.getDefiningOp<arith::SIToFPOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else if (auto op = operand.getDefiningOp<arith::TruncFOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else {
      InFlightDiagnostic diag = emitError(loc)
                                << "other value used in masked load produced "
                                   "by unsupported instruction";
      return nullptr;
    }
  }
  return nullptr;
}

SmallVector<int64_t> getBroadcastDims(RankedTensorType src,
                                      RankedTensorType dst) {
  SmallVector<int64_t> broadcastDims;
  auto srcShape = src.getShape();
  auto dstShape = dst.getShape();

  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (dstShape[i] != srcShape[i]) {
      assert(srcShape[i] == 1 &&
             "Size of source broadcast dimension must be 1");
      broadcastDims.push_back(i);
    }
  }
  assert(!broadcastDims.empty() && "Cannot identify broadcast dimension");
  return broadcastDims;
}

// Dimensions of collapesd tensor is all unbroadcast dims
SmallVector<int64_t> getUnbroadcastDims(RankedTensorType src,
                                        RankedTensorType dst) {
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

scf::ForOp createNestedLoops(
    OpBuilder &builder, Location loc, unsigned currentDim, unsigned totalDims,
    ValueRange LBs, ValueRange UBs, ValueRange steps, SmallVector<Value> &ivs,
    ValueRange initArgs,
    function_ref<void(OpBuilder &, Location, SmallVector<Value> &, ValueRange)>
        bodyBuilder) {

  if (currentDim >= totalDims) {
    bodyBuilder(builder, loc, ivs, initArgs);
    return nullptr;
  }

  auto loop = builder.create<scf::ForOp>(
      loc, LBs[currentDim], UBs[currentDim], steps[currentDim], initArgs,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange iterArgs) {
        ivs.push_back(iv);
        auto innerLoop = createNestedLoops(nestedBuilder, nestedLoc,
                                           currentDim + 1, totalDims, LBs, UBs,
                                           steps, ivs, iterArgs, bodyBuilder);
        if (innerLoop) {
          nestedBuilder.create<scf::YieldOp>(loc, innerLoop.getResults());
        }
      });

  return loop;
}
} // namespace mlir::dicp
