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
#include <variant>

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

FailureOr<TypedAttr> specializeTypelessValueToAttr(TypelessValue value,
                                                   Type type, OpBuilder &b) {
  // Common float and integer MLIR types used as map keys.
  mlir::Type f16Ty = Float16Type::get(b.getContext());
  mlir::Type f32Ty = Float32Type::get(b.getContext());
  mlir::Type bf16Ty = BFloat16Type::get(b.getContext());

  mlir::Type i8TySL = IntegerType::get(
      b.getContext(), 8, IntegerType::SignednessSemantics::Signless);
  mlir::Type i8TyS = IntegerType::get(b.getContext(), 8,
                                      IntegerType::SignednessSemantics::Signed);
  mlir::Type i8TyU = IntegerType::get(
      b.getContext(), 8, IntegerType::SignednessSemantics::Unsigned);

  mlir::Type i16TySL = IntegerType::get(
      b.getContext(), 16, IntegerType::SignednessSemantics::Signless);
  mlir::Type i16TyS = IntegerType::get(
      b.getContext(), 16, IntegerType::SignednessSemantics::Signed);
  mlir::Type i16TyU = IntegerType::get(
      b.getContext(), 16, IntegerType::SignednessSemantics::Unsigned);

  mlir::Type i32TySL = IntegerType::get(
      b.getContext(), 32, IntegerType::SignednessSemantics::Signless);
  mlir::Type i32TyS = IntegerType::get(
      b.getContext(), 32, IntegerType::SignednessSemantics::Signed);
  mlir::Type i32TyU = IntegerType::get(
      b.getContext(), 32, IntegerType::SignednessSemantics::Unsigned);

  mlir::Type i64TySL = IntegerType::get(
      b.getContext(), 64, IntegerType::SignednessSemantics::Signless);
  mlir::Type i64TyS = IntegerType::get(
      b.getContext(), 64, IntegerType::SignednessSemantics::Signed);
  mlir::Type i64TyU = IntegerType::get(
      b.getContext(), 64, IntegerType::SignednessSemantics::Unsigned);

  // Create APFloat values for float semantics (half, single, bfloat).
  llvm::APFloat halfZero = llvm::APFloat::getZero(llvm::APFloat::IEEEhalf());
  llvm::APFloat halfOne(llvm::APFloat::IEEEhalf(), 1);
  llvm::APFloat halfMax = llvm::APFloat::getInf(llvm::APFloat::IEEEhalf());
  llvm::APFloat halfMin =
      llvm::APFloat::getInf(llvm::APFloat::IEEEhalf(), /*Negative=*/true);

  llvm::APFloat floatZero = llvm::APFloat::getZero(llvm::APFloat::IEEEsingle());
  llvm::APFloat floatOne(llvm::APFloat::IEEEsingle(), 1);
  llvm::APFloat floatMax = llvm::APFloat::getInf(llvm::APFloat::IEEEsingle());
  llvm::APFloat floatMin =
      llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(), /*Negative=*/true);

  // BF16 (bfloat16) semantics via APFloat.
  llvm::APFloat bfloatZero = llvm::APFloat::getZero(llvm::APFloat::BFloat());
  llvm::APFloat bfloatOne(llvm::APFloat::BFloat(), 1);
  llvm::APFloat bfloatMax = llvm::APFloat::getInf(llvm::APFloat::BFloat());
  llvm::APFloat bfloatMin =
      llvm::APFloat::getInf(llvm::APFloat::BFloat(), /*Negative=*/true);

  // Helper to use the opaque pointer of a Type as a stable key.
  auto toPtr = [](mlir::Type ty) { return ty.getAsOpaquePointer(); };

  // Store initialization values. Use signed and unsigned integer variants to
  // avoid narrowing/overflow problems.
  using InitValVariant =
      std::variant<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
                   int64_t, uint64_t, llvm::APFloat>;

  std::map<std::pair<TypelessValue, const void *>, InitValVariant> initMap = {
      // Zero values (floats and integers).
      {{TypelessValue::Zero, toPtr(f16Ty)}, halfZero},
      {{TypelessValue::Zero, toPtr(f32Ty)}, floatZero},
      {{TypelessValue::Zero, toPtr(bf16Ty)}, bfloatZero},

      {{TypelessValue::Zero, toPtr(i8TySL)}, (int8_t)0},
      {{TypelessValue::Zero, toPtr(i8TyS)}, (int8_t)0},
      {{TypelessValue::Zero, toPtr(i8TyU)}, (uint8_t)0},

      {{TypelessValue::Zero, toPtr(i16TySL)}, (int16_t)0},
      {{TypelessValue::Zero, toPtr(i16TyS)}, (int16_t)0},
      {{TypelessValue::Zero, toPtr(i16TyU)}, (uint16_t)0},

      {{TypelessValue::Zero, toPtr(i32TySL)}, (int32_t)0},
      {{TypelessValue::Zero, toPtr(i32TyS)}, (int32_t)0},
      {{TypelessValue::Zero, toPtr(i32TyU)}, (uint32_t)0},

      {{TypelessValue::Zero, toPtr(i64TySL)}, (int64_t)0},
      {{TypelessValue::Zero, toPtr(i64TyS)}, (int64_t)0},
      {{TypelessValue::Zero, toPtr(i64TyU)}, (uint64_t)0},

      // Min values (floats and integers).
      {{TypelessValue::Min, toPtr(f16Ty)}, halfMin},
      {{TypelessValue::Min, toPtr(f32Ty)}, floatMin},
      {{TypelessValue::Min, toPtr(bf16Ty)}, bfloatMin},

      {{TypelessValue::Min, toPtr(i8TySL)}, std::numeric_limits<int8_t>::min()},
      {{TypelessValue::Min, toPtr(i8TyS)}, std::numeric_limits<int8_t>::min()},
      {{TypelessValue::Min, toPtr(i8TyU)}, std::numeric_limits<uint8_t>::min()},

      {{TypelessValue::Min, toPtr(i16TySL)},
       std::numeric_limits<int16_t>::min()},
      {{TypelessValue::Min, toPtr(i16TyS)},
       std::numeric_limits<int16_t>::min()},
      {{TypelessValue::Min, toPtr(i16TyU)},
       std::numeric_limits<uint16_t>::min()},

      {{TypelessValue::Min, toPtr(i32TySL)},
       std::numeric_limits<int32_t>::min()},
      {{TypelessValue::Min, toPtr(i32TyS)},
       std::numeric_limits<int32_t>::min()},
      {{TypelessValue::Min, toPtr(i32TyU)},
       std::numeric_limits<uint32_t>::min()},

      {{TypelessValue::Min, toPtr(i64TySL)},
       std::numeric_limits<int64_t>::min()},
      {{TypelessValue::Min, toPtr(i64TyS)},
       std::numeric_limits<int64_t>::min()},
      {{TypelessValue::Min, toPtr(i64TyU)},
       std::numeric_limits<uint64_t>::min()}, // 0

      // Max values (floats and integers).
      {{TypelessValue::Max, toPtr(f16Ty)}, halfMax},
      {{TypelessValue::Max, toPtr(f32Ty)}, floatMax},
      {{TypelessValue::Max, toPtr(bf16Ty)}, bfloatMax},

      {{TypelessValue::Max, toPtr(i8TySL)}, std::numeric_limits<int8_t>::max()},
      {{TypelessValue::Max, toPtr(i8TyS)}, std::numeric_limits<int8_t>::max()},
      {{TypelessValue::Max, toPtr(i8TyU)}, std::numeric_limits<uint8_t>::max()},

      {{TypelessValue::Max, toPtr(i16TySL)},
       std::numeric_limits<int16_t>::max()},
      {{TypelessValue::Max, toPtr(i16TyS)},
       std::numeric_limits<int16_t>::max()},
      {{TypelessValue::Max, toPtr(i16TyU)},
       std::numeric_limits<uint16_t>::max()},

      {{TypelessValue::Max, toPtr(i32TySL)},
       std::numeric_limits<int32_t>::max()},
      {{TypelessValue::Max, toPtr(i32TyS)},
       std::numeric_limits<int32_t>::max()},
      {{TypelessValue::Max, toPtr(i32TyU)},
       std::numeric_limits<uint32_t>::max()},

      {{TypelessValue::Max, toPtr(i64TySL)},
       std::numeric_limits<int64_t>::max()},
      {{TypelessValue::Max, toPtr(i64TyS)},
       std::numeric_limits<int64_t>::max()},
      {{TypelessValue::Max, toPtr(i64TyU)},
       std::numeric_limits<uint64_t>::max()},
  };

  // Lookup key for the requested typeless value + concrete type.
  std::pair<TypelessValue, const void *> key =
      std::make_pair(value, toPtr(type));
  auto it = initMap.find(key);
  if (it == initMap.end())
    return failure();

  // Integer handling: prefer using the provided 'type' for IntegerAttr so
  // signedness/width are preserved.
  if (type.isInteger(8) || type.isInteger(16) || type.isInteger(32) ||
      type.isInteger(64)) {
    unsigned bitWidth = type.getIntOrFloatBitWidth();

    // Signed integers: extract signed variant and create IntegerAttr directly.
    if (type.isSignedInteger(bitWidth)) {
      switch (bitWidth) {
      case 8:
        return success(IntegerAttr::get(type, std::get<int8_t>(it->second)));
      case 16:
        return success(IntegerAttr::get(type, std::get<int16_t>(it->second)));
      case 32:
        return success(IntegerAttr::get(type, std::get<int32_t>(it->second)));
      case 64:
        return success(IntegerAttr::get(type, std::get<int64_t>(it->second)));
      default:
        return failure();
      }
    }

    // Unsigned integers: extract unsigned variant. For 64-bit unsigned use
    // APInt to avoid overflow of signed int64_t.
    if (type.isUnsignedInteger(bitWidth)) {
      switch (bitWidth) {
      case 8:
        return success(IntegerAttr::get(
            type, static_cast<int64_t>(std::get<uint8_t>(it->second))));
      case 16:
        return success(IntegerAttr::get(
            type, static_cast<int64_t>(std::get<uint16_t>(it->second))));
      case 32:
        return success(IntegerAttr::get(
            type, static_cast<int64_t>(std::get<uint32_t>(it->second))));
      case 64: {
        uint64_t uval = std::get<uint64_t>(it->second);
        llvm::APInt apv(/*numBits=*/64, uval, /*isSigned=*/false);
        return success(IntegerAttr::get(type, apv));
      }
      default:
        return failure();
      }
    }

    // Signless integers: treat as signless using the signed variants (original
    // code used signless integers everywhere for constants).
    switch (bitWidth) {
    case 8:
      return success(IntegerAttr::get(type, std::get<int8_t>(it->second)));
    case 16:
      return success(IntegerAttr::get(type, std::get<int16_t>(it->second)));
    case 32:
      return success(IntegerAttr::get(type, std::get<int32_t>(it->second)));
    case 64:
      return success(IntegerAttr::get(type, std::get<int64_t>(it->second)));
    default:
      return failure();
    }
  }

  // Floating-point handling (half, bf16, single).
  if (isa<Float16Type>(type))
    return success(FloatAttr::get(f16Ty, std::get<llvm::APFloat>(it->second)));
  if (isa<Float32Type>(type))
    return success(FloatAttr::get(f32Ty, std::get<llvm::APFloat>(it->second)));
  if (isa<BFloat16Type>(type))
    return success(FloatAttr::get(bf16Ty, std::get<llvm::APFloat>(it->second)));

  return failure();
}

// Specialize the Typeless Value (Zero, Min, Max) into a mlir constant value
FailureOr<Value> specializeTypelessValueToConstant(TypelessValue value,
                                                   Type type, Location loc,
                                                   OpBuilder &b) {
  std::function<mlir::Type(mlir::Type)> getElemType = [&](mlir::Type ty) {
    if (auto ptrType = dyn_cast<triton::PointerType>(getElementTypeOrSelf(ty)))
      return getElemType(ptrType.getPointeeType());
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(ty))
      return getElemType(tensorType.getElementType());
    return ty;
  };

  if (value == TypelessValue::Undefined)
    return failure();
  if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    auto elemType = getElemType(tensorType);
    FailureOr<TypedAttr> typedAttr =
        specializeTypelessValueToAttr(value, elemType, b);
    if (failed(typedAttr))
      return failure();
    auto otherTensorType =
        RankedTensorType::get(tensorType.getShape(), elemType);
    auto denseAttr = DenseElementsAttr::get(otherTensorType, *typedAttr);
    return b.create<mlir::arith::ConstantOp>(loc, denseAttr).getResult();
  }
  if (mlir::isa<mlir::FloatType>(type) || mlir::isa<mlir::IntegerType>(type)) {
    FailureOr<TypedAttr> typedAttr =
        specializeTypelessValueToAttr(value, type, b);
    if (failed(typedAttr))
      return failure();
    return b.create<mlir::arith::ConstantOp>(loc, *typedAttr).getResult();
  }
  return failure();
}

} // namespace mlir::dicp
