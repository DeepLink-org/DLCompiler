#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"
#include "dicp/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dicp-stage-utils"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace mlir::dicp;

namespace {

static FailureOr<StageSubUnitTag> parseStageSubUnitTag(StringRef tagName) {
  constexpr llvm::StringLiteral kStageMarker = ".stage_";
  constexpr llvm::StringLiteral kSubPrefix = "sub_";
  constexpr llvm::StringLiteral kUnitPrefix = "u_";

  StringRef suffix = tagName;
  if (!suffix.starts_with("stage_")) {
    size_t stageMarkerPos = suffix.rfind(kStageMarker);
    if (stageMarkerPos == StringRef::npos)
      return failure();
    suffix = suffix.drop_front(stageMarkerPos + 1);
  }

  SmallVector<StringRef, 3> parts;
  suffix.split(parts, '.');
  if (parts.size() != 3)
    return failure();

  auto parseComponent = [](StringRef text,
                           StringRef prefix) -> FailureOr<int64_t> {
    if (!text.consume_front(prefix))
      return failure();
    int64_t value = -1;
    if (text.getAsInteger(10, value))
      return failure();
    return value;
  };

  FailureOr<int64_t> stage = parseComponent(parts[0], "stage_");
  FailureOr<int64_t> sub = parseComponent(parts[1], kSubPrefix);
  FailureOr<int64_t> unit = parseComponent(parts[2], kUnitPrefix);
  if (failed(stage) || failed(sub) || failed(unit))
    return failure();

  return StageSubUnitTag{*stage, *sub, *unit};
}

/// Finds a stage/sub/unit tag either from canonical unit attributes or from the
/// consolidated tile metadata dictionary.
static std::optional<StageSubUnitTag>
findStageSubUnitTag(Operation *op, StringRef attrPrefix, StringRef dictKey,
                    StringRef debugLabel) {
  if (!op)
    return std::nullopt;

  for (const NamedAttribute &attr : op->getAttrs()) {
    StringRef attrName = attr.getName().strref();
    if (!attrName.starts_with(attrPrefix))
      continue;

    FailureOr<StageSubUnitTag> parsed = parseStageSubUnitTag(attrName);
    if (succeeded(parsed)) {
      LDBG(debugLabel << ": found canonical tag '" << attrName << "' on "
                      << op->getName());
      return *parsed;
    }

    LDBG(debugLabel << ": ignored malformed canonical tag '" << attrName
                    << "' on " << op->getName());
  }

  if (DictionaryAttr dict = getTileMeta(op)) {
    if (StringAttr tagAttr = dict.getAs<StringAttr>(dictKey)) {
      FailureOr<StageSubUnitTag> parsed =
          parseStageSubUnitTag(tagAttr.getValue());
      if (succeeded(parsed)) {
        LDBG(debugLabel << ": found metadata tag '" << tagAttr.getValue()
                        << "' on " << op->getName());
        return *parsed;
      }

      LDBG(debugLabel << ": ignored malformed metadata tag '"
                      << tagAttr.getValue() << "' on " << op->getName());
    }
  }

  return std::nullopt;
}

/// Returns true when an attribute is part of temporary tiling-unit ownership
/// and should be cleared after the scheduling decision is materialized.
static bool isTransientTilingUnitAttr(StringRef attrName) {
  return attrName == stage_attrs::kTileMetaTag ||
         attrName.starts_with(stage_attrs::kTileMetaAnchorTag) ||
         attrName.starts_with(stage_attrs::kTileMetaFuseTag);
}

} // namespace

bool mlir::dicp::isMatMulOp(Operation *op) { return isa<linalg::MatmulOp>(op); }

bool mlir::dicp::isSIMDLikeOp(Operation *op) {
  if (isa<linalg::ReduceOp, linalg::BroadcastOp, linalg::FillOp,
          linalg::GenericOp>(op))
    return true;
  if (isa<arith::ArithDialect, math::MathDialect>(op->getDialect())) {
    auto isShaped = [](Type t) { return isa<ShapedType>(t); };
    return llvm::any_of(op->getOperandTypes(), isShaped) ||
           llvm::any_of(op->getResultTypes(), isShaped);
  }
  return false;
}

bool mlir::dicp::isWriteOp(mlir::Operation *op) {
  return llvm::isa<mlir::memref::CopyOp, mlir::memref::StoreOp,
                   mlir::linalg::CopyOp,
                   mlir::bufferization::MaterializeInDestinationOp,
                   mlir::tensor::InsertSliceOp>(op);
}

void mlir::dicp::propagateDicpAttributes(Operation *srcOp, Operation *dstOp) {
  for (const NamedAttribute &attr : srcOp->getAttrs()) {
    if (attr.getName().strref().starts_with(kDicpStageAttrPrefix)) {
      dstOp->setAttr(attr.getName(), attr.getValue());
    }
  }
}

std::optional<int64_t> mlir::dicp::getStageId(Operation *op) {
  if (!op)
    return std::nullopt;
  auto stageAttr =
      op->getAttrOfType<IntegerAttr>(stage_attrs::kNPUStageAttrName);
  if (!stageAttr)
    return std::nullopt;
  return stageAttr.getInt();
}

std::optional<StageSubUnitTag> mlir::dicp::getProducerFuseTag(Operation *op) {
  return findStageSubUnitTag(op, stage_attrs::kTileMetaFuseTag,
                             stage_attrs::kTileMetaFuseTag,
                             "producer-fuse-tag");
}

std::optional<StageSubUnitTag> mlir::dicp::getAnchorTileTag(Operation *op) {
  return findStageSubUnitTag(op, stage_attrs::kTileMetaAnchorTag,
                             stage_attrs::kTileMetaAnchorTag,
                             "anchor-tile-tag");
}

std::optional<StageSubUnitTag> mlir::dicp::getStageSubUnitTag(Operation *op) {
  if (std::optional<StageSubUnitTag> tag = getProducerFuseTag(op))
    return tag;
  return getAnchorTileTag(op);
}

bool mlir::dicp::isSameStageAndSubstage(const StageSubUnitTag &lhs,
                                        const StageSubUnitTag &rhs) {
  return lhs.isValid() && rhs.isValid() && lhs.stage == rhs.stage &&
         lhs.sub == rhs.sub;
}

DictionaryAttr mlir::dicp::getTileMeta(Operation *op) {
  if (!op)
    return nullptr;
  return op->getAttrOfType<DictionaryAttr>(stage_attrs::kTileMetaTag);
}

FailureOr<SmallVector<int64_t, 4>> mlir::dicp::getTileSizes(Operation *op) {
  auto dict = getTileMeta(op);
  if (!dict)
    return failure();

  auto sizesAttr = dyn_cast_or_null<DenseI64ArrayAttr>(
      dict.get(stage_attrs::kTileMetaTileSizesTag));
  if (!sizesAttr)
    return failure();

  SmallVector<int64_t, 4> tileSizes(sizesAttr.asArrayRef().begin(),
                                    sizesAttr.asArrayRef().end());
  return tileSizes;
}

void mlir::dicp::setTileMeta(Operation *op, StringRef anchorTag,
                             StringRef producerFuseTag,
                             ArrayRef<int64_t> tileSizes) {
  if (!op)
    return;

  OpBuilder builder(op->getContext());
  SmallVector<NamedAttribute, 3> meta;
  meta.push_back(builder.getNamedAttr(stage_attrs::kTileMetaAnchorTag,
                                      builder.getStringAttr(anchorTag)));
  meta.push_back(builder.getNamedAttr(stage_attrs::kTileMetaFuseTag,
                                      builder.getStringAttr(producerFuseTag)));
  meta.push_back(builder.getNamedAttr(stage_attrs::kTileMetaTileSizesTag,
                                      builder.getDenseI64ArrayAttr(tileSizes)));
  op->setAttr(stage_attrs::kTileMetaTag, builder.getDictionaryAttr(meta));
}

void mlir::dicp::clearTilingUnitAttrs(Operation *op) {
  if (!op)
    return;

  SmallVector<StringAttr> attrsToRemove;
  for (const NamedAttribute &attr : op->getAttrs()) {
    if (isTransientTilingUnitAttr(attr.getName().strref()))
      attrsToRemove.push_back(attr.getName());
  }

  if (!attrsToRemove.empty()) {
    LDBG("Clearing " << attrsToRemove.size()
                     << " transient tiling-unit attrs from " << op->getName());
  }
  for (StringAttr attrName : attrsToRemove)
    op->removeAttr(attrName);
}

void mlir::dicp::clearCrossUnitUserAttrs(Operation *op) {
  if (!op)
    return;
  op->removeAttr(stage_attrs::kCrossTillUnitAttr);
}

void mlir::dicp::removeTmpStageAttributes(ModuleOp moduleOp) {
  moduleOp->walk([](Operation *op) {
    SmallVector<StringAttr> attrsToRemove;
    for (const NamedAttribute &attr : op->getAttrs()) {
      if (attr.getName().strref().starts_with(kDicpStageAttrPrefix))
        attrsToRemove.push_back(attr.getName());
    }
    for (StringAttr attrName : attrsToRemove)
      op->removeAttr(attrName);
  });
}

/// Check if the operation is a candidate for elementwise-to-generic conversion.
static bool isPureTensorElementwiseOp(Operation *op) {
  if (!isMemoryEffectFree(op) || op->getNumRegions() != 0)
    return false;
  return OpTrait::hasElementwiseMappableTraits(op) &&
         llvm::any_of(op->getOperandTypes(), llvm::IsaPred<RankedTensorType>) &&
         llvm::all_of(op->getOperandTypes(), llvm::IsaPred<RankedTensorType>);
}

bool mlir::dicp::isPartitionPropagatableTensorOp(Operation *op) {
  if (op->getNumResults() != 1 ||
      !isa<RankedTensorType>(op->getResult(0).getType()))
    return false;

  if (isPureTensorElementwiseOp(op))
    return true;

  return TypeSwitch<Operation *, bool>(op)
      .Case<linalg::FillOp>(
          [](auto fillOp) { return fillOp.hasPureTensorSemantics(); })
      .Case<linalg::BroadcastOp>(
          [](auto broadcastOp) { return broadcastOp.hasPureTensorSemantics(); })
      .Case<linalg::TransposeOp>(
          [](auto transposeOp) { return transposeOp.hasPureTensorSemantics(); })
      .Case<linalg::ReduceOp>(
          [](auto reduceOp) { return reduceOp.hasPureTensorSemantics(); })
      .Case<linalg::MatmulOp>(
          [](auto matmulOp) { return matmulOp.hasPureTensorSemantics(); })
      .Default([](Operation *) { return false; });
}

bool mlir::dicp::isConvertibleElementwiseOp(Operation *op) {
  if (!op->hasAttr(stage_attrs::kNPUStageAttrName) ||
      isa<linalg::GenericOp>(op))
    return false;
  return isPureTensorElementwiseOp(op);
}

/// Retrieve the loop or tensor rank of an operation.
int64_t mlir::dicp::getRank(Operation *op) {
  if (isWriteOp(op))
    return cast<ShapedType>(op->getOperand(0).getType()).getRank();
  return TypeSwitch<Operation *, int64_t>(op)
      .Case<linalg::LinalgOp>(
          [](linalg::LinalgOp linalgOp) { return linalgOp.getNumLoops(); })
      .Default([](Operation *op) -> int64_t {
        if (op->getNumResults() == 0)
          return 0;
        auto shaped = dyn_cast<ShapedType>(op->getResult(0).getType());
        return shaped ? shaped.getRank() : 0;
      });
}

bool mlir::dicp::isDynamicOrScalarOp(Operation *op) {
  if (!isa<tensor::TensorDialect, memref::MemRefDialect, math::MathDialect,
           arith::ArithDialect, linalg::LinalgDialect,
           bufferization::BufferizationDialect>(op->getDialect()))
    return false;

  auto isTrivial = [](Type t) {
    if (auto shaped = dyn_cast<ShapedType>(t))
      return !shaped.hasStaticShape() || shaped.getNumElements() == 1;
    return true;
  };

  return llvm::all_of(op->getOperandTypes(), isTrivial) &&
         llvm::all_of(op->getResultTypes(), isTrivial);
}

bool mlir::dicp::isRootedTileClosureSeedOp(Operation *op) {
  return isa<memref::AllocOp, tensor::EmptyOp, bufferization::AllocTensorOp>(
      op);
}

bool mlir::dicp::isStructuredControlFlowOp(Operation *op) {
  // Use MLIR standard RegionBranchOpInterface to detect structured control
  // flow operations (scf.for, scf.while, scf.forall, scf.if, etc).
  // Terminators are explicitly excluded as they are structural markers,
  // not executable operations that should participate in SubStage boundaries.
  if (!op)
    return false;
  if (op->hasTrait<OpTrait::IsTerminator>())
    return false;
  return isa<RegionBranchOpInterface>(op);
}

bool mlir::dicp::isControlFlowOp(Operation *op) {
  if (!op)
    return false;
  return isa<BranchOpInterface, RegionBranchOpInterface>(op) ||
         op->hasTrait<OpTrait::IsTerminator>();
}

memref::AllocOp mlir::dicp::resolveUnderlyingAlloc(Operation *producer) {
  if (!producer)
    return nullptr;
  if (producer->getNumResults() != 1)
    return nullptr;
  Value sourceRoot = traceToSourceRoot(producer->getResult(0));
  return sourceRoot ? sourceRoot.getDefiningOp<memref::AllocOp>() : nullptr;
}

Value mlir::dicp::getTilingReferenceValue(Operation *anchorOp) {
  if (!anchorOp)
    return Value{};

  return TypeSwitch<Operation *, Value>(anchorOp)
      .Case<linalg::CopyOp>([](auto op) { return op.getInputs().front(); })
      .Case<memref::CopyOp>([](auto op) { return op.getTarget(); })
      .Case<memref::StoreOp>([](auto op) { return op.getMemref(); })
      .Case<bufferization::MaterializeInDestinationOp>(
          [](auto op) { return op.getDest(); })
      .Case<tensor::InsertSliceOp>([](auto op) { return op.getSource(); })
      .Case<linalg::LinalgOp>([](auto op) -> Value {
        if (op->getNumResults() == 1)
          return op->getResult(0);
        auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
        if (linalgOp.getNumDpsInits() == 1)
          return linalgOp.getDpsInits().front();
        return Value{};
      })
      .Default([](Operation *op) -> Value {
        for (Value result : op->getResults())
          if (isa<ShapedType>(result.getType()))
            return result;
        for (Value operand : op->getOperands())
          if (isa<ShapedType>(operand.getType()))
            return operand;
        return Value{};
      });
}

static bool isRootedTileClosurePassthroughOp(Operation *op) {
  if (isa<ViewLikeOpInterface, bufferization::ToTensorOp,
          bufferization::ToBufferOp>(op)) {
    return true;
  }
  return isPartitionPropagatableTensorOp(op);
}

bool mlir::dicp::isStageEligibleForTiling(const StageInfo &stage) {
  for (Operation *op : stage.getOps()) {
    // Rule: linalg.generic with non-trivial body.
    // A body with more than one non-terminator op encodes complex index
    // arithmetic or multi-step reductions (e.g. linalg.index + index_cast)
    // that are not amenable to simple tile-and-fuse. Skip the entire stage.
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      if (llvm::hasNItemsOrMore(genericOp.getBody()->without_terminator(), 2)) {
        LDBG("[StageEligibility] Stage "
             << stage.id << " disqualified: linalg.generic has non-trivial "
             << "body (>1 non-terminator op): " << *op);
        return false;
      }
    }

    // Rule: reject loops carrying extracted indirect memory accesses. Their
    // runtime address computation is not preserved by the current tiling path.
    bool foundDiscreteLoop = false;
    op->walk([&](scf::ForOp forOp) -> WalkResult {
      if (!forOp->hasAttr("ExtractedLoadOrStore"))
        return WalkResult::advance();
      LDBG("[StageEligibility] Stage "
           << stage.id << " disqualified: scf.for has "
           << "'ExtractedLoadOrStore' attribute: " << *forOp);
      foundDiscreteLoop = true;
      return WalkResult::interrupt();
    });
    if (foundDiscreteLoop)
      return false;
  }
  return true;
}
