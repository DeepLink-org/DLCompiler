#include "dicp/Dialect/LinalgExt/Analysis/SliceParallelAnalysis.h"
#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"
#include "dicp/Utils/Utils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::dicp;
#define DEBUG_TYPE "slice-parallel-analysis"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

namespace {

static OpFoldResult getIndexAttr(MLIRContext *ctx, int64_t value) {
  return Builder(ctx).getIndexAttr(value);
}

static std::optional<int64_t> getStaticElementCount(MemRefType type) {
  if (!type.hasStaticShape())
    return std::nullopt;
  return type.getNumElements();
}

static std::optional<MemoryRegion>
getStaticContiguousRootRegion(Value value, Value baseRoot) {
  auto memrefType = dyn_cast<MemRefType>(value.getType());
  if (!memrefType || !memref::isStaticShapeAndContiguousRowMajor(memrefType))
    return std::nullopt;

  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(memrefType.getStridesAndOffset(strides, offset))) {
    return std::nullopt;
  }
  if (ShapedType::isDynamic(offset))
    offset = 0;

  std::optional<int64_t> elementCount = getStaticElementCount(memrefType);
  if (!elementCount)
    return std::nullopt;

  MemoryRegion region;
  region.baseRoot = baseRoot;
  region.offsets.push_back(getIndexAttr(value.getContext(), offset));
  region.sizes.push_back(getIndexAttr(value.getContext(), *elementCount));
  region.strides.push_back(getIndexAttr(value.getContext(), 1));
  region.hasPreciseSlice = true;
  return region;
}

static std::optional<MemoryRegion>
getImpreciseRootRegion(Value baseRoot, Operation *op, StringRef reason) {
  LDBG("[MemoryRegion] Falling back to root region for " << *op << ": "
                                                         << reason);
  return MemoryRegion{baseRoot};
}

static std::optional<OpFoldResult> addWithConstant(OpFoldResult lhs,
                                                   int64_t rhs) {
  if (rhs == 0)
    return lhs;
  if (std::optional<int64_t> lhsCst = getConstantIntValue(lhs))
    return getIndexAttr(lhs.getContext(), (*lhsCst) + rhs);
  return std::nullopt;
}

static std::optional<int64_t> getStaticLinearOffset(memref::SubViewOp subview) {
  auto sourceType = dyn_cast<MemRefType>(subview.getSource().getType());
  if (!sourceType || !memref::isStaticShapeAndContiguousRowMajor(sourceType))
    return std::nullopt;

  SmallVector<int64_t> sourceStrides;
  int64_t sourceOffset = 0;
  if (failed(sourceType.getStridesAndOffset(sourceStrides, sourceOffset)) ||
      ShapedType::isDynamic(sourceOffset))
    return std::nullopt;

  if (subview.getMixedOffsets().size() != sourceStrides.size())
    return std::nullopt;

  int64_t linearOffset = 0;
  for (auto [offset, stride] :
       llvm::zip_equal(subview.getMixedOffsets(), sourceStrides)) {
    std::optional<int64_t> cstOffset = getConstantIntValue(offset);
    if (!cstOffset)
      return std::nullopt;
    linearOffset += (*cstOffset) * stride;
  }
  return linearOffset;
}

static FailureOr<RankedTensorType>
getTensorTypeForSliceLikeOp(OffsetSizeAndStrideOpInterface sliceOp) {
  LDBG("Inferring tensor type for slice-like op: " << *sliceOp.getOperation());
  return TypeSwitch<Operation *, FailureOr<RankedTensorType>>(
             sliceOp.getOperation())
      .Case<tensor::ExtractSliceOp>(
          [&](auto extractOp) { return extractOp.getResultType(); })
      .Case<tensor::InsertSliceOp>(
          [&](auto insertOp) { return insertOp.getSourceType(); })
      .Case<memref::SubViewOp>(
          [&](auto subviewOp) -> FailureOr<RankedTensorType> {
            MemRefType subviewType = subviewOp.getType();
            return RankedTensorType::get(subviewType.getShape(),
                                         subviewType.getElementType());
          })
      .Default([&](Operation *op) -> FailureOr<RankedTensorType> {
        LDBG("Unsupported slice-like op for tensor type inference: " << *op);
        return failure();
      });
}

} // namespace

std::optional<MemoryRegion> mlir::dicp::getMemoryRegion(Value value) {
  if (!value)
    return std::nullopt;

  if (auto toTensor = value.getDefiningOp<bufferization::ToTensorOp>())
    return getMemoryRegion(toTensor.getBuffer());
  if (auto toBuffer = value.getDefiningOp<bufferization::ToBufferOp>())
    return getMemoryRegion(toBuffer.getTensor());
  if (auto castOp = value.getDefiningOp<memref::CastOp>())
    return getMemoryRegion(castOp.getSource());

  if (auto reinterpretCast = value.getDefiningOp<memref::ReinterpretCastOp>()) {
    Value baseRoot = traceToSourceRoot(reinterpretCast.getSource());
    if (std::optional<MemoryRegion> region =
            getStaticContiguousRootRegion(value, baseRoot)) {
      auto offsets = reinterpretCast.getMixedOffsets();
      if (offsets.size() == 1) {
        region->offsets[0] = offsets.front();
        return region;
      }
    }

    return getImpreciseRootRegion(
        baseRoot, reinterpretCast,
        "reinterpret_cast is not a static contiguous 1-D slice");
  }

  if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
    std::optional<MemoryRegion> sourceRegion =
        getMemoryRegion(subview.getSource());
    if (!sourceRegion)
      return std::nullopt;

    Value baseRoot = sourceRegion->baseRoot
                         ? sourceRegion->baseRoot
                         : traceToSourceRoot(subview.getSource());
    if (std::optional<MemoryRegion> region =
            getStaticContiguousRootRegion(value, baseRoot)) {
      std::optional<int64_t> localOffset = getStaticLinearOffset(subview);
      if (!localOffset) {
        return getImpreciseRootRegion(
            baseRoot, subview, "subview offset is not statically linearizable");
      }

      if (!sourceRegion->hasPreciseSlice || sourceRegion->offsets.size() != 1) {
        region->offsets[0] = getIndexAttr(subview.getContext(), *localOffset);
        return region;
      }

      std::optional<OpFoldResult> combinedOffset =
          addWithConstant(sourceRegion->offsets.front(), *localOffset);
      if (!combinedOffset) {
        return getImpreciseRootRegion(baseRoot, subview,
                                      "composed linear offset is symbolic");
      }

      region->offsets[0] = *combinedOffset;
      return region;
    }

    return getImpreciseRootRegion(
        baseRoot, subview, "subview is not a static contiguous 1-D slice");
  }

  if (auto viewLike = value.getDefiningOp<ViewLikeOpInterface>())
    return getMemoryRegion(viewLike.getViewSource());

  Value baseRoot = traceToSourceRoot(value);
  if (std::optional<MemoryRegion> region =
          getStaticContiguousRootRegion(baseRoot, baseRoot))
    return region;
  return MemoryRegion{baseRoot};
}

FailureOr<bool>
mlir::dicp::proveMemoryRegionsDisjoint(const MemoryRegion &lhs,
                                       const MemoryRegion &rhs) {
  if (!lhs.isValid() || !rhs.isValid() || lhs.baseRoot != rhs.baseRoot ||
      !lhs.hasPreciseSlice || !rhs.hasPreciseSlice ||
      lhs.offsets.size() != rhs.offsets.size() ||
      lhs.sizes.size() != rhs.sizes.size() ||
      lhs.strides.size() != rhs.strides.size()) {
    return failure();
  }

  return proveSlicesDisjoint(lhs.baseRoot.getContext(), lhs.offsets, lhs.sizes,
                             lhs.strides, rhs.offsets, rhs.sizes, rhs.strides);
}

//===----------------------------------------------------------------------===//
// SliceRegionAnalysis Implementation
//===----------------------------------------------------------------------===//

std::optional<SliceRegion>
SliceRegionAnalysis::getSliceRegion(Operation *op) const {
  return TypeSwitch<Operation *, std::optional<SliceRegion>>(op)
      .Case<tensor::InsertSliceOp>([&](auto insert) {
        return SliceRegion{insert.getDest(),
                           insert.getMixedOffsets(),
                           insert.getMixedSizes(),
                           insert.getMixedStrides(),
                           op,
                           /*isWrite=*/true};
      })
      .Case<tensor::ExtractSliceOp>([&](auto extract) {
        return SliceRegion{extract.getSource(),
                           extract.getMixedOffsets(),
                           extract.getMixedSizes(),
                           extract.getMixedStrides(),
                           op,
                           /*isWrite=*/false};
      })
      .Case<memref::SubViewOp>([&](auto subview) {
        // A subview is logically a write if any of its users mutate memory.
        bool isWrite = llvm::any_of(subview->getUsers(), [](Operation *user) {
          auto memOp = dyn_cast<MemoryEffectOpInterface>(user);
          return memOp && memOp.hasEffect<MemoryEffects::Write>();
        });
        return SliceRegion{subview.getSource(),
                           subview.getMixedOffsets(),
                           subview.getMixedSizes(),
                           subview.getMixedStrides(),
                           op,
                           isWrite};
      })
      .Default([](Operation *) { return std::nullopt; });
}

//===----------------------------------------------------------------------===//
// DisjointSliceAnalysis Implementation
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<OffsetSizeAndStrideOpInterface>>
DisjointSliceAnalysis::analyze(Value rootValue) const {
  LDBG("Analyzing root value: " << rootValue);

  SmallVector<OffsetSizeAndStrideOpInterface> slices;
  if (failed(collectSlices(rootValue, slices)))
    return failure();
  if (failed(checkTransitiveEscape(slices)))
    return failure();
  if (failed(proveDisjointness(slices)))
    return failure();

  return slices;
}

FailureOr<SmallVector<OffsetSizeAndStrideOpInterface>>
DisjointSliceAnalysis::analyze(Operation *op) const {
  if (op->getNumResults() != 1) {
    LDBG("-> Rejected: Root operation must have exactly one result.");
    return failure();
  }
  return analyze(op->getResult(0));
}

LogicalResult DisjointSliceAnalysis::collectSlices(
    Value root, SmallVectorImpl<OffsetSizeAndStrideOpInterface> &slices) const {
  SmallVector<Value> worklist = {root};
  DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    for (Operation *user : current.getUsers()) {
      auto sliceOp = dyn_cast<OffsetSizeAndStrideOpInterface>(user);
      if (!sliceOp)
        continue;

      bool isBase = false;
      TypeSwitch<Operation *>(user)
          .Case<memref::SubViewOp, tensor::ExtractSliceOp>([&](auto op) {
            if (op.getSource() == current)
              isBase = true;
          })
          .Case<tensor::InsertSliceOp>([&](auto op) {
            if (op.getDest() == current) {
              isBase = true;
              worklist.push_back(op.getResult());
            }
          });

      if (isBase) {
        slices.push_back(sliceOp);
        LDBG("    -> Found slice: " << *sliceOp);
      }
    }
  }

  if (slices.empty()) {
    LDBG("  -> Rejected: No slice operations found.");
    return failure();
  }

  SetVector<OffsetSizeAndStrideOpInterface> uniqueSlices(slices.begin(),
                                                         slices.end());
  slices.assign(uniqueSlices.begin(), uniqueSlices.end());
  return success();
}

LogicalResult DisjointSliceAnalysis::checkTransitiveEscape(
    ArrayRef<OffsetSizeAndStrideOpInterface> slices) const {
  LDBG("[Phase 2] Checking transitive escape...");
  for (auto slice : slices) {
    if (slice->getNumResults() == 0)
      continue;

    bool escapes =
        llvm::any_of(slice->getResult(0).getUsers(),
                     [](Operation *op) { return isa<CallOpInterface>(op); });

    if (escapes) {
      LDBG("    -> Escape Detected: Slice reaches a function call.");
      return failure();
    }
  }
  return success();
}

LogicalResult DisjointSliceAnalysis::proveDisjointness(
    ArrayRef<OffsetSizeAndStrideOpInterface> slices) const {
  LDBG("[Phase 3] Computing N-dimensional disjointness...");
  for (size_t i = 0; i < slices.size(); ++i) {
    for (size_t j = i + 1; j < slices.size(); ++j) {
      if (!areDisjoint(slices[i], slices[j])) {
        LDBG("    -> Overlap Detected between:\n"
             << "       " << slices[i] << "\n"
             << "       and\n"
             << "       " << slices[j]);
        return failure();
      }
    }
  }
  return success();
}

bool DisjointSliceAnalysis::areDisjoint(
    OffsetSizeAndStrideOpInterface a, OffsetSizeAndStrideOpInterface b) const {
  auto offsetsA = a.getMixedOffsets();
  auto sizesA = a.getMixedSizes();
  auto stridesA = a.getMixedStrides();

  auto offsetsB = b.getMixedOffsets();
  auto sizesB = b.getMixedSizes();
  auto stridesB = b.getMixedStrides();

  if (offsetsA.size() != offsetsB.size())
    return false;

  FailureOr<bool> disjoint = proveSlicesDisjoint(
      a->getContext(), offsetsA, sizesA, stridesA, offsetsB, sizesB, stridesB);
  return succeeded(disjoint) && *disjoint;
}

//===----------------------------------------------------------------------===//
// Shared Utilities
//===----------------------------------------------------------------------===//

bool mlir::dicp::areRangesDisjoint1D(OpFoldResult offsetA, OpFoldResult sizeA,
                                     OpFoldResult strideA, OpFoldResult offsetB,
                                     OpFoldResult sizeB, OpFoldResult strideB) {
  MLIRContext *ctx = offsetA.getContext();
  FailureOr<bool> disjoint = proveSlicesDisjoint(
      ctx, {offsetA}, {sizeA}, {strideA}, {offsetB}, {sizeB}, {strideB});
  return succeeded(disjoint) && *disjoint;
}

bool mlir::dicp::isEqualOFR(OpFoldResult a, OpFoldResult b) {
  FailureOr<bool> equal = proveOfrEqual(a, b);
  return succeeded(equal) && *equal;
}

FailureOr<SmallVector<OpFoldResult>>
mlir::dicp::toStaticIndexAttrs(OpBuilder &b, ArrayRef<OpFoldResult> ofrs) {
  SmallVector<OpFoldResult> result;
  result.reserve(ofrs.size());
  for (auto ofr : ofrs) {
    auto cst = getConstantIntValue(ofr);
    if (!cst)
      return failure();
    result.push_back(b.getIndexAttr(*cst));
  }
  return result;
}

FailureOr<DirectSlicePartitionInfo> mlir::dicp::buildDirectSlicePartitionInfo(
    ArrayRef<OffsetSizeAndStrideOpInterface> slices) {
  DirectSlicePartitionInfo info;
  LDBG("[DirectSlice] Grouping " << slices.size()
                                 << " direct slice users into semantic "
                                    "partitions");

  for (OffsetSizeAndStrideOpInterface sliceInterface : slices) {
    auto sliceType = getTensorTypeForSliceLikeOp(sliceInterface);
    if (failed(sliceType)) {
      LDBG("[DirectSlice] Rejected unsupported slice op: "
           << *sliceInterface.getOperation());
      return failure();
    }

    SmallVector<OpFoldResult> offsets = sliceInterface.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = sliceInterface.getMixedSizes();
    SmallVector<OpFoldResult> strides = sliceInterface.getMixedStrides();
    int matchedPart =
        findMatchingPartition(info.partitions, offsets, sizes, strides);
    if (matchedPart >= 0) {
      if (info.partitions[matchedPart].sliceType != *sliceType) {
        LDBG("[DirectSlice] Rejected equivalent slices with mismatched result "
             "types for partition #"
             << matchedPart << ": " << *sliceInterface.getOperation());
        return failure();
      }
      LDBG("[DirectSlice] Reusing partition #"
           << matchedPart << " for slice " << *sliceInterface.getOperation());
      info.sliceToPartIdx[sliceInterface.getOperation()] =
          static_cast<unsigned>(matchedPart);
      continue;
    }

    matchedPart = static_cast<int>(info.partitions.size());
    SlicePartition part;
    part.offsets.assign(offsets.begin(), offsets.end());
    part.sizes.assign(sizes.begin(), sizes.end());
    part.strides.assign(strides.begin(), strides.end());
    part.sliceType = *sliceType;
    info.partitions.push_back(std::move(part));
    LDBG("[DirectSlice] Created partition #" << matchedPart << " for slice "
                                             << *sliceInterface.getOperation());
    info.sliceToPartIdx[sliceInterface.getOperation()] =
        static_cast<unsigned>(matchedPart);
  }

  if (info.partitions.empty()) {
    LDBG("[DirectSlice] Rejected: no semantic partitions discovered");
    return failure();
  }
  return info;
}

LogicalResult mlir::dicp::verifyDirectSlicePartitionsDisjoint(
    ArrayRef<SlicePartition> partitions) {
  LDBG("[DirectSlice] Verifying disjointness of " << partitions.size()
                                                  << " semantic partitions");
  for (auto [lhsIdx, lhs] : llvm::enumerate(partitions)) {
    for (unsigned rhsIdx = lhsIdx + 1; rhsIdx < partitions.size(); ++rhsIdx) {
      const SlicePartition &rhs = partitions[rhsIdx];
      if (lhs.offsets.size() != rhs.offsets.size()) {
        LDBG("[DirectSlice] Rejected mismatched partition ranks between #"
             << lhsIdx << " and #" << rhsIdx);
        return failure();
      }
      bool disjoint = llvm::any_of(
          llvm::zip(lhs.offsets, lhs.sizes, lhs.strides, rhs.offsets, rhs.sizes,
                    rhs.strides),
          [&](auto tuple) {
            auto [lhsOffset, lhsSize, lhsStride, rhsOffset, rhsSize,
                  rhsStride] = tuple;
            return areRangesDisjoint1D(lhsOffset, lhsSize, lhsStride, rhsOffset,
                                       rhsSize, rhsStride);
          });
      if (!disjoint) {
        LDBG("[DirectSlice] Rejected overlapping semantic partitions #"
             << lhsIdx << " and #" << rhsIdx);
        return failure();
      }
    }
  }
  return success();
}

LogicalResult mlir::dicp::verifyDirectSliceNoCallEscape(
    ArrayRef<OffsetSizeAndStrideOpInterface> slices) {
  for (OffsetSizeAndStrideOpInterface sliceInterface : slices) {
    if (sliceInterface->getNumResults() == 0)
      continue;
    Value sliceResult = sliceInterface->getResult(0);
    if (llvm::any_of(sliceResult.getUsers(), [](Operation *user) {
          return isa<CallOpInterface>(user);
        })) {
      LDBG("[DirectSlice] Rejected call escape from direct slice: "
           << *sliceInterface.getOperation());
      return failure();
    }
  }
  return success();
}

int mlir::dicp::findMatchingPartition(ArrayRef<SlicePartition> partitions,
                                      ArrayRef<OpFoldResult> offsets,
                                      ArrayRef<OpFoldResult> sizes,
                                      ArrayRef<OpFoldResult> strides) {
  for (auto [pi, p] : llvm::enumerate(partitions)) {
    if (p.offsets.size() != offsets.size())
      continue;
    FailureOr<bool> equivalent =
        proveSlicesEquivalent(p.sliceType.getContext(), p.offsets, p.sizes,
                              p.strides, offsets, sizes, strides);
    if (succeeded(equivalent) && *equivalent)
      return static_cast<int>(pi);
  }
  return -1;
}

//===----------------------------------------------------------------------===//
// verifyPartitionIsolation
//===----------------------------------------------------------------------===//

LogicalResult mlir::dicp::verifyPartitionIsolation(
    ArrayRef<SmallVector<Value>> partitionRoots,
    const DenseSet<Value> &forbiddenWriteTargets) {
  LDBG("[PartitionIsolation] Checking " << partitionRoots.size()
                                        << " partitions for isolation");

  for (auto [pi, roots] : llvm::enumerate(partitionRoots)) {
    // Compute the forward slice of all root values in this partition.
    SetVector<Operation *> fwSlice;
    for (Value root : roots)
      getForwardSlice(root, &fwSlice);

    LDBG("  Partition " << pi << ": forward slice has " << fwSlice.size()
                        << " ops");

    for (Operation *op : fwSlice) {
      // Check 1: No escapes to function calls.
      if (isa<CallOpInterface>(op)) {
        LDBG("  [REJECT] Partition " << pi << " escapes to call: " << *op);
        return failure();
      }

      // Check 2: No writes to forbidden memref targets via side effects.
      if (auto memOp = dyn_cast<MemoryEffectOpInterface>(op)) {
        SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
        memOp.getEffects(effects);
        for (auto &effect : effects) {
          if (!isa<MemoryEffects::Write>(effect.getEffect()))
            continue;
          Value written = effect.getValue();
          if (written && forbiddenWriteTargets.contains(written)) {
            LDBG("  [REJECT] Partition "
                 << pi << " writes to forbidden target: " << *op);
            return failure();
          }
        }
      }

      // Check 3: materialize_in_destination (may not expose via interface).
      if (auto mat = dyn_cast<bufferization::MaterializeInDestinationOp>(op)) {
        if (forbiddenWriteTargets.contains(mat.getDest())) {
          LDBG("  [REJECT] Partition " << pi
                                       << " materializes into forbidden "
                                          "target: "
                                       << *op);
          return failure();
        }
      }

      // Check 4: memref.copy target (may not expose via interface).
      if (auto copy = dyn_cast<memref::CopyOp>(op)) {
        if (forbiddenWriteTargets.contains(copy.getTarget())) {
          LDBG("  [REJECT] Partition "
               << pi << " copies into forbidden target: " << *op);
          return failure();
        }
      }
    }
  }

  LDBG("[PartitionIsolation] All partitions are isolated");
  return success();
}

//===----------------------------------------------------------------------===//
// InsertSliceChainAnalysis Implementation
//===----------------------------------------------------------------------===//

/// Returns true if `memref` and all its derived views (subview, cast)
/// are never written to.  Transitively follows view-like ops.
static bool isMemrefSubtreeReadOnly(Value memref) {
  SmallVector<Value> worklist = {memref};
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second)
      continue;
    for (Operation *user : v.getUsers()) {
      // Follow through view-like ops.
      if (isa<memref::SubViewOp, memref::CastOp>(user)) {
        worklist.push_back(user->getResult(0));
        continue;
      }
      if (auto memOp = dyn_cast<MemoryEffectOpInterface>(user)) {
        if (memOp.getEffectOnValue<MemoryEffects::Write>(v))
          return false;
        continue;
      }
      if (isMemoryEffectFree(user))
        continue;
      return false; // unknown op — conservative
    }
  }
  return true;
}

FailureOr<InsertSliceChainAnalysis::Result> InsertSliceChainAnalysis::analyze(
    Value root, function_ref<bool(Operation *)> allowedUser) const {
  Result result;

  // --- Phase 1: Collect chain values and slice ops (single-block). ---
  //
  // Chain values: root  +  every insert_slice result whose dest is a chain
  //               value.
  // Allowed users of a chain value:
  //   - tensor.extract_slice  (source == chain)        → read, collect
  //   - tensor.insert_slice   (dest   == chain)        → write, follow
  //   - tensor.insert_slice   (source == chain)        → read (not a chain op)
  //   - supported pure tensor op                       → read (not a chain op)
  //   - bufferization.to_memref → OK iff memref subtree is read-only
  //   - caller-provided allowedUser callback
  //
  // Invariant: insert_slice results must never be written through any alias.

  LDBG("[ChainAnalysis] Collecting chain from root: " << root);
  SmallVector<Value> worklist = {root};

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!result.chainValues.insert(current).second)
      continue;

    for (Operation *user : current.getUsers()) {
      bool ok =
          TypeSwitch<Operation *, bool>(user)
              .Case<tensor::ExtractSliceOp>([&](auto ext) {
                if (ext.getSource() == current) {
                  result.slices.push_back(
                      cast<OffsetSizeAndStrideOpInterface>(ext.getOperation()));
                }
                return true; // read — always safe
              })
              .Case<tensor::InsertSliceOp>([&](auto ins) {
                if (ins.getDest() == current) {
                  result.slices.push_back(
                      cast<OffsetSizeAndStrideOpInterface>(ins.getOperation()));
                  worklist.push_back(ins.getResult());
                }
                // source == current is a read — safe
                return true;
              })
              .Case<bufferization::ToBufferOp>([&](auto toBuf) {
                // to_memref creates a memref alias; the entire alias
                // subtree (including subview / cast) must be read-only.
                return isMemrefSubtreeReadOnly(toBuf.getResult());
              })
              .Default([&](Operation *op) {
                if (isPartitionPropagatableTensorOp(op))
                  return true;
                return allowedUser && allowedUser(op);
              });

      if (!ok) {
        LDBG("[ChainAnalysis] Rejected user: " << *user);
        return failure();
      }
    }
  }

  if (result.slices.size() < 2) {
    LDBG("[ChainAnalysis] Too few slices (" << result.slices.size() << ")");
    return failure();
  }

  // --- Phase 2: Group slices into partitions by offset/size/stride. ---
  LDBG("[ChainAnalysis] Grouping " << result.slices.size()
                                   << " slices into partitions");
  for (auto slice : result.slices) {
    auto offsets = slice.getMixedOffsets();
    auto sizes = slice.getMixedSizes();
    auto strides = slice.getMixedStrides();

    int foundIdx =
        findMatchingPartition(result.partitions, offsets, sizes, strides);

    if (foundIdx < 0) {
      foundIdx = static_cast<int>(result.partitions.size());
      SlicePartition p;
      p.offsets.assign(offsets.begin(), offsets.end());
      p.sizes.assign(sizes.begin(), sizes.end());
      p.strides.assign(strides.begin(), strides.end());
      if (auto ext = dyn_cast<tensor::ExtractSliceOp>(slice.getOperation()))
        p.sliceType = ext.getResultType();
      else if (auto ins = dyn_cast<tensor::InsertSliceOp>(slice.getOperation()))
        p.sliceType = ins.getSourceType();
      result.partitions.push_back(std::move(p));
    }
    result.sliceToPartIdx[slice.getOperation()] =
        static_cast<unsigned>(foundIdx);
  }

  if (result.partitions.size() < 2) {
    LDBG("[ChainAnalysis] Only " << result.partitions.size()
                                 << " partition - nothing to decompose");
    return failure();
  }

  // --- Phase 3: Verify pairwise disjointness. ---
  LDBG("[ChainAnalysis] Verifying disjointness of " << result.partitions.size()
                                                    << " partitions");
  for (size_t i = 0; i < result.partitions.size(); ++i) {
    for (size_t j = i + 1; j < result.partitions.size(); ++j) {
      bool disjoint = false;
      for (size_t dim = 0; dim < result.partitions[i].offsets.size(); ++dim) {
        if (areRangesDisjoint1D(result.partitions[i].offsets[dim],
                                result.partitions[i].sizes[dim],
                                result.partitions[i].strides[dim],
                                result.partitions[j].offsets[dim],
                                result.partitions[j].sizes[dim],
                                result.partitions[j].strides[dim])) {
          disjoint = true;
          break;
        }
      }
      if (!disjoint) {
        LDBG("[ChainAnalysis] Partitions " << i << " and " << j
                                           << " are NOT disjoint");
        return failure();
      }
    }
  }

  LDBG("[ChainAnalysis] Success: " << result.partitions.size()
                                   << " disjoint partitions");
  for (auto [idx, part] : llvm::enumerate(result.partitions))
    LDBG("  Partition " << idx << ": type=" << part.sliceType);

  return result;
}

//===----------------------------------------------------------------------===//
// LoopSliceParallelAnalysis Implementation
//===----------------------------------------------------------------------===//

LogicalResult LoopSliceParallelAnalysis::analyze(scf::ForOp loop) const {
  LDBG("[LoopParallelAnalysis] Analyzing loop at " << loop.getLoc());

  SmallVector<SliceRegion> writes;
  SmallVector<SliceRegion> reads;

  if (failed(collectAndVerifySideEffects(loop, writes, reads)))
    return failure();

  if (failed(verifyCrossIterationDisjointness(loop, writes)))
    return failure();

  if (failed(verifyNoCrossIterationDependencies(loop, writes, reads)))
    return failure();

  LDBG("[LoopParallelAnalysis] -> Success! Loop is safe for scf.forall "
       "conversion.");
  return success();
}

LogicalResult LoopSliceParallelAnalysis::collectAndVerifySideEffects(
    scf::ForOp loop, SmallVectorImpl<SliceRegion> &writes,
    SmallVectorImpl<SliceRegion> &reads) const {
  LDBG("  [Phase 1] Collecting slice ops and verifying side effects...");
  bool hasInvalidSideEffect = false;

  loop.getBody()->walk([&](Operation *op) {
    if (hasInvalidSideEffect)
      return WalkResult::interrupt();

    if (auto sliceOpt = sliceAnalyzer.getSliceRegion(op)) {
      if (sliceOpt->isWrite)
        writes.push_back(*sliceOpt);
      else
        reads.push_back(*sliceOpt);
      return WalkResult::advance();
    }

    if (isMemoryEffectFree(op))
      return WalkResult::advance();

    if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      if (memInterface.hasEffect<MemoryEffects::Write>()) {
        bool isSafeWrite =
            llvm::any_of(op->getOperands(), [this](Value operand) {
              return isDerivedFromSubview(operand);
            });
        if (!isSafeWrite) {
          LDBG("    -> [Reject] Non-slice memory write detected: " << *op);
          hasInvalidSideEffect = true;
          return WalkResult::interrupt();
        }
      }
    } else if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
      LDBG("    -> [Reject] Unknown op memory behavior: " << *op);
      hasInvalidSideEffect = true;
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  return success(!hasInvalidSideEffect);
}

LogicalResult LoopSliceParallelAnalysis::verifyCrossIterationDisjointness(
    scf::ForOp loop, ArrayRef<SliceRegion> writes) const {
  LDBG("  [Phase 2] Verifying cross-iteration disjointness for writes...");

  auto stepCst = getConstantIntValue(loop.getStep());
  if (!stepCst) {
    LDBG("    -> [Reject] Dynamic loop step is unsupported.");
    return failure();
  }
  int64_t step = *stepCst;

  for (const auto &write : writes) {
    bool isDisjoint = false;

    for (auto [dim, offsetOFR, sizeOFR] :
         llvm::zip(llvm::seq<size_t>(0, write.offsets.size()), write.offsets,
                   write.sizes)) {
      auto sizeCst = getConstantIntValue(sizeOFR);
      if (!sizeCst)
        continue;

      int64_t multiplier = 0;
      if (Value offsetVal = offsetOFR.dyn_cast<Value>()) {
        if (auto m = getIVMultiplier(offsetVal, loop)) {
          multiplier = *m;
        }
      }

      if (multiplier != 0 && std::abs(multiplier) * step >= *sizeCst) {
        LDBG("    -> Proven disjoint on dim "
             << dim << " (multiplier=" << std::abs(multiplier) << ", step="
             << step << " >= size=" << *sizeCst << ") for: " << *write.op);
        isDisjoint = true;
        break;
      }
    }

    if (!isDisjoint) {
      LDBG(
          "    -> [Reject] Cannot prove disjointness for write: " << *write.op);
      return failure();
    }
  }

  return success();
}

LogicalResult LoopSliceParallelAnalysis::verifyNoCrossIterationDependencies(
    scf::ForOp loop, ArrayRef<SliceRegion> writes,
    ArrayRef<SliceRegion> reads) const {
  LDBG("  [Phase 3] Checking for RAW/WAR cross-iteration dependencies...");

  for (const auto &read : reads) {
    Value readRoot = getRootValue(read.baseTensor, loop);
    bool aliasesWithWrite = false;
    bool isLocallyContained = false;

    for (const auto &write : writes) {
      Value writeRoot = getRootValue(write.baseTensor, loop);
      if (readRoot == writeRoot) {
        aliasesWithWrite = true;
        if (areSlicesIdentical(read, write)) {
          isLocallyContained = true;
          break;
        }
      }
    }

    if (aliasesWithWrite && !isLocallyContained) {
      LDBG("    -> [Reject] Read slice might access other iterations' data: "
           << *read.op);
      return failure();
    }
  }

  return success();
}

// --- Utility Methods Implementation ---

Value LoopSliceParallelAnalysis::getRootValue(Value val,
                                              scf::ForOp loop) const {
  while (true) {
    if (auto blockArg = dyn_cast<BlockArgument>(val)) {
      if (blockArg.getOwner() == loop.getBody())
        return blockArg;
      break;
    }
    Operation *op = val.getDefiningOp();
    if (!op)
      break;

    if (auto insert = dyn_cast<tensor::InsertSliceOp>(op)) {
      val = insert.getDest();
      continue;
    }
    if (auto extract = dyn_cast<tensor::ExtractSliceOp>(op)) {
      val = extract.getSource();
      continue;
    }
    if (auto subview = dyn_cast<memref::SubViewOp>(op)) {
      val = subview.getSource();
      continue;
    }
    if (auto cast = dyn_cast<tensor::CastOp>(op)) {
      val = cast.getSource();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(op)) {
      val = cast.getSource();
      continue;
    }
    break;
  }
  return val;
}

bool LoopSliceParallelAnalysis::isDerivedFromSubview(Value val) const {
  while (val) {
    if (val.getDefiningOp<memref::SubViewOp>())
      return true;
    if (auto cast = val.getDefiningOp<memref::CastOp>()) {
      val = cast.getSource();
    } else {
      break;
    }
  }
  return false;
}

bool LoopSliceParallelAnalysis::areSlicesIdentical(const SliceRegion &a,
                                                   const SliceRegion &b) const {
  if (a.offsets.size() != b.offsets.size())
    return false;
  FailureOr<bool> equivalent =
      proveSlicesEquivalent(a.op->getContext(), a.offsets, a.sizes, a.strides,
                            b.offsets, b.sizes, b.strides);
  return succeeded(equivalent) && *equivalent;
}

bool LoopSliceParallelAnalysis::isEqualOFR(OpFoldResult a,
                                           OpFoldResult b) const {
  return mlir::dicp::isEqualOFR(a, b);
}

std::optional<int64_t>
LoopSliceParallelAnalysis::getIVMultiplier(Value val, scf::ForOp loop) const {

  Value iv = loop.getInductionVar();

  if (val == iv)
    return 1;

  if (!val)
    return std::nullopt;

  if (loop.isDefinedOutsideOfLoop(val))
    return 0;

  Operation *op = val.getDefiningOp();
  if (!op)
    return std::nullopt;

  // ------------------------------------------------------------------
  // Cast / type conversion ops (strip them)
  // ------------------------------------------------------------------

  if (auto cast = dyn_cast<arith::IndexCastOp>(op))
    return getIVMultiplier(cast.getIn(), loop);

  if (auto ext = dyn_cast<arith::ExtSIOp>(op))
    return getIVMultiplier(ext.getIn(), loop);

  if (auto ext = dyn_cast<arith::ExtUIOp>(op))
    return getIVMultiplier(ext.getIn(), loop);

  if (auto trunc = dyn_cast<arith::TruncIOp>(op))
    return getIVMultiplier(trunc.getIn(), loop);

  // ------------------------------------------------------------------
  // Constant
  // ------------------------------------------------------------------

  if (auto cst = dyn_cast<arith::ConstantIndexOp>(op))
    return 0;

  if (matchPattern(val, m_Constant()))
    return 0;

  // ------------------------------------------------------------------
  // Add
  // ------------------------------------------------------------------

  if (auto add = dyn_cast<arith::AddIOp>(op)) {

    auto lhs = getIVMultiplier(add.getLhs(), loop);
    auto rhs = getIVMultiplier(add.getRhs(), loop);

    if (lhs && rhs)
      return *lhs + *rhs;

    return std::nullopt;
  }

  // ------------------------------------------------------------------
  // Sub
  // ------------------------------------------------------------------

  if (auto sub = dyn_cast<arith::SubIOp>(op)) {

    auto lhs = getIVMultiplier(sub.getLhs(), loop);
    auto rhs = getIVMultiplier(sub.getRhs(), loop);

    if (lhs && rhs)
      return *lhs - *rhs;

    return std::nullopt;
  }

  // ------------------------------------------------------------------
  // Mul
  // ------------------------------------------------------------------

  if (auto mul = dyn_cast<arith::MulIOp>(op)) {

    auto lhs = getIVMultiplier(mul.getLhs(), loop);
    auto rhs = getIVMultiplier(mul.getRhs(), loop);

    auto lhsConst = getConstantIntValue(mul.getLhs());
    auto rhsConst = getConstantIntValue(mul.getRhs());

    // iv * C
    if (lhs && rhsConst)
      return (*lhs) * (*rhsConst);

    // C * iv
    if (rhs && lhsConst)
      return (*rhs) * (*lhsConst);

    // constant * constant
    if (lhs && rhs && *lhs == 0 && *rhs == 0)
      return 0;

    return std::nullopt;
  }

  // ------------------------------------------------------------------
  // affine.apply
  // ------------------------------------------------------------------

  if (auto apply = dyn_cast<affine::AffineApplyOp>(op)) {
    return evaluateAffineMultiplier(apply.getAffineMap().getResult(0),
                                    apply.getMapOperands(),
                                    apply.getAffineMap(), loop);
  }

  // ------------------------------------------------------------------
  // Unsupported op
  // ------------------------------------------------------------------

  return std::nullopt;
}

std::optional<int64_t> LoopSliceParallelAnalysis::evaluateAffineMultiplier(
    AffineExpr expr, ValueRange operands, AffineMap map,
    scf::ForOp loop) const {
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    return getIVMultiplier(operands[dimExpr.getPosition()], loop);
  } else if (auto symExpr = dyn_cast<AffineSymbolExpr>(expr)) {
    return getIVMultiplier(operands[map.getNumDims() + symExpr.getPosition()],
                           loop);
  } else if (isa<AffineConstantExpr>(expr)) {
    return 0;
  } else if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    auto lhs = evaluateAffineMultiplier(binExpr.getLHS(), operands, map, loop);
    auto rhs = evaluateAffineMultiplier(binExpr.getRHS(), operands, map, loop);
    if (!lhs || !rhs)
      return std::nullopt;

    switch (binExpr.getKind()) {
    case AffineExprKind::Add:
      return *lhs + *rhs;
    case AffineExprKind::Mul:
      if (*lhs == 0 && *rhs == 0)
        return 0;
      if (*lhs == 0 && isa<AffineConstantExpr>(binExpr.getLHS())) {
        return cast<AffineConstantExpr>(binExpr.getLHS()).getValue() * (*rhs);
      }
      if (*rhs == 0 && isa<AffineConstantExpr>(binExpr.getRHS())) {
        return cast<AffineConstantExpr>(binExpr.getRHS()).getValue() * (*lhs);
      }
      return std::nullopt;
    default:
      return std::nullopt;
    }
  }
  return std::nullopt;
}
