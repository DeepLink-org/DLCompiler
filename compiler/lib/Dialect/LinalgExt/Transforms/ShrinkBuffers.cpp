#include "dicp/Dialect/LinalgExt/Analysis/SliceParallelAnalysis.h"
#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"
#include "dicp/TransformOps/Transforms.h"
#include "dicp/Utils/Utils.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "buffer-shrink"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace dicp;
using namespace LinalgExt;

namespace mlir::dicp::LinalgExt {
#define GEN_PASS_DEF_SHRINKBUFFERS
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace mlir::dicp::LinalgExt

namespace {

/// Create a MemRefType with identity (contiguous) layout, dropping any
/// subview-induced stride information.
static MemRefType createIdentityMemRefType(ArrayRef<int64_t> shape,
                                           Type elementType,
                                           Attribute memorySpace) {
  return MemRefType::get(shape, elementType, MemRefLayoutAttrInterface(),
                         memorySpace);
}

//===----------------------------------------------------------------------===//
// Pattern: ToTensorSliceToSubviewPattern
//===----------------------------------------------------------------------===//

/// Matches `bufferization.to_tensor` operations where **all** uses are
/// `tensor.extract_slice`. It rewrites the sequence by creating a
/// `memref.subview` on the original buffer and converting that subview
/// back to a tensor.
///
/// Source:
///   %t = bufferization.to_tensor %m : memref<...> to tensor<...>
///   %s = tensor.extract_slice %t[...] : tensor<...> to tensor<...>
///
/// Target:
///   %sv = memref.subview %m[...] : memref<...> to memref<...>
///   %s = bufferization.to_tensor %sv : memref<...> to tensor<...>
struct ToTensorSliceToSubviewPattern
    : public OpRewritePattern<bufferization::ToTensorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::ToTensorOp op,
                                PatternRewriter &rewriter) const override {
    if (op->use_empty())
      return failure();

    bool allUsesAreSlices = llvm::all_of(op->getUsers(), [](Operation *user) {
      return isa<tensor::ExtractSliceOp>(user);
    });

    if (!allUsesAreSlices) {
      LLVM_DEBUG(llvm::dbgs() << "[" << DEBUG_TYPE << "] Skipping " << op
                              << ": not all uses are tensor.extract_slice.\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[" << DEBUG_TYPE << "] Matching " << op << " with "
               << std::distance(op->getUsers().begin(), op->getUsers().end())
               << " slice users.\n");

    Value sourceMemref = op.getBuffer();

    // Mutating the user list while iterating requires early-inc traversal.
    for (Operation *user : llvm::make_early_inc_range(op->getUsers())) {
      auto extractOp = cast<tensor::ExtractSliceOp>(user);

      LLVM_DEBUG(llvm::dbgs() << "  Rewriting user: " << *user << "\n");
      rewriter.setInsertionPointAfter(extractOp);
      auto subViewOp = rewriter.create<memref::SubViewOp>(
          extractOp.getLoc(), sourceMemref, extractOp.getMixedOffsets(),
          extractOp.getMixedSizes(), extractOp.getMixedStrides());

      auto newToTensor = rewriter.create<bufferization::ToTensorOp>(
          extractOp.getLoc(), extractOp.getType(), subViewOp,
          /*restrict=*/true, /*writable=*/true);
      rewriter.replaceOp(extractOp, newToTensor);
    }

    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Shared: Partition Legality and Rewrite Utilities
//===----------------------------------------------------------------------===//

/// Emit a detailed debug dump for one discovered partition.
static void logPartitionDetail(StringRef debugTag, unsigned partIdx,
                               const SlicePartition &part) {
  LLVM_DEBUG({
    auto printOfrList = [](ArrayRef<OpFoldResult> ofrs) {
      llvm::dbgs() << "[";
      for (auto [idx, ofr] : llvm::enumerate(ofrs)) {
        if (idx != 0)
          llvm::dbgs() << ", ";
        if (auto cst = getConstantIntValue(ofr)) {
          llvm::dbgs() << *cst;
          continue;
        }
        if (Value value = llvm::dyn_cast<Value>(ofr)) {
          llvm::dbgs() << value;
          continue;
        }
        llvm::dbgs() << "<attr>";
      }
      llvm::dbgs() << "]";
    };

    llvm::dbgs() << "[" DEBUG_TYPE << "] " << debugTag << " partition #"
                 << partIdx << ": type=" << part.sliceType << ", offsets=";
    printOfrList(part.offsets);
    llvm::dbgs() << ", sizes=";
    printOfrList(part.sizes);
    llvm::dbgs() << ", strides=";
    printOfrList(part.strides);
    llvm::dbgs() << "\n";
  });
}

/// Verify that non-loop patterns do not rewrite values that are already
/// transported through control-flow state.
static LogicalResult
verifyChainNoLoopTransport(const DenseSet<Value> &chainValues,
                           StringRef debugTag) {
  LDBG(debugTag << " verifying that " << chainValues.size()
                << " chain values are not loop-transported");

  for (Value chainValue : chainValues) {
    for (OpOperand &use : chainValue.getUses()) {
      Operation *owner = use.getOwner();
      if (isa<scf::YieldOp>(owner)) {
        LDBG(debugTag << " rejected: chain value " << chainValue
                      << " is yielded by " << *owner);
        return failure();
      }
      if (isa<scf::IfOp>(owner)) {
        LDBG(debugTag << " rejected: chain value " << chainValue
                      << " is used by scf.if " << *owner);
        return failure();
      }
      if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
        if (use.getOperandNumber() >= forOp.getNumControlOperands()) {
          LDBG(debugTag << " rejected: chain value " << chainValue
                        << " is used as scf.for init_arg by " << forOp);
          return failure();
        }
      }
    }
  }

  return success();
}

/// Build isolation roots from the extract_slice ops discovered on each
/// partition.
static SmallVector<SmallVector<Value>>
buildPartitionRoots(const DenseMap<Operation *, unsigned> &sliceToPartIdx,
                    size_t numPartitions) {
  SmallVector<SmallVector<Value>> partitionRoots(numPartitions);
  for (auto [op, partIdx] : sliceToPartIdx)
    if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(op))
      partitionRoots[partIdx].push_back(extractOp.getResult());

  LLVM_DEBUG({
    llvm::dbgs() << "[" DEBUG_TYPE << "] built partition roots:";
    for (auto [partIdx, roots] : llvm::enumerate(partitionRoots))
      llvm::dbgs() << " P" << partIdx << "=" << roots.size();
    llvm::dbgs() << "\n";
  });
  return partitionRoots;
}

/// Extract a static partition shape. Prefer the partition result type to keep
/// rank-reducing slices consistent; otherwise fall back to constant sizes.
static FailureOr<SmallVector<int64_t>>
getStaticPartitionShape(const SlicePartition &part) {
  if (part.sliceType.hasStaticShape())
    return SmallVector<int64_t>(part.sliceType.getShape().begin(),
                                part.sliceType.getShape().end());

  if (part.sliceType.getRank() != static_cast<int64_t>(part.sizes.size()))
    return failure();

  SmallVector<int64_t> shape;
  shape.reserve(part.sizes.size());
  for (OpFoldResult size : part.sizes) {
    std::optional<int64_t> cst = getConstantIntValue(size);
    if (!cst)
      return failure();
    shape.push_back(*cst);
  }
  return shape;
}

struct AllocUserInfo {
  SmallVector<memref::SubViewOp> subviews;
  bufferization::ToTensorOp toTensorOp;
};

/// Classify the direct alloc users handled by this pass. The alloc root stays
/// deliberately simple: a set of subviews plus at most one to_tensor.
static FailureOr<AllocUserInfo> classifyAllocUsers(memref::AllocOp allocOp) {
  AllocUserInfo info;
  for (Operation *user : allocOp->getUsers()) {
    bool accepted =
        TypeSwitch<Operation *, bool>(user)
            .Case<memref::SubViewOp>([&](auto subview) {
              info.subviews.push_back(subview);
              return true;
            })
            .Case<bufferization::ToTensorOp>([&](auto toTensorOp) {
              if (info.toTensorOp) {
                LDBG("[SplitAlloc] rejected: multiple to_tensor users on "
                     << allocOp);
                return false;
              }
              info.toTensorOp = toTensorOp;
              return true;
            })
            .Default([&](Operation *op) {
              LDBG("[SplitAlloc] rejected: unsupported direct alloc user: "
                   << *op);
              return false;
            });
    if (!accepted)
      return failure();
  }
  return info;
}

/// Resolve which discovered partition a slice operation belongs to.
///
/// Prefer the exact analysis result when available, but fall back to semantic
/// equivalence on offsets/sizes/strides so that slices created after the
/// initial partition discovery can still be matched.
static std::optional<unsigned>
findPartitionForSliceOp(ArrayRef<SlicePartition> partitions,
                        const DenseMap<Operation *, unsigned> &sliceToPartIdx,
                        Operation *sliceOp) {
  if (auto it = sliceToPartIdx.find(sliceOp); it != sliceToPartIdx.end())
    return it->second;

  if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(sliceOp)) {
    int matched = findMatchingPartition(partitions, extractOp.getMixedOffsets(),
                                        extractOp.getMixedSizes(),
                                        extractOp.getMixedStrides());
    if (matched >= 0)
      return static_cast<unsigned>(matched);
  }

  if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(sliceOp)) {
    int matched = findMatchingPartition(partitions, insertOp.getMixedOffsets(),
                                        insertOp.getMixedSizes(),
                                        insertOp.getMixedStrides());
    if (matched >= 0)
      return static_cast<unsigned>(matched);
  }

  return std::nullopt;
}

static FailureOr<SmallVector<Type>>
inferPartitionResultTypes(Operation *op, ArrayRef<Value> partitionOperands) {
  if (op->getNumResults() != 1 ||
      !isa<RankedTensorType>(op->getResult(0).getType()))
    return failure();

  // Special handling for linalg.reduce: result type is determined by input
  // shape and reduction dimensions, NOT by init operand. In the "no reduction
  // dim slicing" invariant, result shape remains unchanged from the original.
  if (auto reduceOp = dyn_cast<linalg::ReduceOp>(op)) {
    auto origResultType =
        cast<RankedTensorType>(reduceOp.getResult(0).getType());

    // Sanity check: all partition init operands must match original result type
    for (OpOperand &initOperand : reduceOp.getDpsInitsMutable()) {
      unsigned operandNumber = initOperand.getOperandNumber();
      if (operandNumber >= partitionOperands.size())
        return failure();

      auto initType = dyn_cast<RankedTensorType>(
          partitionOperands[operandNumber].getType());
      if (!initType || initType != origResultType)
        return failure(); // Init type mismatch - invalid partition
    }

    // Return original result types unchanged
    return SmallVector<Type>(reduceOp->getResultTypes().begin(),
                             reduceOp->getResultTypes().end());
  }

  // Standard DPS path for other ops where result type can be derived from init
  if (auto dstStyle = dyn_cast<DestinationStyleOpInterface>(op)) {
    SmallVector<Type> resultTypes(op->getNumResults());
    bool hasTiedResult = false;
    for (OpOperand &initOperand : dstStyle.getDpsInitsMutable()) {
      OpResult tiedResult = dstStyle.getTiedOpResult(&initOperand);
      if (!tiedResult)
        continue;
      unsigned operandNumber = initOperand.getOperandNumber();
      if (operandNumber >= partitionOperands.size() ||
          !isa<RankedTensorType>(partitionOperands[operandNumber].getType()))
        return failure();
      resultTypes[tiedResult.getResultNumber()] =
          partitionOperands[operandNumber].getType();
      hasTiedResult = true;
    }
    if (hasTiedResult)
      return resultTypes;
  }

  // Fallback: infer from first ranked tensor operand
  auto origResultType = cast<RankedTensorType>(op->getResult(0).getType());
  RankedTensorType partitionShapeType;
  for (Value operand : partitionOperands) {
    auto rankedType = dyn_cast<RankedTensorType>(operand.getType());
    if (!rankedType)
      continue;
    partitionShapeType = rankedType;
    break;
  }
  if (!partitionShapeType)
    return failure();

  SmallVector<Type> resultTypes;
  resultTypes.push_back(RankedTensorType::get(partitionShapeType.getShape(),
                                              origResultType.getElementType(),
                                              origResultType.getEncoding()));
  return resultTypes;
}

/// Create a partition-local clone of a supported pure tensor op. DPS-style ops
/// derive their result types from the partition-local init operands.
static FailureOr<Value>
clonePartitionPropagatableOpForPartition(RewriterBase &rewriter, Operation *op,
                                         ArrayRef<Value> partitionOperands) {
  auto resultTypes = inferPartitionResultTypes(op, partitionOperands);
  if (failed(resultTypes))
    return failure();

  Operation *newOp = clone(rewriter, op, *resultTypes, partitionOperands);
  if (newOp->getNumResults() != 1)
    return failure();
  return newOp->getResult(0);
}

/// Try to derive partition values for `op` from already-partitioned operands.
static FailureOr<SmallVector<Value>>
materializePartitionableOpPartitions(RewriterBase &rewriter, Operation *op,
                                     DenseMap<Value, SmallVector<Value>> &pvm,
                                     StringRef debugTag) {
  if (!isPartitionPropagatableTensorOp(op))
    return failure();

  if (auto dstStyle = dyn_cast<DestinationStyleOpInterface>(op)) {
    bool hasPartitionedInit = llvm::any_of(
        dstStyle.getDpsInits(), [&](Value init) { return pvm.contains(init); });
    if (op->getNumResults() != 0 && !hasPartitionedInit) {
      LDBG(debugTag << " cannot partition-propagate " << *op
                    << ": destination-style result is not rooted in a "
                       "partition-local init");
      return failure();
    }
  }

  unsigned numPartitions = 0;
  bool sawPartitionedTensorOperand = false;
  for (Value operand : op->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType()))
      continue;
    auto it = pvm.find(operand);
    if (it == pvm.end())
      continue;
    sawPartitionedTensorOperand = true;
    if (numPartitions == 0) {
      numPartitions = it->second.size();
      continue;
    }
    if (numPartitions != it->second.size()) {
      LDBG(debugTag << " cannot partition-propagate " << *op
                    << ": inconsistent partition counts");
      return failure();
    }
  }

  if (!sawPartitionedTensorOperand || numPartitions == 0)
    return failure();

  SmallVector<Value> partResults;
  partResults.reserve(numPartitions);
  rewriter.setInsertionPoint(op);
  for (unsigned partIdx = 0; partIdx < numPartitions; ++partIdx) {
    SmallVector<Value> partOperands;
    partOperands.reserve(op->getNumOperands());
    for (Value operand : op->getOperands()) {
      if (isa<RankedTensorType>(operand.getType()) && pvm.contains(operand))
        partOperands.push_back(pvm.lookup(operand)[partIdx]);
      else
        partOperands.push_back(operand);
    }
    auto cloned =
        clonePartitionPropagatableOpForPartition(rewriter, op, partOperands);
    if (failed(cloned)) {
      LDBG(debugTag << " failed to clone partition-local tensor op: " << *op);
      return failure();
    }
    partResults.push_back(*cloned);
  }
  return partResults;
}

/// Walk `block` recursively and collect the extract/insert rewrites driven by
/// the current partition-value map.
static LogicalResult collectPvmRewrites(
    PatternRewriter &rewriter, Block *block,
    ArrayRef<SlicePartition> partitions,
    const DenseMap<Operation *, unsigned> &sliceToPartIdx,
    DenseMap<Value, SmallVector<Value>> &pvm,
    SmallVector<std::pair<tensor::ExtractSliceOp, Value>> &extractRepls,
    SmallVector<Operation *> &opsToTryErase,
    DenseSet<Operation *> &handledSliceOps, StringRef debugTag) {
  for (Operation &op : *block) {
    if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(&op)) {
      auto pvIt = pvm.find(extractOp.getSource());
      if (pvIt != pvm.end()) {
        auto partIdx = findPartitionForSliceOp(partitions, sliceToPartIdx, &op);
        if (!partIdx || *partIdx >= pvIt->second.size()) {
          LDBG(debugTag << " failed: cannot match extract_slice to partition "
                        << *extractOp);
          return failure();
        }
        extractRepls.push_back({extractOp, pvIt->second[*partIdx]});
        if (sliceToPartIdx.contains(&op))
          handledSliceOps.insert(&op);
        LDBG(debugTag << " rewiring extract_slice " << *extractOp
                      << " to partition value #" << *partIdx);
        continue;
      }
    }

    if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(&op)) {
      auto pvIt = pvm.find(insertOp.getDest());
      if (pvIt != pvm.end()) {
        auto partIdx = findPartitionForSliceOp(partitions, sliceToPartIdx, &op);
        if (!partIdx || *partIdx >= pvIt->second.size()) {
          LDBG(debugTag << " failed: cannot match insert_slice to partition "
                        << *insertOp);
          return failure();
        }
        SmallVector<Value> forwardedValues(pvIt->second);
        forwardedValues[*partIdx] = insertOp.getSource();
        pvm[insertOp.getResult()] = std::move(forwardedValues);
        opsToTryErase.push_back(insertOp);
        if (sliceToPartIdx.contains(&op))
          handledSliceOps.insert(&op);
        LDBG(debugTag << " forwarding insert_slice " << *insertOp
                      << " into partition value #" << *partIdx);
        continue;
      }
    }

    auto partResults =
        materializePartitionableOpPartitions(rewriter, &op, pvm, debugTag);
    if (succeeded(partResults)) {
      pvm[op.getResult(0)] = std::move(*partResults);
      opsToTryErase.push_back(&op);
      LDBG(debugTag << " partition-propagated tensor op " << op);
      continue;
    }

    for (Region &region : op.getRegions())
      for (Block &nested : region)
        if (failed(collectPvmRewrites(
                rewriter, &nested, partitions, sliceToPartIdx, pvm,
                extractRepls, opsToTryErase, handledSliceOps, debugTag)))
          return failure();
  }

  return success();
}

/// Apply a partition-value-map rewrite rooted at `chainRoot`.
static LogicalResult
applyPvmRewrite(PatternRewriter &rewriter, Block *block, Value chainRoot,
                ArrayRef<Value> partValues, ArrayRef<SlicePartition> partitions,
                const DenseMap<Operation *, unsigned> &sliceToPartIdx,
                StringRef debugTag) {
  LDBG(debugTag << " applying PVM rewrite for root " << chainRoot << " with "
                << partValues.size() << " partition values");

  DenseMap<Value, SmallVector<Value>> pvm;
  pvm[chainRoot] = SmallVector<Value>(partValues);

  SmallVector<std::pair<tensor::ExtractSliceOp, Value>> extractRepls;
  SmallVector<Operation *> opsToTryErase;
  DenseSet<Operation *> handledSliceOps;

  if (failed(collectPvmRewrites(rewriter, block, partitions, sliceToPartIdx,
                                pvm, extractRepls, opsToTryErase,
                                handledSliceOps, debugTag)))
    return failure();

  for (auto [op, partIdx] : sliceToPartIdx) {
    (void)partIdx;
    if (!handledSliceOps.contains(op)) {
      LDBG(debugTag << " failed: analyzed slice op was not rewritten: " << *op);
      return failure();
    }
  }

  for (auto [extractOp, replacement] : extractRepls)
    rewriter.replaceOp(extractOp, replacement);
  for (auto it = opsToTryErase.rbegin(); it != opsToTryErase.rend(); ++it)
    if ((*it)->use_empty())
      rewriter.eraseOp(*it);

  LDBG(debugTag << " PVM rewrite complete: replaced " << extractRepls.size()
                << " extract_slice ops and cleaned " << opsToTryErase.size()
                << " candidate ops");
  return success();
}

//===----------------------------------------------------------------------===//
// tensor.empty splitting
//===----------------------------------------------------------------------===//

static LogicalResult rewriteDisjointTensorEmpty(tensor::EmptyOp emptyOp,
                                                PatternRewriter &rewriter) {
  SmallVector<OffsetSizeAndStrideOpInterface> slices;
  slices.reserve(static_cast<size_t>(
      std::distance(emptyOp->getUsers().begin(), emptyOp->getUsers().end())));
  for (Operation *user : emptyOp->getUsers()) {
    auto extractOp = dyn_cast<tensor::ExtractSliceOp>(user);
    if (!extractOp) {
      LDBG("[SplitTensorEmpty] rejected: unexpected non-extract user "
           << *user);
      return failure();
    }
    slices.push_back(
        cast<OffsetSizeAndStrideOpInterface>(extractOp.getOperation()));
  }

  auto partitionInfo = buildDirectSlicePartitionInfo(slices);
  if (failed(partitionInfo))
    return failure();
  if (partitionInfo->partitions.size() < 2) {
    LDBG("[SplitTensorEmpty] rejected: only "
         << partitionInfo->partitions.size() << " semantic partition");
    return failure();
  }
  for (auto [partIdx, part] : llvm::enumerate(partitionInfo->partitions))
    logPartitionDetail("[SplitTensorEmpty]", partIdx, part);
  if (failed(verifyDirectSliceNoCallEscape(slices)))
    return failure();
  if (failed(verifyDirectSlicePartitionsDisjoint(partitionInfo->partitions))) {
    return failure();
  }

  LDBG("[SplitTensorEmpty] rewriting disjoint tensor.empty "
       << emptyOp << " into " << partitionInfo->partitions.size()
       << " semantic partitions");

  SmallVector<Value> partitionValues;
  partitionValues.reserve(partitionInfo->partitions.size());
  rewriter.setInsertionPoint(emptyOp);
  for (auto [partIdx, part] : llvm::enumerate(partitionInfo->partitions)) {
    auto shape = getStaticPartitionShape(part);
    if (failed(shape)) {
      LDBG("[SplitTensorEmpty] rejected: partition #" << partIdx
                                                      << " does not have a "
                                                         "static shape");
      return failure();
    }
    auto newEmpty = rewriter.create<tensor::EmptyOp>(
        emptyOp.getLoc(), *shape, part.sliceType.getElementType());
    partitionValues.push_back(newEmpty);
    LDBG("[SplitTensorEmpty] created partition tensor #"
         << partIdx << " with shape rank " << shape->size());
  }

  for (Operation *user : llvm::make_early_inc_range(emptyOp->getUsers())) {
    auto extractOp = cast<tensor::ExtractSliceOp>(user);
    unsigned partIdx =
        partitionInfo->sliceToPartIdx.lookup(extractOp.getOperation());
    LDBG("[SplitTensorEmpty] replacing " << extractOp << " with partition #"
                                         << partIdx);
    rewriter.replaceOp(extractOp, partitionValues[partIdx]);
  }

  rewriter.eraseOp(emptyOp);
  return success();
}

static LogicalResult rewritePartitionedTensorEmpty(tensor::EmptyOp emptyOp,
                                                   PatternRewriter &rewriter) {
  InsertSliceChainAnalysis chainAnalysis;
  auto chain =
      chainAnalysis.analyze(emptyOp.getResult(), [](Operation *user) -> bool {
        return isa<scf::ForOp>(user);
      });
  if (failed(chain)) {
    LDBG("[SplitTensorEmpty] chain analysis failed for " << emptyOp);
    return failure();
  }

  LDBG("[SplitTensorEmpty] discovered " << chain->partitions.size()
                                        << " partitions for " << emptyOp);
  for (auto [partIdx, part] : llvm::enumerate(chain->partitions))
    logPartitionDetail("[SplitTensorEmpty]", partIdx, part);

  if (failed(
          verifyChainNoLoopTransport(chain->chainValues, "[SplitTensorEmpty]")))
    return failure();

  SmallVector<SmallVector<Value>> partitionRoots =
      buildPartitionRoots(chain->sliceToPartIdx, chain->partitions.size());
  DenseSet<Value> forbiddenWriteTargets;
  if (failed(verifyPartitionIsolation(partitionRoots, forbiddenWriteTargets))) {
    LDBG("[SplitTensorEmpty] rejected: partition isolation failed for "
         << emptyOp);
    return failure();
  }

  SmallVector<Value> partitionValues;
  partitionValues.reserve(chain->partitions.size());
  rewriter.setInsertionPoint(emptyOp);
  for (auto [partIdx, part] : llvm::enumerate(chain->partitions)) {
    auto shape = getStaticPartitionShape(part);
    if (failed(shape)) {
      LDBG("[SplitTensorEmpty] rejected: partition #" << partIdx
                                                      << " does not have a "
                                                         "static shape");
      return failure();
    }
    partitionValues.push_back(rewriter.create<tensor::EmptyOp>(
        emptyOp.getLoc(), *shape, emptyOp.getType().getElementType()));
    LDBG("[SplitTensorEmpty] created partition tensor #"
         << partIdx << " with shape rank " << shape->size());
  }

  if (failed(applyPvmRewrite(rewriter, emptyOp->getBlock(), emptyOp.getResult(),
                             partitionValues, chain->partitions,
                             chain->sliceToPartIdx, "[SplitTensorEmpty]")))
    return failure();

  if (emptyOp->use_empty()) {
    LDBG("[SplitTensorEmpty] erasing dead root tensor.empty " << emptyOp);
    rewriter.eraseOp(emptyOp);
  }
  return success();
}

/// Dispatch between the fast disjoint-slice case and the insert_slice-chain
/// case for a tensor root.
struct SplitTensorEmptyPattern : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::EmptyOp emptyOp,
                                PatternRewriter &rewriter) const override {
    if (emptyOp->use_empty())
      return failure();

    bool allUsesAreExtractSlices =
        llvm::all_of(emptyOp->getUsers(), [](Operation *user) {
          return isa<tensor::ExtractSliceOp>(user);
        });
    if (allUsesAreExtractSlices)
      return rewriteDisjointTensorEmpty(emptyOp, rewriter);

    return rewritePartitionedTensorEmpty(emptyOp, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// memref.alloc splitting
//===----------------------------------------------------------------------===//

static LogicalResult rewriteDisjointAlloc(memref::AllocOp allocOp,
                                          PatternRewriter &rewriter) {
  FailureOr<AllocUserInfo> userInfo = classifyAllocUsers(allocOp);
  if (failed(userInfo) || userInfo->toTensorOp) {
    LDBG("[SplitAlloc] rejected: expected pure memref.subview users for "
         << allocOp);
    return failure();
  }

  SmallVector<OffsetSizeAndStrideOpInterface> slices;
  slices.reserve(userInfo->subviews.size());
  for (memref::SubViewOp subview : userInfo->subviews)
    slices.push_back(
        cast<OffsetSizeAndStrideOpInterface>(subview.getOperation()));

  auto partitionInfo = buildDirectSlicePartitionInfo(slices);
  if (failed(partitionInfo))
    return failure();
  if (partitionInfo->partitions.size() < 2) {
    LDBG("[SplitAlloc] rejected: only " << partitionInfo->partitions.size()
                                        << " semantic partition");
    return failure();
  }
  for (auto [partIdx, part] : llvm::enumerate(partitionInfo->partitions))
    logPartitionDetail("[SplitAlloc]", partIdx, part);
  if (failed(verifyDirectSliceNoCallEscape(slices)))
    return failure();
  if (failed(verifyDirectSlicePartitionsDisjoint(partitionInfo->partitions))) {
    return failure();
  }

  LDBG("[SplitAlloc] rewriting disjoint alloc "
       << allocOp << " into " << partitionInfo->partitions.size()
       << " semantic partitions");

  auto allocType = allocOp.getType();
  SmallVector<Value> subAllocs;
  subAllocs.reserve(partitionInfo->partitions.size());
  rewriter.setInsertionPoint(allocOp);
  for (auto [partIdx, part] : llvm::enumerate(partitionInfo->partitions)) {
    auto shape = getStaticPartitionShape(part);
    if (failed(shape)) {
      LDBG("[SplitAlloc] rejected: partition #" << partIdx
                                                << " does not have a static "
                                                   "shape");
      return failure();
    }

    auto subAllocType = createIdentityMemRefType(
        *shape, allocType.getElementType(), allocType.getMemorySpace());
    auto subAlloc =
        rewriter.create<memref::AllocOp>(allocOp.getLoc(), subAllocType);
    propagateDicpAttributes(allocOp, subAlloc);
    subAllocs.push_back(subAlloc);
    LDBG("[SplitAlloc] created partition alloc #" << partIdx << " of type "
                                                  << subAllocType);
  }

  for (memref::SubViewOp subview : userInfo->subviews) {
    unsigned partIdx =
        partitionInfo->sliceToPartIdx.lookup(subview.getOperation());
    LDBG("[SplitAlloc] replacing " << subview << " with partition #"
                                   << partIdx);
    rewriter.replaceOp(subview, subAllocs[partIdx]);
  }

  rewriter.eraseOp(allocOp);
  return success();
}

static LogicalResult rewritePartitionedAlloc(memref::AllocOp allocOp,
                                             PatternRewriter &rewriter) {
  FailureOr<AllocUserInfo> userInfo = classifyAllocUsers(allocOp);
  if (failed(userInfo))
    return failure();
  if (!userInfo->toTensorOp)
    return failure();

  bufferization::ToTensorOp toTensorOp = userInfo->toTensorOp;
  if (!llvm::all_of(toTensorOp->getUsers(), [](Operation *user) {
        return isa<tensor::ExtractSliceOp, tensor::InsertSliceOp>(user) ||
               isPartitionPropagatableTensorOp(user);
      })) {
    LDBG("[SplitAlloc] rejected: to_tensor has non-slice users for "
         << allocOp);
    return failure();
  }

  InsertSliceChainAnalysis chainAnalysis;
  auto chain = chainAnalysis.analyze(
      toTensorOp.getResult(),
      [](Operation *user) -> bool { return isa<scf::ForOp>(user); });
  if (failed(chain)) {
    LDBG("[SplitAlloc] chain analysis failed for " << allocOp);
    return failure();
  }

  LDBG("[SplitAlloc] discovered "
       << chain->partitions.size() << " partitions for " << allocOp << " with "
       << userInfo->subviews.size() << " memref.subview users");
  for (auto [partIdx, part] : llvm::enumerate(chain->partitions))
    logPartitionDetail("[SplitAlloc]", partIdx, part);

  if (failed(verifyChainNoLoopTransport(chain->chainValues, "[SplitAlloc]")))
    return failure();

  DenseMap<Operation *, unsigned> subviewToPartIdx;
  for (memref::SubViewOp subview : userInfo->subviews) {
    int matchedPart = findMatchingPartition(
        chain->partitions, subview.getMixedOffsets(), subview.getMixedSizes(),
        subview.getMixedStrides());
    if (matchedPart < 0) {
      LDBG("[SplitAlloc] rejected: subview does not match any partition: "
           << subview);
      return failure();
    }
    subviewToPartIdx[subview.getOperation()] =
        static_cast<unsigned>(matchedPart);
    LDBG("[SplitAlloc] matched subview " << subview << " to partition #"
                                         << matchedPart);
  }

  SmallVector<SmallVector<Value>> partitionRoots =
      buildPartitionRoots(chain->sliceToPartIdx, chain->partitions.size());
  DenseSet<Value> forbiddenWriteTargets;
  forbiddenWriteTargets.insert(allocOp.getResult());
  for (memref::SubViewOp subview : userInfo->subviews)
    forbiddenWriteTargets.insert(subview.getResult());
  if (failed(verifyPartitionIsolation(partitionRoots, forbiddenWriteTargets))) {
    LDBG("[SplitAlloc] rejected: partition isolation failed for " << allocOp);
    return failure();
  }

  auto allocType = allocOp.getType();
  SmallVector<Value> subAllocs;
  subAllocs.reserve(chain->partitions.size());
  rewriter.setInsertionPoint(allocOp);
  for (auto [partIdx, part] : llvm::enumerate(chain->partitions)) {
    auto shape = getStaticPartitionShape(part);
    if (failed(shape)) {
      LDBG("[SplitAlloc] rejected: partition #" << partIdx
                                                << " does not have a static "
                                                   "shape");
      return failure();
    }
    auto subType = createIdentityMemRefType(*shape, allocType.getElementType(),
                                            allocType.getMemorySpace());
    auto subAlloc = rewriter.create<memref::AllocOp>(allocOp.getLoc(), subType);
    propagateDicpAttributes(allocOp, subAlloc);
    subAllocs.push_back(subAlloc);
    LDBG("[SplitAlloc] created sub-alloc #" << partIdx << " of type "
                                            << subType);
  }

  for (memref::SubViewOp subview : userInfo->subviews) {
    unsigned partIdx = subviewToPartIdx.lookup(subview.getOperation());
    rewriter.replaceOp(subview, subAllocs[partIdx]);
  }

  SmallVector<Value> partitionTensors;
  partitionTensors.reserve(chain->partitions.size());
  for (auto [partIdx, part] : llvm::enumerate(chain->partitions)) {
    rewriter.setInsertionPointAfter(subAllocs[partIdx].getDefiningOp());
    partitionTensors.push_back(rewriter.create<bufferization::ToTensorOp>(
        allocOp.getLoc(), part.sliceType, subAllocs[partIdx],
        /*restrict=*/true, /*writable=*/true));
    LDBG("[SplitAlloc] created per-partition to_tensor #" << partIdx << " : "
                                                          << part.sliceType);
  }

  if (failed(applyPvmRewrite(rewriter, toTensorOp->getBlock(),
                             toTensorOp.getResult(), partitionTensors,
                             chain->partitions, chain->sliceToPartIdx,
                             "[SplitAlloc]")))
    return failure();

  if (toTensorOp->use_empty()) {
    LDBG("[SplitAlloc] erasing dead to_tensor " << toTensorOp);
    rewriter.eraseOp(toTensorOp);
  }
  if (allocOp->use_empty()) {
    LDBG("[SplitAlloc] erasing dead root alloc " << allocOp);
    rewriter.eraseOp(allocOp);
  }
  return success();
}

/// Dispatch between the pure-subview case and the mixed memref/tensor chain
/// case for an alloc root.
struct SplitAllocPattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    if (allocOp->use_empty())
      return failure();

    FailureOr<AllocUserInfo> userInfo = classifyAllocUsers(allocOp);
    if (failed(userInfo))
      return failure();

    if (!userInfo->toTensorOp)
      return rewriteDisjointAlloc(allocOp, rewriter);
    return rewritePartitionedAlloc(allocOp, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// Loop Iter-Arg Decomposition
//===----------------------------------------------------------------------===//
//
// Identifies scf.for loops whose tensor iter_args are accessed exclusively
// through disjoint static partitions (extract_slice / insert_slice), and
// splits each such iter_arg into multiple partition-scoped iter_args.
//
// High-level algorithm (per loop, iterated to fixed-point, inner-to-outer):
//   1. Partition Discovery  — InsertSliceChainAnalysis on body block arg
//   2. Legality Analysis    — disjointness, isolation, yield membership
//   3. Init Materialization — trace insert_slice chain to find partition inits
//   4. Loop Rewrite         — new ForOp + partValueMap body rewrite
//   5. Exit Rewrite         — elide/reconstruct post-loop users
//===----------------------------------------------------------------------===//

/// Holds the decomposition plan for a single tensor iter_arg.
struct IterArgDecompPlan {
  unsigned origIdx;
  SmallVector<SlicePartition> partitions;
  DenseMap<Operation *, unsigned> sliceToPartIdx;
  DenseSet<Value> chainValues;
  SmallVector<Value> partInits; // one per partition
};

//===--- Stage 1: Partition Discovery -------------------------------------===//

/// Discover disjoint partitions for a tensor iter_arg's body block argument.
static FailureOr<InsertSliceChainAnalysis::Result>
discoverBodyPartitions(BlockArgument bodyArg) {
  LDBG("[Decompose] Stage 1: discovering partitions for body arg #"
       << bodyArg.getArgNumber() << " type=" << bodyArg.getType());

  InsertSliceChainAnalysis analysis;
  auto result = analysis.analyze(
      bodyArg, [](Operation *user) -> bool { return isa<scf::YieldOp>(user); });

  if (succeeded(result)) {
    LDBG("  -> " << result->partitions.size() << " partitions found");
    for (auto [i, p] : llvm::enumerate(result->partitions))
      LDBG("     P" << i << ": type=" << p.sliceType);
  }
  return result;
}

//===--- Stage 2: Legality Analysis ---------------------------------------===//

/// Verify that decomposition is safe for the given chain analysis result.
static LogicalResult
verifyDecompLegality(const InsertSliceChainAnalysis::Result &chain,
                     BlockArgument bodyArg, scf::ForOp forOp) {
  unsigned argIdx = bodyArg.getArgNumber() - 1; // skip IV
  LDBG("[Decompose] Stage 2: legality for arg #" << argIdx);

  // The yielded value must be a recognized successor on the chain.
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!chain.chainValues.contains(yieldOp.getOperand(argIdx))) {
    LDBG("  [REJECT] yield operand not in chain");
    return failure();
  }

  // Every chain user must remain resolvable during body cloning.
  for (Value cv : chain.chainValues) {
    for (Operation *user : cv.getUsers()) {
      if (isa<scf::YieldOp>(user))
        continue;
      if (chain.sliceToPartIdx.contains(user))
        continue;
      // Reads of a chain value through insert_slice source operands are not
      // representable in the per-partition rewrite and must be rejected.
      if (auto ins = dyn_cast<tensor::InsertSliceOp>(user)) {
        if (ins.getSource() == cv && ins.getDest() != cv) {
          LDBG("  [REJECT] chain value used as insert_slice source: " << *user);
          return failure();
        }
      }
      LDBG("  [REJECT] chain value has unresolvable user: " << *user);
      return failure();
    }
  }

  // Check 3: partition isolation.
  SmallVector<SmallVector<Value>> roots =
      buildPartitionRoots(chain.sliceToPartIdx, chain.partitions.size());
  DenseSet<Value> forbidden;
  if (failed(verifyPartitionIsolation(roots, forbidden))) {
    LDBG("  [REJECT] isolation check failed");
    return failure();
  }

  LDBG("  -> legality OK");
  return success();
}

//===--- Stage 3: Init Materialization ------------------------------------===//

/// Return mixed slice parameters that are available at `anchor`.
///
/// Dynamic values are accepted when they dominate the anchor operation;
/// constants are forwarded as-is.
static FailureOr<SmallVector<OpFoldResult>>
getAvailableSliceParams(ArrayRef<OpFoldResult> params, Operation *anchor,
                        StringRef debugTag, StringRef paramKind) {
  DominanceInfo domInfo(anchor->getParentOp());
  SmallVector<OpFoldResult> result;
  result.reserve(params.size());

  for (auto [idx, param] : llvm::enumerate(params)) {
    if (auto attr = llvm::dyn_cast_if_present<Attribute>(param)) {
      result.push_back(attr);
      continue;
    }

    Value value = cast<Value>(param);
    if (!domInfo.dominates(value, anchor)) {
      LDBG(debugTag << " rejected: " << paramKind << "[" << idx
                    << "] is not available at " << *anchor << ": " << value);
      return failure();
    }
    result.push_back(value);
  }

  return result;
}

/// Recursively trace the loop init producer chain and materialize a
/// partition-scoped init value. Reuse an existing SSA value whenever possible.
static FailureOr<Value> materializePartInit(OpBuilder &b, Value init,
                                            const SlicePartition &part,
                                            unsigned pi,
                                            Operation *availabilityAnchor) {
  // Rule: tensor.empty -> smaller empty.
  if (auto empty = init.getDefiningOp<tensor::EmptyOp>()) {
    LDBG("  [mat] P" << pi << ": tensor.empty -> new empty " << part.sliceType);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointAfter(empty);
    return b
        .create<tensor::EmptyOp>(empty.getLoc(), part.sliceType.getShape(),
                                 part.sliceType.getElementType())
        .getResult();
  }

  // Rule: insert_slice(src, dst, region).
  if (auto ins = init.getDefiningOp<tensor::InsertSliceOp>()) {
    auto off = ins.getMixedOffsets();
    auto sz = ins.getMixedSizes();
    auto st = ins.getMixedStrides();
    // Exact match -> return src.
    SmallVector<SlicePartition> single = {part};
    if (findMatchingPartition(single, off, sz, st) >= 0) {
      LDBG("  [mat] P" << pi << ": insert_slice exact match -> src");
      return ins.getSource();
    }
    // Disjoint -> recurse on dest.
    for (size_t d = 0; d < off.size() && d < part.offsets.size(); ++d) {
      if (areRangesDisjoint1D(off[d], sz[d], st[d], part.offsets[d],
                              part.sizes[d], part.strides[d])) {
        LDBG("  [mat] P" << pi << ": insert_slice disjoint -> recurse dest");
        return materializePartInit(b, ins.getDest(), part, pi,
                                   availabilityAnchor);
      }
    }
    LDBG("  [mat] P" << pi << ": insert_slice overlaps but not exact -> FAIL");
    return failure();
  }

  // Rule: linalg.fill — reuse if result type matches, else recreate.
  if (auto fill = init.getDefiningOp<linalg::FillOp>()) {
    if (fill.getResult(0).getType() == part.sliceType) {
      LDBG("  [mat] P" << pi << ": fill already produces partition type");
      return fill.getResult(0);
    }
    auto dst = materializePartInit(b, fill.getDpsInits()[0], part, pi,
                                   availabilityAnchor);
    if (failed(dst))
      return failure();
    LDBG("  [mat] P" << pi << ": fill -> new fill on partition output");
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointAfter(fill);
    return b.create<linalg::FillOp>(fill.getLoc(), fill.getInputs(), *dst)
        .getResult(0);
  }

  // Rule: extract_slice that exactly matches the partition.
  if (auto ext = init.getDefiningOp<tensor::ExtractSliceOp>()) {
    if (ext.getResultType() == part.sliceType) {
      SmallVector<SlicePartition> single = {part};
      if (findMatchingPartition(single, ext.getMixedOffsets(),
                                ext.getMixedSizes(),
                                ext.getMixedStrides()) >= 0) {
        LDBG("  [mat] P" << pi << ": extract_slice exact match");
        return ext.getResult();
      }
    }
  }

  // Fallback: materialize the partition with a standalone extract_slice.
  // Dynamic slice parameters are supported when they dominate the loop that is
  // being rewritten; body-local values remain illegal here.
  if (isa<RankedTensorType>(init.getType())) {
    auto offsets = getAvailableSliceParams(part.offsets, availabilityAnchor,
                                           "  [mat]", "offset");
    auto sizes = getAvailableSliceParams(part.sizes, availabilityAnchor,
                                         "  [mat]", "size");
    auto strides = getAvailableSliceParams(part.strides, availabilityAnchor,
                                           "  [mat]", "stride");
    if (failed(offsets) || failed(sizes) || failed(strides)) {
      LDBG("  [mat] P" << pi
                       << ": fallback rejected — slice params unavailable");
      return failure();
    }
    LDBG("  [mat] P" << pi << ": fallback -> extract_slice from init");
    OpBuilder::InsertionGuard g(b);
    if (auto *defOp = init.getDefiningOp())
      b.setInsertionPointAfter(defOp);
    else
      b.setInsertionPoint(availabilityAnchor);
    return b
        .create<tensor::ExtractSliceOp>(init.getLoc(), part.sliceType, init,
                                        *offsets, *sizes, *strides)
        .getResult();
  }

  LDBG("  [mat] P" << pi << ": unsupported producer -> FAIL");
  return failure();
}

//===--- Stage 4+5: Loop Rewrite (signature + body) -----------------------===//

/// Rewrite a loop by splitting selected tensor iter_args and cloning the body
/// with a partition-value map.
static FailureOr<scf::ForOp> rewriteLoop(IRRewriter &rewriter, scf::ForOp forOp,
                                         ArrayRef<IterArgDecompPlan> plans) {
  Location loc = forOp.getLoc();
  unsigned numOrig = forOp.getNumRegionIterArgs();
  Block *oldBody = forOp.getBody();

  LDBG("[Decompose] Stage 4: rewriting loop at " << loc);

  // Build mapping: origArgIdx -> planIdx.
  DenseMap<unsigned, unsigned> argToPlan;
  for (auto [i, p] : llvm::enumerate(plans))
    argToPlan[p.origIdx] = i;

  // Build new init args and record each orig arg's start position.
  SmallVector<Value> newInits;
  SmallVector<int> origStart(numOrig);
  for (unsigned i = 0; i < numOrig; ++i) {
    origStart[i] = static_cast<int>(newInits.size());
    auto it = argToPlan.find(i);
    if (it != argToPlan.end())
      newInits.append(plans[it->second].partInits);
    else
      newInits.push_back(forOp.getInitArgs()[i]);
  }

  // Create new loop.
  rewriter.setInsertionPoint(forOp);
  auto newLoop = rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                             forOp.getUpperBound(),
                                             forOp.getStep(), newInits);
  // Preserve attributes (e.g. tt.num_stages).
  for (auto attr : forOp->getAttrs())
    if (attr.getName() != "operandSegmentSizes")
      newLoop->setAttr(attr.getName(), attr.getValue());

  Block *newBody = newLoop.getBody();
  // Erase the builder-provided terminator before cloning the rewritten body.
  if (!newBody->empty() && newBody->back().hasTrait<OpTrait::IsTerminator>())
    rewriter.eraseOp(&newBody->back());

  LDBG("  new loop: " << newLoop.getNumRegionIterArgs() << " iter_args (was "
                      << numOrig << ")");

  // Clone the loop body while forwarding partition values through the chain.
  IRMapping mapping;
  mapping.map(oldBody->getArgument(0), newBody->getArgument(0)); // IV

  // partValueMap[oldChainValue] = [partVal_0, partVal_1, ...]
  DenseMap<Value, SmallVector<Value>> pvm;
  DenseMap<Value, unsigned> pvmPlanIdx;

  for (unsigned i = 0; i < numOrig; ++i) {
    BlockArgument oldArg = oldBody->getArgument(i + 1);
    int start = origStart[i];
    auto it = argToPlan.find(i);
    if (it != argToPlan.end()) {
      auto &plan = plans[it->second];
      SmallVector<Value> parts;
      for (unsigned p = 0; p < plan.partitions.size(); ++p)
        parts.push_back(newBody->getArgument(start + p + 1)); // +1 for IV
      pvm[oldArg] = std::move(parts);
      pvmPlanIdx[oldArg] = it->second;
    } else {
      mapping.map(oldArg, newBody->getArgument(start + 1));
    }
  }

  rewriter.setInsertionPointToEnd(newBody);

  for (Operation &op : *oldBody) {
    if (isa<scf::YieldOp>(&op))
      continue;

    bool consumesPartitionedValue = llvm::any_of(
        op.getOperands(), [&](Value operand) { return pvm.contains(operand); });

    if (auto ext = dyn_cast<tensor::ExtractSliceOp>(&op)) {
      Value src = ext.getSource();
      auto pvIt = pvm.find(src);
      if (pvIt != pvm.end()) {
        auto planIdxIt = pvmPlanIdx.find(src);
        if (planIdxIt == pvmPlanIdx.end()) {
          LDBG(
              "  [ERROR] missing partition owner for extract source: " << *ext);
          return failure();
        }
        const auto &plan = plans[planIdxIt->second];
        auto partIdx =
            findPartitionForSliceOp(plan.partitions, plan.sliceToPartIdx, &op);
        if (!partIdx || *partIdx >= pvIt->second.size()) {
          LDBG(
              "  [ERROR] failed to match extract_slice to partition: " << *ext);
          return failure();
        }
        mapping.map(ext.getResult(), pvIt->second[*partIdx]);
        LDBG("  extract_slice -> P" << *partIdx << " (elided)");
        continue;
      }
    }

    if (auto ins = dyn_cast<tensor::InsertSliceOp>(&op)) {
      Value dest = ins.getDest();
      auto pvIt = pvm.find(dest);
      if (pvIt != pvm.end()) {
        auto planIdxIt = pvmPlanIdx.find(dest);
        if (planIdxIt == pvmPlanIdx.end()) {
          LDBG("  [ERROR] missing partition owner for insert dest: " << *ins);
          return failure();
        }
        const auto &plan = plans[planIdxIt->second];
        auto partIdx =
            findPartitionForSliceOp(plan.partitions, plan.sliceToPartIdx, &op);
        if (!partIdx || *partIdx >= pvIt->second.size()) {
          LDBG("  [ERROR] failed to match insert_slice to partition: " << *ins);
          return failure();
        }

        SmallVector<Value> fwd(pvIt->second);
        fwd[*partIdx] = mapping.lookupOrDefault(ins.getSource());
        pvm[ins.getResult()] = std::move(fwd);
        pvmPlanIdx[ins.getResult()] = planIdxIt->second;
        LDBG("  insert_slice -> P" << *partIdx << " (updated pvm)");
        continue;
      }
    }

    if (consumesPartitionedValue) {
      if (isPartitionPropagatableTensorOp(&op)) {
        std::optional<unsigned> planIdx;
        std::optional<unsigned> numPartitions;
        SmallVector<Value> partResults;
        if (auto dstStyle = dyn_cast<DestinationStyleOpInterface>(&op)) {
          bool hasPartitionedInit =
              llvm::any_of(dstStyle.getDpsInits(),
                           [&](Value init) { return pvm.contains(init); });
          if (op.getNumResults() != 0 && !hasPartitionedInit) {
            LDBG("  [ERROR] destination-style op lacks partition-local init: "
                 << op);
            return failure();
          }
        }

        for (Value operand : op.getOperands()) {
          if (!isa<RankedTensorType>(operand.getType()))
            continue;

          auto pvIt = pvm.find(operand);
          if (pvIt == pvm.end())
            continue;

          auto planIdxIt = pvmPlanIdx.find(operand);
          if (planIdxIt == pvmPlanIdx.end()) {
            LDBG("  [ERROR] missing partition owner for operand of: " << op);
            return failure();
          }

          if (!planIdx) {
            planIdx = planIdxIt->second;
            numPartitions = pvIt->second.size();
            partResults.reserve(pvIt->second.size());
            continue;
          }

          if (*planIdx != planIdxIt->second ||
              *numPartitions != pvIt->second.size()) {
            LDBG("  [ERROR] incompatible partition schemes for: " << op);
            return failure();
          }
        }

        if (!planIdx || !numPartitions) {
          LDBG("  [ERROR] tensor-op propagation found no partition owner: "
               << op);
          return failure();
        }

        if (*numPartitions != plans[*planIdx].partitions.size()) {
          LDBG("  [ERROR] partition count mismatch for: " << op);
          return failure();
        }

        for (unsigned partIdx = 0; partIdx < *numPartitions; ++partIdx) {
          SmallVector<Value> partOperands;
          partOperands.reserve(op.getNumOperands());
          for (Value operand : op.getOperands()) {
            if (isa<RankedTensorType>(operand.getType()) &&
                pvm.contains(operand))
              partOperands.push_back(pvm.lookup(operand)[partIdx]);
            else
              partOperands.push_back(mapping.lookupOrDefault(operand));
          }

          auto cloned = clonePartitionPropagatableOpForPartition(rewriter, &op,
                                                                 partOperands);
          if (failed(cloned)) {
            LDBG("  [ERROR] failed to clone partition-local tensor op: " << op);
            return failure();
          }
          partResults.push_back(*cloned);
        }

        pvm[op.getResult(0)] = std::move(partResults);
        pvmPlanIdx[op.getResult(0)] = *planIdx;
        LDBG("  tensor op -> partition-propagated");
        continue;
      }

      LDBG("  [ERROR] unsupported whole-tensor consumer of partitioned value: "
           << op);
      return failure();
    }

    {
      Operation *cloned = rewriter.clone(op, mapping);
      for (auto [oldR, newR] : llvm::zip(op.getResults(), cloned->getResults()))
        mapping.map(oldR, newR);
    }
  }

  // Build new yield.
  auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
  SmallVector<Value> newYieldVals;
  for (unsigned i = 0; i < numOrig; ++i) {
    auto it = argToPlan.find(i);
    if (it != argToPlan.end()) {
      Value yieldedOld = oldYield.getOperand(i);
      auto pvIt = pvm.find(yieldedOld);
      if (pvIt == pvm.end()) {
        LDBG("  [ERROR] yield value not in pvm for arg #" << i);
        return failure();
      }
      newYieldVals.append(pvIt->second);
    } else {
      newYieldVals.push_back(mapping.lookupOrDefault(oldYield.getOperand(i)));
    }
  }
  rewriter.create<scf::YieldOp>(loc, newYieldVals);

  return newLoop;
}

//===--- Stage 6: Exit Rewrite --------------------------------------------===//

/// Rewrite users of the original loop results after iter_arg decomposition.
static LogicalResult
rewriteExitUsers(IRRewriter &rewriter, scf::ForOp oldLoop, scf::ForOp newLoop,
                 ArrayRef<IterArgDecompPlan> plans, ArrayRef<int> origStart,
                 const DenseMap<unsigned, unsigned> &argToPlan) {
  LDBG("[Decompose] Stage 6: rewriting exit users");
  unsigned numOrig = oldLoop.getNumRegionIterArgs();

  for (unsigned i = 0; i < numOrig; ++i) {
    Value oldRes = oldLoop.getResult(i);
    int start = origStart[i];
    auto planIt = argToPlan.find(i);

    if (planIt == argToPlan.end()) {
      rewriter.replaceAllUsesWith(oldRes, newLoop.getResult(start));
      continue;
    }

    auto &plan = plans[planIt->second];
    unsigned np = plan.partitions.size();
    SmallVector<Value> partRes;
    for (unsigned p = 0; p < np; ++p)
      partRes.push_back(newLoop.getResult(start + p));

    bool needReconstruct = false;
    SmallVector<std::pair<tensor::ExtractSliceOp, unsigned>> elides;

    for (OpOperand &use : llvm::make_early_inc_range(oldRes.getUses())) {
      if (auto ext = dyn_cast<tensor::ExtractSliceOp>(use.getOwner())) {
        int mi =
            findMatchingPartition(plan.partitions, ext.getMixedOffsets(),
                                  ext.getMixedSizes(), ext.getMixedStrides());
        if (mi >= 0) {
          elides.push_back({ext, static_cast<unsigned>(mi)});
          continue;
        }
      }
      needReconstruct = true;
    }

    for (auto [ext, pi] : elides)
      rewriter.replaceOp(ext, partRes[pi]);

    if (needReconstruct) {
      LDBG("  result #" << i << ": reconstructing whole tensor");
      rewriter.setInsertionPointAfter(newLoop);
      // Rebuild the whole tensor when external users still require it. Using
      // the original init as the base preserves the outer fixed-point behavior.
      Value base = oldLoop.getInitArgs()[i];
      for (auto [p, part] : llvm::enumerate(plan.partitions)) {
        auto offsets = getAvailableSliceParams(part.offsets, newLoop,
                                               "[Decompose]", "offset");
        auto sizes =
            getAvailableSliceParams(part.sizes, newLoop, "[Decompose]", "size");
        auto strides = getAvailableSliceParams(part.strides, newLoop,
                                               "[Decompose]", "stride");
        if (failed(offsets) || failed(sizes) || failed(strides)) {
          LDBG("  result #"
               << i << ": reconstruct failed — slice params unavailable");
          return failure();
        }
        base = rewriter.create<tensor::InsertSliceOp>(
            newLoop.getLoc(), partRes[p], base, *offsets, *sizes, *strides);
      }
      rewriter.replaceAllUsesWith(oldRes, base);
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern: DecomposeLoopTensorIterArgsPattern
//===----------------------------------------------------------------------===//

/// Decomposes loop-carried tensor iter_args that are accessed exclusively
/// through disjoint static partitions into multiple partition-scoped iter_args.
///
/// The greedy pattern driver handles fixed-point iteration automatically:
/// decomposing an inner loop introduces extract/insert ops in the outer loop
/// body, making the outer loop matchable in the next worklist round.
struct DecomposeLoopTensorIterArgsPattern
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    Block *body = forOp.getBody();
    unsigned numArgs = forOp.getNumRegionIterArgs();
    SmallVector<IterArgDecompPlan, 4> plans;

    // Stages 1-3: analyze each tensor iter_arg.
    for (unsigned i = 0; i < numArgs; ++i) {
      BlockArgument bArg = body->getArgument(i + 1);
      if (!isa<RankedTensorType>(bArg.getType()))
        continue;

      auto chain = discoverBodyPartitions(bArg);
      if (failed(chain))
        continue;
      if (failed(verifyDecompLegality(*chain, bArg, forOp)))
        continue;
      for (auto [partIdx, part] : llvm::enumerate(chain->partitions))
        logPartitionDetail("[Decompose]", partIdx, part);

      // Stage 3: materialize inits.
      OpBuilder builder(rewriter.getContext());
      builder.setInsertionPoint(forOp);
      SmallVector<Value> inits;
      bool ok = true;
      for (auto [pi, part] : llvm::enumerate(chain->partitions)) {
        auto v = materializePartInit(builder, forOp.getInitArgs()[i], part, pi,
                                     forOp);
        if (failed(v)) {
          ok = false;
          break;
        }
        inits.push_back(*v);
      }
      if (!ok)
        continue;

      IterArgDecompPlan plan;
      plan.origIdx = i;
      plan.partitions = std::move(chain->partitions);
      plan.sliceToPartIdx = std::move(chain->sliceToPartIdx);
      plan.chainValues = std::move(chain->chainValues);
      plan.partInits = std::move(inits);
      plans.push_back(std::move(plan));
      LDBG("[Decompose] arg #" << i << " eligible ("
                               << plans.back().partitions.size()
                               << " partitions)");
    }

    if (plans.empty())
      return failure();

    LDBG("[Decompose] " << plans.size() << " args decomposable in loop at "
                        << forOp.getLoc());

    // Stage 4+5: rewrite loop.
    IRRewriter irRewriter(rewriter);
    auto newLoop = rewriteLoop(irRewriter, forOp, plans);
    if (failed(newLoop))
      return failure();

    // Rebuild origStart and argToPlan for exit rewriting.
    DenseMap<unsigned, unsigned> argToPlan;
    for (auto [i, p] : llvm::enumerate(plans))
      argToPlan[p.origIdx] = i;
    SmallVector<int> origStart(numArgs);
    unsigned idx = 0;
    for (unsigned i = 0; i < numArgs; ++i) {
      origStart[i] = static_cast<int>(idx);
      idx += argToPlan.count(i) ? plans[argToPlan[i]].partitions.size() : 1;
    }

    // Stage 6: exit rewrite.
    if (failed(rewriteExitUsers(irRewriter, forOp, *newLoop, plans, origStart,
                                argToPlan)))
      return failure();

    // Drop internal uses before erasure. The body cloning via IRMapping may
    // leave residual cross-references between old and new bodies when PVM-
    // handled chain values (insert_slice results) were not registered in the
    // IRMapping. Dropping defined-value uses within the old body is safe
    // because all *external* uses of the ForOp results have already been
    // replaced by Stage 6.
    forOp.getBody()->dropAllDefinedValueUses();
    rewriter.eraseOp(forOp);
    LDBG("[Decompose] loop decomposition complete\n");
    return success();
  }
};

struct ShrinkBuffersPass
    : public mlir::dicp::LinalgExt::impl::ShrinkBuffersBase<ShrinkBuffersPass> {
  using ShrinkBuffersBase::ShrinkBuffersBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = &getContext();
    {
      RewritePatternSet patterns(ctx);
      patterns.add<ToTensorSliceToSubviewPattern>(ctx);
      memref::SubViewOp::getCanonicalizationPatterns(patterns, ctx);
      memref::CastOp::getCanonicalizationPatterns(patterns, ctx);
      tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, ctx);
      bufferization::ToTensorOp::getCanonicalizationPatterns(patterns, ctx);
      if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
        LLVM_DEBUG(llvm::dbgs() << "Pattern application failed.\n");
        signalPassFailure();
        return;
      }
    }

    // Run the buffer shrinking patterns greedily to a fixed point. Loop
    // decomposition can expose new non-loop roots, so all patterns stay in the
    // same worklist-driven rewrite round.
    {
      RewritePatternSet splitPatterns(ctx);
      splitPatterns.add<SplitTensorEmptyPattern, SplitAllocPattern>(ctx);
      if (this->tileAllBlocks)
        splitPatterns.add<DecomposeLoopTensorIterArgsPattern>(ctx);

      if (failed(applyPatternsGreedily(moduleOp, std::move(splitPatterns)))) {
        LLVM_DEBUG(llvm::dbgs() << "Buffer splitting patterns failed.\n");
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::dicp::LinalgExt::createShrinkBuffersPass() {
  return std::make_unique<ShrinkBuffersPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::dicp::LinalgExt::createShrinkBuffersPass(
    const ShrinkBuffersOptions &options) {
  return std::make_unique<ShrinkBuffersPass>(options);
}
