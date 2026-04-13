
#include "dicp/TransformOps/Transforms.h"

#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/TransformOps/Syntax.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dicp-transform-op-utils"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::transform;
using namespace mlir::dicp;
using namespace mlir::dicp::stage_attrs;

//===----------------------------------------------------------------------===//
// Common Utilities
//===----------------------------------------------------------------------===//

static bool isSubsetOp(Operation *op) {
  return isa<OffsetSizeAndStrideOpInterface>(op) &&
         (isa<SubsetExtractionOpInterface>(op) ||
          isa<SubsetInsertionOpInterface>(op));
}

/// Specific handler for scf.forall reconstruction.
static scf::ForallOp appendToForall(RewriterBase &rewriter,
                                    scf::ForallOp forallOp, Value newOutput,
                                    Value tiledVal,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> sizes) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forallOp);

  // 1. Create new forall op with an additional output operand
  SmallVector<Value> outputs = llvm::to_vector(forallOp.getOutputs());
  outputs.push_back(newOutput);

  auto newForallOp = rewriter.create<scf::ForallOp>(
      forallOp.getLoc(), forallOp.getMixedLowerBound(),
      forallOp.getMixedUpperBound(), forallOp.getMixedStep(), outputs,
      forallOp.getMapping());
  propagateDicpAttributes(forallOp, newForallOp);

  rewriter.eraseBlock(newForallOp.getBody());
  newForallOp.getRegion().takeBody(forallOp.getRegion());

  // Note: ForallOp's bbArgs are [induction_vars..., output_iter_args...]
  BlockArgument newBBArg = newForallOp.getBody()->addArgument(
      newOutput.getType(), newOutput.getLoc());

  // 3. Update the scf.in_parallel terminator
  auto terminator = newForallOp.getTerminator();
  rewriter.setInsertionPointToEnd(terminator.getBody());
  SmallVector<OpFoldResult> strides(offsets.size(), rewriter.getIndexAttr(1));
  rewriter.create<tensor::ParallelInsertSliceOp>(
      newForallOp.getLoc(), tiledVal, newBBArg, offsets, sizes, strides);

  // 4. Update uses of the original loop results (SSA Chain Rewriting).
  // The new loop returns the original results plus the appended one.
  for (auto [oldRes, newRes] :
       llvm::zip(forallOp.getResults(), newForallOp.getResults())) {
    rewriter.replaceAllUsesWith(oldRes, newRes);
  }

  // 5. Restore a valid dummy body to prevent verification failure for the old
  // op. The old op is dead code but must remain valid until cleanup.
  Block *ghostBlock = rewriter.createBlock(&forallOp.getRegion());
  // Add induction variables
  for (int i = 0; i < forallOp.getRank(); ++i)
    ghostBlock->addArgument(rewriter.getIndexType(), forallOp.getLoc());
  // Add output arguments
  for (Value out : forallOp.getOutputs())
    ghostBlock->addArgument(out.getType(), forallOp.getLoc());

  rewriter.setInsertionPointToEnd(ghostBlock);
  rewriter.create<scf::InParallelOp>(forallOp.getLoc());

  return newForallOp;
}

/// Specific handler for scf.for reconstruction.
/// Appends a new output to the scf.for loop, moves the body, updates the yield,
/// and ensures the original loop remains syntactically valid (though dead).
static scf::ForOp appendToFor(RewriterBase &rewriter, scf::ForOp forOp,
                              Value newOutput, Value tiledVal,
                              ArrayRef<OpFoldResult> offsets,
                              ArrayRef<OpFoldResult> sizes) {
  Location loc = forOp.getLoc();
  SmallVector<OpFoldResult> strides(offsets.size(), rewriter.getIndexAttr(1));

  LDBG("Appending an additional iter_arg/result to scf.for via "
       "replaceWithAdditionalYields: "
       << *forOp);
  FailureOr<LoopLikeOpInterface> maybeNewLoop =
      forOp.replaceWithAdditionalYields(
          rewriter, ValueRange{newOutput},
          /*replaceInitOperandUsesInLoop=*/true,
          [&](OpBuilder &b, Location yieldLoc,
              ArrayRef<BlockArgument> newBbArgs) -> SmallVector<Value> {
            assert(newBbArgs.size() == 1 &&
                   "expected exactly one appended iter arg");
            Value updatedTensor = b.create<tensor::InsertSliceOp>(
                yieldLoc, tiledVal, newBbArgs.front(), offsets, sizes, strides);
            LDBG("  Added tensor.insert_slice to feed the appended iter_arg");
            return SmallVector<Value>{updatedTensor};
          });

  if (failed(maybeNewLoop)) {
    LDBG("Failed to append an additional result to scf.for: " << *forOp);
    return nullptr;
  }

  auto newForOp = cast<scf::ForOp>(*maybeNewLoop);
  propagateDicpAttributes(forOp, newForOp);
  LDBG("Created replacement scf.for with " << newForOp.getNumResults()
                                           << " results: " << *newForOp);
  return newForOp;
}

/// Main logic to append output to a loop and update dependencies.
static Operation *
appendLoopResultAndFuse(RewriterBase &rewriter, Diagnostic &diag,
                        Operation *producerOp, Operation *containingOp,
                        TilingResult &tileAndFuseResult, int64_t resultNumber,
                        SmallVector<OpFoldResult> &offsets,
                        SmallVector<OpFoldResult> &sizes) {

  LLVM_DEBUG(llvm::dbgs() << "Checking if output appending is needed for: "
                          << *producerOp << "\n");
  producerOp->setAttr(kHadFusedAttr, UnitAttr::get(rewriter.getContext()));
  // 1. Dominance check for users outside the loop
  SetVector<Operation *> dominatedUsers;
  DominanceInfo domInfo(containingOp);
  Value producerResult = producerOp->getResult(resultNumber);

  for (Operation *user : producerResult.getUsers()) {
    if (!containingOp->isAncestor(user) &&
        domInfo.dominates(containingOp, user)) {
      LLVM_DEBUG(llvm::dbgs() << "[dominatedUsers]: " << *user << "\n");
      dominatedUsers.insert(user);
    }
  }

  bool hasCrossSubStageAttr = producerOp->hasAttr(kCrossTillUnitAttr);
  // If no dominated users and no cross-stage attribute, we don't need to append
  // the result to the loop.
  if (dominatedUsers.empty() && !hasCrossSubStageAttr)
    return nullptr;

  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(producerOp);
  if (!dpsOp)
    return nullptr;

  Value newOutput = dpsOp.getDpsInits()[resultNumber];
  Value tiledVal = tileAndFuseResult.tiledValues[0];
  Operation *newLoop = nullptr;

  // 2. Branch based on loop type
  if (auto forallOp = dyn_cast<scf::ForallOp>(containingOp)) {
    newLoop =
        appendToForall(rewriter, forallOp, newOutput, tiledVal, offsets, sizes);
  } else if (auto forOp = dyn_cast<scf::ForOp>(containingOp)) {
    newLoop = appendToFor(rewriter, forOp, newOutput, tiledVal, offsets, sizes);
  }

  if (!newLoop)
    return nullptr;

  // 3. Update IR usage inside the loop
  BlockArgument newBBArg = newLoop->getRegion(0).getArguments().back();
  rewriter.replaceUsesWithIf(newOutput, newBBArg, [&](OpOperand &use) {
    return newLoop->isProperAncestor(use.getOwner());
  });

  // 4. Connect external dominated users to the new loop result.
  // The new loop has the appended result at the end.
  // IMPORTANT: Only replace uses that the new loop result properly dominates.
  // The new loop is placed just before the old loop, so any use of the producer
  // that appears BEFORE the new loop in program order must NOT be replaced,
  // as that would create a forward-reference dominance violation.
  Value newLoopResult = newLoop->getResults().back();
  if (!dominatedUsers.empty() || hasCrossSubStageAttr) {
    DominanceInfo newDomInfo(newLoop->getParentOp());
    rewriter.replaceUsesWithIf(
        producerResult, newLoopResult, [&](OpOperand &use) {
          Operation *owner = use.getOwner();
          if (newLoop->isAncestor(owner) || owner->hasAttr(kHadFusedAttr))
            return false;
          return newDomInfo.properlyDominates(newLoop, owner);
        });
  }
  return newLoop;
}

static Value tryRankReduce(RewriterBase &rewriter, Location loc, Value value,
                           Type targetType) {
  if (value.getType() == targetType)
    return value;

  auto targetRT = dyn_cast<RankedTensorType>(targetType);
  if (!targetRT)
    return nullptr;

  auto maybeRankReduced = tensor::ExtractSliceOp::rankReduceIfNeeded(
      rewriter, loc, value, targetRT.getShape());

  if (succeeded(maybeRankReduced) &&
      maybeRankReduced->getType() == targetType) {
    return *maybeRankReduced;
  }
  return nullptr;
}

static bool replaceSubsetExtraction(RewriterBase &rewriter, Operation *op,
                                    Value tiledValue) {
  if (op->getNumResults() != 1) {
    return false;
  }

  Value replacement = tryRankReduce(rewriter, op->getLoc(), tiledValue,
                                    op->getResult(0).getType());

  if (!replacement) {
    LDBG("  [SubsetExtraction] Shape mismatch: "
         << tiledValue.getType() << " vs " << op->getResult(0).getType());
    return false;
  }

  rewriter.replaceOp(op, replacement);
  return true;
}

static bool replaceParallelInsertSlice(RewriterBase &rewriter,
                                       tensor::ParallelInsertSliceOp pInsert,
                                       Value tiledValue, Value originalValue) {
  // warning
  pInsert->emitWarning()
      << "Currently, no processing is performed on the ParallelInsertSliceOp.";
  return false;
}

static bool replaceInsertSlice(RewriterBase &rewriter,
                               tensor::InsertSliceOp insertOp, Value tiledValue,
                               Value originalValue) {
  const bool isSource = insertOp.getSource() == originalValue;
  const bool isDest = insertOp.getDest() == originalValue;

  // Nothing to do if the original value is neither source nor destination.
  if (!isSource && !isDest)
    return false;

  // Helper: check whether two values have identical ranked tensor shapes.
  auto hasSameRankedShape = [](Value a, Value b) -> bool {
    auto aType = dyn_cast<RankedTensorType>(a.getType());
    auto bType = dyn_cast<RankedTensorType>(b.getType());
    if (!aType || !bType)
      return true; // Non-ranked types are considered compatible.
    return aType.getShape() == bType.getShape();
  };

  // If we are replacing the destination, require shape compatibility between
  // the insert_slice result and the tiled value.
  if (isDest && !hasSameRankedShape(insertOp.getResult(), tiledValue)) {
    LDBG("replaceInsertSlice: result shape does not match tiledValue shape; "
         "aborting replacement");
    return false;
  }

  Value newSource = insertOp.getSource();
  Value newDest = insertOp.getDest();
  Location loc = insertOp.getLoc();

  // Update source operand if needed.
  if (isSource) {
    if (Value reduced = tryRankReduce(rewriter, loc, tiledValue,
                                      insertOp.getSourceType())) {
      newSource = reduced;
    } else {
      // Source matched but cannot be rank-reduced; keep traversal going.
      return true;
    }
  }

  // Update destination operand if needed.
  if (isDest) {
    if (Value reduced =
            tryRankReduce(rewriter, loc, tiledValue, insertOp.getDestType())) {
      newDest = reduced;
    } else if (Value fallback = tryRankReduce(rewriter, loc, tiledValue,
                                              insertOp.getSourceType())) {
      // Fallback: match the source (slice) shape.
      newDest = fallback;
    } else {
      // Destination matched but cannot be adapted.
      return true;
    }
  }

  rewriter.setInsertionPoint(insertOp);
  rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
      insertOp, newSource, newDest, insertOp.getMixedOffsets(),
      insertOp.getMixedSizes(), insertOp.getMixedStrides());

  return true;
}

static bool replaceSliceWithTiledValue(RewriterBase &rewriter, Operation *op,
                                       Value tiledValue, Value originalValue) {
  OpBuilder::InsertionGuard guard(rewriter);

  if (isa<SubsetExtractionOpInterface>(op)) {
    return replaceSubsetExtraction(rewriter, op, tiledValue);
  }

  if (auto pInsert = dyn_cast<tensor::ParallelInsertSliceOp>(op)) {
    return replaceParallelInsertSlice(rewriter, pInsert, tiledValue,
                                      originalValue);
  }

  if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(op)) {
    return replaceInsertSlice(rewriter, insertOp, tiledValue, originalValue);
  }

  llvm_unreachable("Expected subset extraction or insertion op");
}

/// Return an IRMapping that maps operands of `producerOp` (must be a
/// linalg::LinalgOp) to the corresponding block arguments inside `loopOp`.
///
/// - For scf.for: maps initArgs[i] -> bodyArg(i+1) (bodyArg 0 is induction
/// var).
/// - For scf.forall: maps outputs[i] -> bodyArg(numIVs + i).
/// If no mapping found (producer is not linalg or loop type unsupported)
/// an empty IRMapping is returned.
mlir::IRMapping mapProducerOperandsToLoopArgs(Operation *producerOp,
                                              Operation *loopOp) {
  mlir::IRMapping mapping;

  // require producer to be a LinalgOp
  auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(producerOp);
  if (!linalgOp) {
    LLVM_DEBUG(llvm::dbgs()
               << "[mapProducerOperandsToLoopArgs] "
                  "producer is not a LinalgOp. Returning empty map.\n");
    return mapping;
  }

  // Build a temporary map from loop-defining Value -> corresponding block arg
  llvm::DenseMap<Value, Value> loopValueToArg;

  // scf.for : init args -> body arguments (body arg 0 = iv)
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    Block &body = forOp.getRegion().front();
    auto initArgs = forOp.getInitArgs();
    // body args: [iv, loop-carried-arg0, loop-carried-arg1, ...]
    for (unsigned i = 0, e = initArgs.size(); i != e; ++i) {
      unsigned bodyArgIdx = i + 1;
      if (bodyArgIdx < body.getNumArguments())
        loopValueToArg.try_emplace(initArgs[i], body.getArgument(bodyArgIdx));
      else
        LLVM_DEBUG(llvm::dbgs()
                   << "[mapProducerOperandsToLoopArgs] "
                      "forOp body does not have expected arg index.\n");
    }
  }
  // scf.forall : outputs -> body args after induction vars
  else if (auto forallOp = dyn_cast<scf::ForallOp>(loopOp)) {
    Block &body = forallOp.getRegion().front();
    unsigned numIVs = forallOp.getInductionVars().size();
    auto outputs = forallOp.getOutputs();
    for (unsigned i = 0, e = outputs.size(); i != e; ++i) {
      unsigned bodyArgIdx = numIVs + i;
      if (bodyArgIdx < body.getNumArguments())
        loopValueToArg.try_emplace(outputs[i], body.getArgument(bodyArgIdx));
      else
        LLVM_DEBUG(llvm::dbgs()
                   << "[mapProducerOperandsToLoopArgs] "
                      "forallOp body does not have expected arg index.\n");
    }
  } else {
    // unsupported loop type -> return empty mapping
    LLVM_DEBUG(
        llvm::dbgs()
        << "[mapProducerOperandsToLoopArgs] "
           "loopOp is not scf::ForOp or scf::ForallOp. Returning empty map.\n");
    return mapping;
  }

  // Now, for each operand of the linalg op, map it if it matches a
  // loop-defining value.
  for (Value operand : linalgOp->getOperands()) {
    auto it = loopValueToArg.find(operand);
    if (it != loopValueToArg.end()) {
      mapping.map(operand, it->second);
      LLVM_DEBUG(llvm::dbgs()
                 << "[mapProducerOperandsToLoopArgs] "
                    "mapped operand "
                 << operand << " -> loop block-arg " << it->second << "\n");
    }
  }

  return mapping;
}

static void applyMappingToGeneratedSlices(IRMapping &mapping,
                                          TilingResult &tilingResult,
                                          RewriterBase &rewriter) {

  for (unsigned i = 0, e = tilingResult.generatedSlices.size(); i < e; ++i) {
    auto sliceOp = tilingResult.generatedSlices[i];

    auto extract = dyn_cast<tensor::ExtractSliceOp>(sliceOp);
    if (!extract)
      continue;

    Value mappedBase = mapping.lookupOrNull(extract.getSource());
    if (!mappedBase)
      continue;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(extract);
    auto newSlice = rewriter.create<tensor::ExtractSliceOp>(
        extract.getLoc(), extract.getType(), mappedBase,
        extract.getMixedOffsets(), extract.getMixedSizes(),
        extract.getMixedStrides());

    tilingResult.generatedSlices[i] = newSlice;
    rewriter.replaceOp(extract, newSlice);
  }
}

/// Tile a tensor.expand_shape producer by pushing extract_slice operations
/// through the reshape.  For each reassociation group, the multi-dimensional
/// slice offsets/sizes on the expanded dims are linearized back to a single
/// offset/size on the corresponding source dimension.
///
/// Preconditions (bail on violation):
///   - Result type must be statically shaped.
///   - All extract_slice strides must be 1.
///   - For multi-dim groups, trailing expanded dims must be sliced at offset 0
///     with full extent (only the leading dim may be non-trivially sliced).
///
/// Example:
///   %e = expand_shape %src [[0, 1]] output_shape [64, 1]
///        : tensor<64xf32> into tensor<64x1xf32>
///   %s = extract_slice %e [off, 0] [32, 1] [1, 1]
///   ─────────────────────────────────────────────────
///   %sub = extract_slice %src [off] [32] [1]
///   %s   = expand_shape %sub [[0, 1]] output_shape [32, 1]
static std::tuple<SmallVector<Operation *>, Operation *>
tileExpandShapeIntoContainingOp(RewriterBase &rewriter, Diagnostic &diag,
                                tensor::ExpandShapeOp expandOp,
                                Operation *containingOp) {
  LDBG("Tiling expand_shape producer: " << *expandOp);

  RankedTensorType resultType = expandOp.getResultType();
  if (!resultType.hasStaticShape()) {
    diag.attachNote(expandOp.getLoc())
        << "cannot tile expand_shape with dynamic result shape";
    return {};
  }

  Value source = expandOp.getSrc();
  auto reassocIndices = expandOp.getReassociationIndices();

  // Precompute inner product per group (product of trailing expanded dims):
  //   innerProduct[g] = resultShape[g_1] * resultShape[g_2] * ...
  SmallVector<int64_t> innerProducts;
  for (const auto &group : reassocIndices) {
    int64_t p = 1;
    for (unsigned i = 1; i < group.size(); ++i)
      p *= resultType.getDimSize(group[i]);
    innerProducts.push_back(p);
  }

  // Collect extract_slice users inside the containing op.
  SmallVector<tensor::ExtractSliceOp> innerSlices;
  for (Operation *user : expandOp->getUsers()) {
    if (auto extract = dyn_cast<tensor::ExtractSliceOp>(user);
        extract && containingOp->isProperAncestor(extract))
      innerSlices.push_back(extract);
  }

  if (innerSlices.empty()) {
    diag.attachNote(expandOp.getLoc())
        << "no extract_slice users of expand_shape inside containing op";
    return {};
  }

  SmallVector<Operation *> fusedOps;

  for (tensor::ExtractSliceOp extractOp : innerSlices) {
    // Require unit strides on all dims.
    if (!llvm::all_of(extractOp.getMixedStrides(), [](OpFoldResult s) {
          auto v = getConstantIntValue(s);
          return v && *v == 1;
        })) {
      LDBG("  Skip non-unit-stride extract: " << extractOp);
      continue;
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(extractOp);
    Location loc = extractOp.getLoc();

    auto sliceOffsets = extractOp.getMixedOffsets();
    auto sliceSizes = extractOp.getMixedSizes();

    // Linearize: collapse multi-dim slice coords back to 1D per source dim.
    SmallVector<OpFoldResult> srcOffsets, srcSizes, srcStrides;
    // Collect per-group output_shape for the tiled expand_shape.
    SmallVector<OpFoldResult> tiledOutputShape;
    bool bailout = false;

    for (auto [gIdx, group] : llvm::enumerate(reassocIndices)) {
      // Single-dim group: pass through unchanged.
      if (group.size() == 1) {
        srcOffsets.push_back(sliceOffsets[group[0]]);
        srcSizes.push_back(sliceSizes[group[0]]);
        srcStrides.push_back(rewriter.getIndexAttr(1));
        tiledOutputShape.push_back(sliceSizes[group[0]]);
        continue;
      }

      int64_t inner = innerProducts[gIdx];

      // Trailing expanded dims must be sliced at offset 0 with full extent.
      for (unsigned i = 1; i < group.size(); ++i) {
        auto trailOff = getConstantIntValue(sliceOffsets[group[i]]);
        auto trailSz = getConstantIntValue(sliceSizes[group[i]]);
        int64_t fullExtent = resultType.getDimSize(group[i]);
        if (!trailOff || *trailOff != 0 || !trailSz || *trailSz != fullExtent) {
          LDBG("  Trailing dim " << group[i] << " is not full-extent slice"
                                 << " (off=" << (trailOff ? *trailOff : -1)
                                 << ", sz=" << (trailSz ? *trailSz : -1)
                                 << ", full=" << fullExtent << ")");
          bailout = true;
          break;
        }
      }
      if (bailout)
        break;

      // Linearize offset: leadingOff * inner + 0 (trailing offsets are 0).
      auto mulByInner = [&](OpFoldResult ofr) -> OpFoldResult {
        if (auto c = getConstantIntValue(ofr))
          return rewriter.getIndexAttr(*c * inner);
        Value v = getValueOrCreateConstantIndexOp(rewriter, loc, ofr);
        Value d = rewriter.create<arith::ConstantIndexOp>(loc, inner);
        return rewriter.create<arith::MulIOp>(loc, v, d).getResult();
      };

      srcOffsets.push_back(mulByInner(sliceOffsets[group[0]]));
      srcSizes.push_back(mulByInner(sliceSizes[group[0]]));
      srcStrides.push_back(rewriter.getIndexAttr(1));

      // Build output_shape for tiled expand: leading dim is sliced size,
      // trailing dims keep their original extent.
      tiledOutputShape.push_back(sliceSizes[group[0]]);
      for (unsigned i = 1; i < group.size(); ++i)
        tiledOutputShape.push_back(
            rewriter.getIndexAttr(resultType.getDimSize(group[i])));
    }

    if (bailout)
      continue;

    // Compute the static result type for the tiled expand_shape.
    SmallVector<int64_t> tiledExpandShape;
    for (OpFoldResult ofr : tiledOutputShape) {
      auto c = getConstantIntValue(ofr);
      tiledExpandShape.push_back(c ? *c : ShapedType::kDynamic);
    }
    auto tiledResultType =
        RankedTensorType::get(tiledExpandShape, resultType.getElementType());

    // Build: extract_slice on source → expand_shape on the tile.
    auto tiledExtract = rewriter.create<tensor::ExtractSliceOp>(
        loc, source, srcOffsets, srcSizes, srcStrides);
    auto tiledExpand = rewriter.create<tensor::ExpandShapeOp>(
        loc, tiledResultType, tiledExtract.getResult(), reassocIndices,
        tiledOutputShape);

    LDBG("  Tiled extract: " << *tiledExtract);
    LDBG("  Tiled expand: " << *tiledExpand);

    // Insert tensor.cast if static/dynamic type mismatch.
    Value result = tiledExpand.getResult();
    if (result.getType() != extractOp.getResultType()) {
      LDBG("  Inserting tensor.cast: " << result.getType() << " -> "
                                       << extractOp.getResultType());
      result = rewriter.create<tensor::CastOp>(loc, extractOp.getResultType(),
                                               result);
    }

    rewriter.replaceOp(extractOp, result);
    fusedOps.push_back(tiledExpand);
  }

  if (fusedOps.empty())
    return {};

  return {fusedOps, containingOp};
}

/// Tile a tensor.collapse_shape producer by pushing extract_slice operations
/// through the reshape.  For each reassociation group with N source dims, the
/// 1D slice offset/size is delinearized: the leading source dim receives the
/// divided offset/size, while trailing dims take their full extent.
///
/// Preconditions (bail on violation):
///   - Source type must be statically shaped.
///   - All extract_slice strides must be 1.
///   - For multi-dim groups, offset and size must be aligned to the inner
///     product of the trailing source dimensions.
///
/// Example:
///   %c = collapse_shape %src [[0,1,2]] : tensor<128x4x8xf32> →
///   tensor<4096xf32> %s = extract_slice %c [off] [1024] [1]
///   ──────────────────────────────────────
///   %sub = extract_slice %src [off/32, 0, 0] [32, 4, 8] [1,1,1]
///   %s   = collapse_shape %sub [[0,1,2]]
static std::tuple<SmallVector<Operation *>, Operation *>
tileCollapseShapeIntoContainingOp(RewriterBase &rewriter, Diagnostic &diag,
                                  tensor::CollapseShapeOp collapseOp,
                                  Operation *containingOp) {
  LDBG("Tiling collapse_shape producer: " << *collapseOp);

  RankedTensorType sourceType = collapseOp.getSrcType();
  if (!sourceType.hasStaticShape()) {
    diag.attachNote(collapseOp.getLoc())
        << "cannot tile collapse_shape with dynamic source shape";
    return {};
  }

  Value source = collapseOp.getSrc();
  auto reassocIndices = collapseOp.getReassociationIndices();

  // Precompute inner product per group:
  //   innerProduct[g] = s_{g,1} * s_{g,2} * ... * s_{g,k}
  // i.e. product of all source dims except the leading one in the group.
  SmallVector<int64_t> innerProducts;
  for (const auto &group : reassocIndices) {
    int64_t p = 1;
    for (unsigned i = 1; i < group.size(); ++i)
      p *= sourceType.getDimSize(group[i]);
    innerProducts.push_back(p);
  }

  // Collect extract_slice users inside the containing op.
  SmallVector<tensor::ExtractSliceOp> innerSlices;
  for (Operation *user : collapseOp->getUsers()) {
    if (auto extract = dyn_cast<tensor::ExtractSliceOp>(user);
        extract && containingOp->isProperAncestor(extract))
      innerSlices.push_back(extract);
  }

  if (innerSlices.empty()) {
    diag.attachNote(collapseOp.getLoc())
        << "no extract_slice users of collapse_shape inside containing op";
    return {};
  }

  SmallVector<Operation *> fusedOps;

  for (tensor::ExtractSliceOp extractOp : innerSlices) {
    // Require unit strides on all dims.
    if (!llvm::all_of(extractOp.getMixedStrides(), [](OpFoldResult s) {
          auto v = getConstantIntValue(s);
          return v && *v == 1;
        })) {
      LDBG("  Skip non-unit-stride extract: " << extractOp);
      continue;
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(extractOp);
    Location loc = extractOp.getLoc();

    auto sliceOffsets = extractOp.getMixedOffsets();
    auto sliceSizes = extractOp.getMixedSizes();

    // Delinearize: expand each collapsed dim's 1D slice into multi-dim coords.
    SmallVector<OpFoldResult> srcOffsets, srcSizes, srcStrides;
    bool bailout = false;

    for (auto [gIdx, group] : llvm::enumerate(reassocIndices)) {
      // Single-dim group: pass through unchanged.
      if (group.size() == 1) {
        srcOffsets.push_back(sliceOffsets[gIdx]);
        srcSizes.push_back(sliceSizes[gIdx]);
        srcStrides.push_back(rewriter.getIndexAttr(1));
        continue;
      }

      int64_t inner = innerProducts[gIdx];

      // Static alignment check (when values are known at compile time).
      auto constOff = getConstantIntValue(sliceOffsets[gIdx]);
      auto constSz = getConstantIntValue(sliceSizes[gIdx]);
      if ((constOff && *constOff % inner != 0) ||
          (constSz && *constSz % inner != 0)) {
        LDBG("  Alignment violation in group "
             << gIdx << ": inner=" << inner
             << ", off=" << (constOff ? *constOff : -1)
             << ", sz=" << (constSz ? *constSz : -1));
        bailout = true;
        break;
      }

      // Divide offset and size by inner product for the leading source dim.
      auto divByInner = [&](OpFoldResult ofr) -> OpFoldResult {
        if (auto c = getConstantIntValue(ofr))
          return rewriter.getIndexAttr(*c / inner);
        Value v = getValueOrCreateConstantIndexOp(rewriter, loc, ofr);
        Value d = rewriter.create<arith::ConstantIndexOp>(loc, inner);
        return rewriter.create<arith::DivUIOp>(loc, v, d).getResult();
      };

      srcOffsets.push_back(divByInner(sliceOffsets[gIdx]));
      srcSizes.push_back(divByInner(sliceSizes[gIdx]));
      srcStrides.push_back(rewriter.getIndexAttr(1));

      // Trailing dims in the group: full extent at offset 0.
      for (unsigned i = 1; i < group.size(); ++i) {
        srcOffsets.push_back(rewriter.getIndexAttr(0));
        srcSizes.push_back(
            rewriter.getIndexAttr(sourceType.getDimSize(group[i])));
        srcStrides.push_back(rewriter.getIndexAttr(1));
      }
    }

    if (bailout)
      continue;

    // Build: extract_slice on source → collapse_shape on the tile.
    auto tiledExtract = rewriter.create<tensor::ExtractSliceOp>(
        loc, source, srcOffsets, srcSizes, srcStrides);
    auto tiledCollapse = rewriter.create<tensor::CollapseShapeOp>(
        loc, tiledExtract.getResult(), reassocIndices);

    LDBG("  Tiled extract: " << *tiledExtract);
    LDBG("  Tiled collapse: " << *tiledCollapse);

    // Insert tensor.cast if static/dynamic type mismatch.
    Value result = tiledCollapse.getResult();
    if (result.getType() != extractOp.getResultType()) {
      LDBG("  Inserting tensor.cast: " << result.getType() << " → "
                                       << extractOp.getResultType());
      result = rewriter.create<tensor::CastOp>(loc, extractOp.getResultType(),
                                               result);
    }

    rewriter.replaceOp(extractOp, result);
    fusedOps.push_back(tiledCollapse);
  }

  if (fusedOps.empty())
    return {};

  return {fusedOps, containingOp};
}

std::tuple<SmallVector<Operation *>, Operation *>
mlir::dicp::tileAndFuseAllSubsetOps(RewriterBase &rewriter, Diagnostic &diag,
                                    Operation *producerOp,
                                    Operation *containingOp,
                                    bool duplicateProducer) {
  LLVM_DEBUG(DBGS() << "Try to fuse all extract uses for producer: "
                    << *producerOp << "\n");

  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer) {
    // Tile collapse_shape by delinearizing extract_slice through the reshape.
    if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(producerOp))
      return tileCollapseShapeIntoContainingOp(rewriter, diag, collapseOp,
                                               containingOp);

    // Tile expand_shape by linearizing extract_slice through the reshape.
    if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(producerOp))
      return tileExpandShapeIntoContainingOp(rewriter, diag, expandOp,
                                             containingOp);

    LDBG("Tile-based subset fusion is unavailable for producer: "
         << producerOp->getName());
    return {};
  }

  // Identify valid subset users inside containingOp.
  SmallVector<Operation *> subsetOps;
  for (Operation *user : producerOp->getUsers()) {
    if (!containingOp->isProperAncestor(user) || !isSubsetOp(user))
      continue;
    subsetOps.push_back(user);
  }

  llvm::sort(subsetOps, [](Operation *a, Operation *b) {
    if (a->getBlock() == b->getBlock())
      return a->isBeforeInBlock(b);
    return a->getBlock()->getParentOp()->isAncestor(
        b->getBlock()->getParentOp());
  });

  if (subsetOps.empty()) {
    diag.attachNote(producerOp->getLoc())
        << "could not find fusion opportunity for: " << *producerOp;
    return {};
  }

  // Group by result number.
  std::map<unsigned, SmallVector<Operation *>> resultToSubsetOps;
  for (Operation *op : subsetOps) {
    unsigned resNum = cast<OpResult>(op->getOperand(0)).getResultNumber();
    resultToSubsetOps[resNum].push_back(op);
  }

  SmallVector<Operation *> tiledOps;
  Operation *currentContainingOp = containingOp;

  for (auto &entry : resultToSubsetOps) {
    unsigned resultNumber = entry.first;
    SmallVector<Operation *> &ops = entry.second;
    Operation *firstSliceOp = ops.front();

    auto sliceInterface = cast<OffsetSizeAndStrideOpInterface>(firstSliceOp);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(firstSliceOp);

    SmallVector<OpFoldResult> offsets = sliceInterface.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = sliceInterface.getMixedSizes();

    // For non-tiled dimensions (offset is constant 0 and extract size <
    // producer dim), use the producer's full dimension size for tiling.
    // This ensures the tiled producer covers the full extent of untiled
    // dimensions (e.g., fill should zero the full tile, not just the
    // data subset). A post-extraction slice will trim the result to
    // match the original extract_slice sizes.
    auto resultType =
        cast<RankedTensorType>(producerOp->getResult(resultNumber).getType());
    SmallVector<OpFoldResult> tileSizes(sizes);
    bool needsPostSlice = false;
    for (unsigned i = 0; i < sizes.size(); i++) {
      auto offsetConst = getConstantIntValue(offsets[i]);
      auto sizeConst = getConstantIntValue(sizes[i]);
      int64_t producerDim = resultType.getDimSize(i);
      if (offsetConst && *offsetConst == 0 && sizeConst &&
          producerDim != ShapedType::kDynamic && *sizeConst < producerDim) {
        tileSizes[i] = rewriter.getIndexAttr(producerDim);
        needsPostSlice = true;
        LDBG("  dim " << i << ": adjusting tile size from " << *sizeConst
                      << " to " << producerDim
                      << " (non-tiled dim, use producer's full extent)");
      }
    }

    FailureOr<TilingResult> result = tileableProducer.generateResultTileValue(
        rewriter, resultNumber, offsets, tileSizes);

    if (failed(result)) {
      diag.attachNote(tileableProducer->getLoc())
          << "failed to tile producer op: " << *tileableProducer;
      return {};
    }
    mlir::IRMapping mapping =
        mapProducerOperandsToLoopArgs(producerOp, containingOp);
    if (!mapping.getValueMap().empty()) {
      applyMappingToGeneratedSlices(mapping, *result, rewriter);
    }

    // If we used larger tile sizes for non-tiled dimensions, insert a
    // post-extraction slice to trim the tiled result back to the original
    // extract_slice sizes.
    // e.g., fill(64x16) -> extract_slice(64x16 -> 57x16)
    //
    // Save the full tiled value before post-slicing — appendLoopResultAndFuse
    // needs the full tiled result (64x16) for the parallel_insert_slice, not
    // the trimmed post-slice result (57x16).
    Value fullTiledValue = result->tiledValues[0];
    if (needsPostSlice) {
      SmallVector<OpFoldResult> postOffsets(sizes.size(),
                                            rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> postStrides(sizes.size(),
                                            rewriter.getIndexAttr(1));
      auto postSlice = rewriter.create<tensor::ExtractSliceOp>(
          fullTiledValue.getLoc(), fullTiledValue, postOffsets, sizes,
          postStrides);
      result->tiledValues[0] = postSlice.getResult();
      LDBG("  Inserted post-extraction slice: " << postSlice);
    }

    tiledOps.append(result->tiledOps.begin(), result->tiledOps.end());

    // Replace all subset ops in this group.
    Value tiledValue = result->tiledValues[0];
    Value originalValue = producerOp->getResult(resultNumber);
    for (Operation *opToReplace : ops) {
      replaceSliceWithTiledValue(rewriter, opToReplace, tiledValue,
                                 originalValue);
    }

    if (duplicateProducer) {
      continue;
    }

    // Update containing op signature if needed.
    // Use the full tiled value (before post-slice) and tile sizes for the
    // parallel_insert_slice that writes back to the full tensor.
    result->tiledValues[0] = fullTiledValue;
    Operation *newContainingOp =
        appendLoopResultAndFuse(rewriter, diag, producerOp, currentContainingOp,
                                *result, resultNumber, offsets, tileSizes);

    if (newContainingOp) {
      currentContainingOp = newContainingOp;
    }
  }

  return std::make_tuple(tiledOps, currentContainingOp);
}

static BlockArgument getTiedBlockArgument(LoopLikeOpInterface loop,
                                          OpOperand *opOperand) {
  if (auto forallOp = dyn_cast<scf::ForallOp>(loop.getOperation()))
    return forallOp.getTiedBlockArgument(opOperand);

  if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation())) {
    auto inits = forOp.getInits();
    auto it = llvm::find(inits, opOperand->get());
    if (it != inits.end()) {
      unsigned initIdx = std::distance(inits.begin(), it);
      return forOp.getRegionIterArgs()[initIdx];
    }
  }
  return nullptr;
}

SmallVector<Operation *>
mlir::dicp::tileAndFuseAllSubsetOpsThroughContainingOpBlockArgument(
    RewriterBase &rewriter, Diagnostic &diag, Operation *producerOp,
    LoopLikeOpInterface containingOp) {
  LLVM_DEBUG(DBGS() << "Try to fuse extract uses through block argument for: "
                    << *producerOp << "\n");

  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer) {
    diag.attachNote(producerOp->getLoc())
        << "producer is not a TilingInterface: " << *producerOp;
    return {};
  }

  // Find use by containing op.
  OpOperand *pUse = nullptr;
  for (OpOperand &use : producerOp->getUses()) {
    if (use.getOwner() == containingOp.getOperation()) {
      pUse = &use;
      break;
    }
  }

  if (!pUse) {
    diag.attachNote(producerOp->getLoc())
        << "could not find a use by the containing op: " << *producerOp;
    return {};
  }

  BlockArgument bbArg = getTiedBlockArgument(containingOp, pUse);
  if (!bbArg) {
    diag.attachNote(containingOp.getLoc())
        << "containing op does not have a tied block argument";
    return {};
  }

  // Collect only extraction subset ops (reads). Insertion ops (e.g.,
  // tensor.parallel_insert_slice) are write-back operations and must not
  // be replaced with the tiled producer value.
  SmallVector<Operation *> subsetOps;
  for (Operation *user : bbArg.getUsers()) {
    if (!containingOp->isProperAncestor(user))
      continue;
    if (!isa<SubsetExtractionOpInterface>(user) ||
        !isa<OffsetSizeAndStrideOpInterface>(user))
      continue;
    subsetOps.push_back(user);
  }

  llvm::sort(subsetOps, [](Operation *a, Operation *b) {
    if (a->getBlock() == b->getBlock())
      return a->isBeforeInBlock(b);
    return a->getBlock()->getParentOp()->isAncestor(
        b->getBlock()->getParentOp());
  });

  if (subsetOps.empty()) {
    diag.attachNote(containingOp.getLoc())
        << "could not find extraction-based fusion opportunity for bbArg: "
        << bbArg;
    return {};
  }

  SmallVector<Operation *> tiledOps;
  Operation *firstSliceOp = subsetOps.front();
  auto sliceInterface = cast<OffsetSizeAndStrideOpInterface>(firstSliceOp);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(firstSliceOp);

  int64_t resultNumber = cast<OpResult>(pUse->get()).getResultNumber();

  SmallVector<Value> destinationTensors;
  if (failed(tensor::getOrCreateDestinations(
          rewriter, tileableProducer->getLoc(), tileableProducer,
          destinationTensors))) {
    diag.attachNote(tileableProducer->getLoc())
        << "failed to get destination tensors for: " << *tileableProducer;
    return {};
  }

  // Clone producer to map destination to block arg, then tile the clone.
  IRMapping bvm;
  bvm.map(destinationTensors[resultNumber], bbArg);
  auto tileableProducerClone =
      cast<TilingInterface>(rewriter.clone(*tileableProducer, bvm));

  auto scopeGuard =
      llvm::make_scope_exit([&]() { rewriter.eraseOp(tileableProducerClone); });

  FailureOr<TilingResult> tileAndFuseResult =
      tileableProducerClone.generateResultTileValue(
          rewriter, resultNumber, sliceInterface.getMixedOffsets(),
          sliceInterface.getMixedSizes());

  if (failed(tileAndFuseResult)) {
    diag.attachNote(tileableProducer->getLoc())
        << "failed to tile producer op: " << *tileableProducer;
    return {};
  }

  tiledOps.append(tileAndFuseResult->tiledOps.begin(),
                  tileAndFuseResult->tiledOps.end());
  Value tiledValueToReplace = tileAndFuseResult->tiledValues[0];

  for (Operation *sliceOpToReplace : subsetOps) {
    replaceSliceWithTiledValue(rewriter, sliceOpToReplace, tiledValueToReplace,
                               bbArg);
  }

  // Update containing op operand to point to destination.
  (void)tensor::getOrCreateDestinations(rewriter, tileableProducer->getLoc(),
                                        tileableProducer, destinationTensors);
  rewriter.modifyOpInPlace(containingOp, [&]() {
    containingOp->setOperand(pUse->getOperandNumber(),
                             destinationTensors.front());
  });

  return tiledOps;
}

Operation *mlir::dicp::cloneAndFuseAllSubsetOps(RewriterBase &rewriter,
                                                Diagnostic &diag,
                                                Operation *producerOp,
                                                Operation *containingOp) {
  LDBG("Try to fuse all uses by cloning for: " << *producerOp);

  // Reshape ops must not be cloned — cloning would duplicate the full-size
  // tensor inside the loop body instead of properly tiling through the reshape.
  // These ops should have been handled by tileExpandShapeIntoContainingOp or
  // tileCollapseShapeIntoContainingOp; reaching here means tiling failed.
  if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(producerOp)) {
    diag.attachNote(producerOp->getLoc())
        << "cloning fusion is prohibited for reshape ops "
        << "(collapse_shape/expand_shape); "
        << "tiling-based fusion must be used instead";
    return nullptr;
  }

  // If the producer has cross-substage users, cloning is not allowed because
  // it would break the requirement of maintaining a single consistent tensor
  // state.
  if (producerOp->hasAttr(kCrossTillUnitAttr)) {
    diag.attachNote(producerOp->getLoc())
        << "cloning fusion is prohibited for ops with cross-substage users; "
        << "use tiling-based fusion instead to maintain tensor state.";
    return nullptr;
  }

  if (isa<OffsetSizeAndStrideOpInterface>(producerOp)) {
    diag.attachNote(producerOp->getLoc())
        << "cloning fusion is unsupported for slice-like producers; "
        << "only the dedicated tiling paths may fuse subset operations";
    return nullptr;
  }

  SmallVector<OpOperand *> usesToFuse;
  for (OpResult result : producerOp->getOpResults()) {
    for (OpOperand &use : result.getUses()) {
      Operation *user = use.getOwner();
      if (containingOp->isProperAncestor(user)) {
        usesToFuse.push_back(&use);
      }
    }
  }

  if (usesToFuse.empty()) {
    diag.attachNote(producerOp->getLoc()) << "no fusion opportunity by cloning";
    return nullptr;
  }

  Operation *lastFusedOp = nullptr;
  for (OpOperand *use : usesToFuse) {
    unsigned resultNumber = cast<OpResult>(use->get()).getResultNumber();
    Operation *user = use->getOwner();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(user);
    if (auto toBufferOp = dyn_cast<bufferization::ToBufferOp>(producerOp)) {
      if (auto subViewOp = dyn_cast<memref::SubViewOp>(user)) {
        auto sliceOp = rewriter.create<tensor::ExtractSliceOp>(
            subViewOp.getLoc(), toBufferOp.getTensor(),
            subViewOp.getMixedOffsets(), subViewOp.getMixedSizes(),
            subViewOp.getMixedStrides());

        auto newToBuffer = rewriter.create<bufferization::ToBufferOp>(
            toBufferOp.getLoc(), subViewOp.getType(), sliceOp.getResult());

        rewriter.replaceOp(subViewOp, newToBuffer.getResult());
        lastFusedOp = newToBuffer;
        continue;
      }
    }

    if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(producerOp)) {
      if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user)) {
        auto subViewOp = rewriter.create<memref::SubViewOp>(
            extractSliceOp.getLoc(), toTensorOp.getBuffer(),
            extractSliceOp.getMixedOffsets(), extractSliceOp.getMixedSizes(),
            extractSliceOp.getMixedStrides());

        auto newToTensor = rewriter.create<bufferization::ToTensorOp>(
            toTensorOp.getLoc(), extractSliceOp.getType(),
            subViewOp.getResult());

        rewriter.replaceOp(extractSliceOp, newToTensor.getResult());
        lastFusedOp = newToTensor;
        continue;
      }
    }

    // Default case: clone the producer directly at the use site.
    Operation *fusedOp = rewriter.clone(*producerOp);
    rewriter.modifyOpInPlace(
        user, [&] { use->set(fusedOp->getOpResult(resultNumber)); });
    lastFusedOp = fusedOp;
  }

  return lastFusedOp;
}

//===----------------------------------------------------------------------===//
// TransformApplier
//===----------------------------------------------------------------------===//

void TransformApplier::apply(ModuleOp module,
                             TransformGenerationCallback generator) {
  LLVM_DEBUG(llvm::dbgs()
             << "[TransformApplier] Applying unified transformation...\n");

  // Clone module to isolate transformation attempts
  ModuleOp cloned = module.clone();
  MLIRContext *ctx = module.getContext();

  if (!cloned->hasAttr("transform.with_named_sequence")) {
    cloned->setAttr("transform.with_named_sequence", UnitAttr::get(ctx));
  }

  OpBuilder builder(cloned.getBodyRegion());
  Location loc = cloned.getLoc();
  std::string seqName = "__transform_main";
  Type rootType = builder.getType<transform::AnyOpType>();

  auto seqOp = builder.create<transform::NamedSequenceOp>(
      loc, seqName, rootType, TypeRange{},
      [&](OpBuilder &b, Location bodyLoc, BlockArgument rootHandle) {
        // Invoke the specific generator callback provided by the pass
        generator(b, bodyLoc, rootHandle);
        b.create<transform::YieldOp>(bodyLoc);
      });

  auto entryPoint = cast<transform::TransformOpInterface>(seqOp.getOperation());
  transform::TransformOptions options;
  // Subset-fusion rewrites may erase payload ops while stale transform handles
  // still exist in the cloned interpreter state. The expensive verifier walks
  // every payload attached to every handle and can trip over those dead
  // payloads, even though the generated sequence no longer consumes them.
  options.enableExpensiveChecks(false);

  if (failed(transform::applyTransformNamedSequence(cloned, entryPoint, {},
                                                    options))) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "[TransformApplier] Application of transform sequence failed.\n");
    cloned->emitWarning("Failed to apply unified transformation, reverting.");
    cloned->erase();
    return;
  }

  // Success: Replace original body
  entryPoint.erase();
  module.getBodyRegion().getBlocks().clear();
  IRMapping map;
  cloned.getBodyRegion().cloneInto(&module.getBodyRegion(),
                                   module.getBodyRegion().begin(), map);
  cloned->erase();
  LLVM_DEBUG(llvm::dbgs()
             << "[TransformApplier] Transformation applied successfully.\n");
}
