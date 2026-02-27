#include "dicp/TransformOps/Transforms.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/TransformOps/Syntax.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dicp-transform-op-utils"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::transform;
using namespace mlir::dicp;

//===----------------------------------------------------------------------===//
// Common Utilities
//===----------------------------------------------------------------------===//

static bool isSubsetOp(Operation *op) {
  return isa<OffsetSizeAndStrideOpInterface>(op) &&
         (isa<SubsetExtractionOpInterface>(op) ||
          isa<SubsetInsertionOpInterface>(op));
}

static SmallVector<Value> recursiveClone(RewriterBase &rewriter,
                                         SmallVector<Value> values,
                                         Operation *clonePoint) {
  LDBG("Start recursiveClone");
  SmallVector<Value> newValues;
  for (auto value : values) {
    if (isa<BlockArgument>(value)) {
      newValues.push_back(value);
      continue;
    }

    auto *defOperation = value.getDefiningOp();
    if (defOperation == nullptr) {
      return newValues;
    }

    // Clone dependency if defined before the clone point in the same block.
    if (clonePoint->getBlock() == defOperation->getBlock() &&
        clonePoint->isBeforeInBlock(defOperation)) {
      LDBG("  Cloning dependency: " << *defOperation);
      auto operands = defOperation->getOperands();
      auto clonedValues = recursiveClone(rewriter, operands, clonePoint);

      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(clonePoint);

      IRMapping mapping;
      mapping.map(operands, clonedValues);
      auto *clonedOp = rewriter.clone(*defOperation, mapping);
      newValues.push_back(
          clonedOp->getResult(cast<OpResult>(value).getResultNumber()));
    } else {
      newValues.push_back(value);
    }
  }
  return newValues;
}

static bool isValidSliceOpInContainingOp(Operation *op,
                                         Operation *containingOp) {
  if (!op || !containingOp->isProperAncestor(op)) {
    return false;
  }

  if (!isSubsetOp(op)) {
    return false;
  }

  // Only support unit strides for union logic.
  auto sliceOp = cast<OffsetSizeAndStrideOpInterface>(op);
  auto staticStrides = sliceOp.getStaticStrides();
  if (llvm::count_if(staticStrides, [](int64_t s) { return s != 1; }) > 0) {
    LDBG("Slice has non-unit stride, invalid for union: " << *op);
    return false;
  }

  return true;
}

static void getFirstSliceUserInContainingOp(
    Operation *producerOp, Operation *containingOp,
    llvm::DenseMap<Value, Operation *> *result2FirstSliceOp,
    llvm::DenseMap<Value, int> *result2ValidNum) {
  LDBG("Scanning for first slice user of producer: " << producerOp->getName());
  for (auto res : producerOp->getResults()) {
    Operation *firstSliceOp = nullptr;
    int validNum = 0;
    for (auto user : res.getUsers()) {
      if (!isValidSliceOpInContainingOp(user, containingOp)) {
        continue;
      }

      if (!firstSliceOp || user->isBeforeInBlock(firstSliceOp)) {
        firstSliceOp = user;
      }
      validNum++;
    }
    result2ValidNum->insert(std::pair(res, validNum));
    if (firstSliceOp) {
      result2FirstSliceOp->insert(std::pair(res, firstSliceOp));
      LDBG("  Found first slice: " << *firstSliceOp
                                   << " (Total valid: " << validNum << ")");
    }
  }
}

enum class MODE {
  UNION_MAX,
  UNION_MIN,
  COMPUTE_SLICE_MAX,
  COMPUTE_SUB,
  COMPUTE_DISTANCE
};

static SmallVector<Value> compute(RewriterBase &rewriter, MODE mode,
                                  const SmallVectorImpl<Value> &lhs,
                                  const SmallVectorImpl<Value> &rhs,
                                  Location loc) {
  auto symA = rewriter.getAffineSymbolExpr(0);
  auto symB = rewriter.getAffineSymbolExpr(1);
  auto one = rewriter.getAffineConstantExpr(1);
  AffineMap map;

  switch (mode) {
  case MODE::UNION_MAX:
  case MODE::UNION_MIN:
    map = AffineMap::get(0, 2, {symA, symB}, rewriter.getContext());
    break;
  case MODE::COMPUTE_SLICE_MAX:
    map = AffineMap::get(0, 2, {symA + symB - one}, rewriter.getContext());
    break;
  case MODE::COMPUTE_SUB:
    map = AffineMap::get(0, 2, {symA - symB}, rewriter.getContext());
    break;
  case MODE::COMPUTE_DISTANCE:
    map = AffineMap::get(0, 2, {symA - symB + one}, rewriter.getContext());
    break;
  }

  SmallVector<Value> results;
  for (auto it : llvm::zip(lhs, rhs)) {
    auto l = std::get<0>(it);
    auto r = std::get<1>(it);
    Value result;
    switch (mode) {
    case MODE::UNION_MAX:
      result = rewriter.create<affine::AffineMaxOp>(loc, map, ValueRange{l, r});
      break;
    case MODE::UNION_MIN:
      result = rewriter.create<affine::AffineMinOp>(loc, map, ValueRange{l, r});
      break;
    case MODE::COMPUTE_SLICE_MAX:
    case MODE::COMPUTE_SUB:
    case MODE::COMPUTE_DISTANCE:
      result =
          rewriter.create<affine::AffineApplyOp>(loc, map, ValueRange{l, r});
      break;
    }
    results.push_back(result);
  }
  return results;
}

SmallVector<OpFoldResult> convert(SmallVectorImpl<Value> &values) {
  SmallVector<OpFoldResult> results;
  for (auto it : values) {
    results.push_back(OpFoldResult(it));
  }
  return results;
}

static SmallVector<Value> createEqualZeroOp(const SmallVector<Value> &targets,
                                            RewriterBase &rewriter,
                                            Location loc) {
  SmallVector<Value> results;
  for (Value target : targets) {
    // Cast to i64 for arithmetic comparison.
    Value castResult =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), target);
    Value zero =
        rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI64Type(), 0);
    Value cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                castResult, zero);
    results.push_back(cond);
  }
  return results;
}

static SmallVector<Value> createSelectOp(const SmallVector<Value> &conds,
                                         const SmallVector<Value> &trues,
                                         const SmallVector<Value> &falses,
                                         RewriterBase &rewriter, Location loc) {
  SmallVector<Value> results;
  for (size_t i = 0; i < conds.size(); ++i) {
    Value result =
        rewriter.create<arith::SelectOp>(loc, conds[i], trues[i], falses[i]);
    results.push_back(result);
  }
  return results;
}

static void unionFirstProducerUser(RewriterBase &rewriter,
                                   Operation *firstSliceOp,
                                   SmallVector<Value> &unionOffsets,
                                   SmallVector<Value> &unionMaxes) {
  LDBG("first SliceOp \n" << *firstSliceOp);
  rewriter.setInsertionPoint(firstSliceOp);

  auto sliceInterface = cast<OffsetSizeAndStrideOpInterface>(firstSliceOp);
  auto sliceOffsets = getValueOrCreateConstantIndexOp(
      rewriter, firstSliceOp->getLoc(), sliceInterface.getMixedOffsets());
  auto sliceSizes = getValueOrCreateConstantIndexOp(
      rewriter, firstSliceOp->getLoc(), sliceInterface.getMixedSizes());

  Value source;
  if (auto viewLike = dyn_cast<ViewLikeOpInterface>(firstSliceOp)) {
    source = viewLike.getViewSource();
  } else {
    source = firstSliceOp->getOperand(0);
  }

  auto srcMixedSizes =
      tensor::getMixedSizes(rewriter, firstSliceOp->getLoc(), source);
  auto srcSizes = getValueOrCreateConstantIndexOp(
      rewriter, firstSliceOp->getLoc(), srcMixedSizes);

  auto isSizesZero =
      createEqualZeroOp(sliceSizes, rewriter, firstSliceOp->getLoc());

  // If slice size is 0, use source size as offset (MAX_VALUE) to avoid
  // affecting min.
  unionOffsets = createSelectOp(isSizesZero, srcSizes, sliceOffsets, rewriter,
                                firstSliceOp->getLoc());

  // If slice size is 0, use slice size (0/MIN_VALUE) to avoid affecting max.
  auto initMaxes = compute(rewriter, MODE::COMPUTE_SLICE_MAX, unionOffsets,
                           sliceSizes, firstSliceOp->getLoc());
  unionMaxes = createSelectOp(isSizesZero, sliceSizes, initMaxes, rewriter,
                              firstSliceOp->getLoc());
}

static void unionNextProducerUser(RewriterBase &rewriter, Location loc,
                                  const SmallVector<Value> &offsets,
                                  const SmallVector<Value> &sizes,
                                  SmallVector<Value> &unionOffsets,
                                  SmallVector<Value> &unionMaxes) {
  LDBG("Unioning next user...");
  auto isSizesZero = createEqualZeroOp(sizes, rewriter, loc);

  // Update union offsets: min(current, new).
  auto newOffsets =
      createSelectOp(isSizesZero, unionOffsets, offsets, rewriter, loc);
  unionOffsets =
      compute(rewriter, MODE::UNION_MIN, unionOffsets, newOffsets, loc);

  // Update union maxes: max(current, new_end).
  auto computeMaxes =
      compute(rewriter, MODE::COMPUTE_SLICE_MAX, newOffsets, sizes, loc);
  auto clonedMaxes =
      createSelectOp(isSizesZero, unionMaxes, computeMaxes, rewriter, loc);
  unionMaxes = compute(rewriter, MODE::UNION_MAX, unionMaxes, clonedMaxes, loc);
}

static tensor::ExtractSliceOp
sliceFromUnion(RewriterBase &rewriter, tensor::ExtractSliceOp unionSlice,
               const SmallVector<Value> &unionOffsets, Operation *sliceOp) {
  LDBG("Creating sliceFromUnion for: " << *sliceOp);
  rewriter.setInsertionPoint(sliceOp);

  auto sliceInterface = cast<OffsetSizeAndStrideOpInterface>(sliceOp);

  auto offsets = getValueOrCreateConstantIndexOp(
      rewriter, sliceOp->getLoc(), sliceInterface.getMixedOffsets());
  auto sizes = getValueOrCreateConstantIndexOp(rewriter, sliceOp->getLoc(),
                                               sliceInterface.getMixedSizes());
  auto isSizesZero = createEqualZeroOp(sizes, rewriter, sliceOp->getLoc());

  // If zero-sized, reset offset to union offset to avoid out-of-bounds
  // calculation.
  offsets = createSelectOp(isSizesZero, unionOffsets, offsets, rewriter,
                           sliceOp->getLoc());
  auto newOffsets = compute(rewriter, MODE::COMPUTE_SUB, offsets, unionOffsets,
                            sliceOp->getLoc());

  auto newSlice = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp->getLoc(), unionSlice.getResult(), convert(newOffsets),
      sliceInterface.getMixedSizes(), unionSlice.getMixedStrides());
  return newSlice;
}

void mlir::dicp::unionProducerUsers(RewriterBase &rewriter, Diagnostic &diag,
                                    Operation *producerOp,
                                    Operation *containingOp) {
  LDBG("unionProducerUsers entry for producer: " << *producerOp);
  llvm::DenseMap<Value, Operation *> result2FirstSliceOp;
  llvm::DenseMap<Value, int> result2ValidNum;
  getFirstSliceUserInContainingOp(producerOp, containingOp,
                                  &result2FirstSliceOp, &result2ValidNum);

  for (auto produceResult : producerOp->getResults()) {
    int validSliceOpNum = result2ValidNum[produceResult];

    // Optimization primarily for > 1 user.
    if (validSliceOpNum < 2) {
      continue;
    }

    auto firstSliceOp = result2FirstSliceOp[produceResult];
    SmallVector<Value> unionOffsets;
    SmallVector<Value> unionMaxes;

    LDBG("begin to union \n" << *containingOp);
    unionFirstProducerUser(rewriter, firstSliceOp, unionOffsets, unionMaxes);

    for (auto *user : produceResult.getUsers()) {
      if (!isValidSliceOpInContainingOp(user, containingOp) ||
          user == firstSliceOp) {
        continue;
      }

      LDBG("union slice \n" << *user);
      auto sliceInterface = cast<OffsetSizeAndStrideOpInterface>(user);

      // Clone values to ensure availability at the union point.
      auto curOffsets = getValueOrCreateConstantIndexOp(
          rewriter, user->getLoc(), sliceInterface.getMixedOffsets());
      auto clonedOffsets = recursiveClone(rewriter, curOffsets, firstSliceOp);

      auto curSizes = getValueOrCreateConstantIndexOp(
          rewriter, user->getLoc(), sliceInterface.getMixedSizes());
      auto clonedSizes = recursiveClone(rewriter, curSizes, firstSliceOp);

      unionNextProducerUser(rewriter, user->getLoc(), clonedOffsets,
                            clonedSizes, unionOffsets, unionMaxes);
    }

    auto unionSizes = compute(rewriter, MODE::COMPUTE_DISTANCE, unionMaxes,
                              unionOffsets, firstSliceOp->getLoc());

    auto firstSliceInterface =
        cast<OffsetSizeAndStrideOpInterface>(firstSliceOp);
    Value source;
    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(firstSliceOp)) {
      source = viewLike.getViewSource();
    } else {
      source = firstSliceOp->getOperand(0);
    }

    auto unionSlice = rewriter.create<tensor::ExtractSliceOp>(
        firstSliceOp->getLoc(), source, convert(unionOffsets),
        convert(unionSizes), firstSliceInterface.getMixedStrides());

    LDBG("insert union slice \n" << unionSlice);

    // Update users to extract from the new union slice.
    for (auto *user : llvm::make_early_inc_range(produceResult.getUsers())) {
      if (!isValidSliceOpInContainingOp(user, containingOp) ||
          user == unionSlice) {
        continue;
      }
      auto newSliceOp =
          sliceFromUnion(rewriter, unionSlice, unionOffsets, user);
      rewriter.replaceOp(user, newSliceOp.getResult());
    }
  }
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
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forOp);
  Location loc = forOp.getLoc();

  // 1. Prepare new iter_args (original inits + new output)
  SmallVector<Value> newIterArgs = llvm::to_vector(forOp.getInits());
  newIterArgs.push_back(newOutput);

  // 2. Create new scf.for op with the expanded signature
  auto newForOp = rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                              forOp.getUpperBound(),
                                              forOp.getStep(), newIterArgs);

  // 3. Transfer the body from the old loop to the new loop
  // We erase the default empty block created by the builder for newForOp first.
  rewriter.eraseBlock(newForOp.getBody());
  newForOp.getRegion().takeBody(forOp.getRegion());

  // 4. Fix the new loop body
  // Add the new block argument corresponding to the new iter_arg
  BlockArgument newBlockArg =
      newForOp.getBody()->addArgument(newOutput.getType(), newOutput.getLoc());

  // Replace uses of the new output *inside* the loop with the new block
  // argument. This enables fusion into the new loop.
  rewriter.replaceUsesWithIf(newOutput, newBlockArg, [&](OpOperand &use) {
    Operation *op = use.getOwner();
    return newForOp->isProperAncestor(op);
  });

  // 5. Update scf.yield terminator in the new loop
  auto yieldOp = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
  rewriter.setInsertionPoint(yieldOp);

  // Create the InsertSliceOp to update the new iter_arg
  // (tiledVal -> slice of newBlockArg)
  SmallVector<OpFoldResult> strides(offsets.size(), rewriter.getIndexAttr(1));
  Value updatedTensor = rewriter.create<tensor::InsertSliceOp>(
      tiledVal.getLoc(), tiledVal, newBlockArg, offsets, sizes, strides);

  // Update yield operands: originals + updated new tensor
  SmallVector<Value> newYieldOperands = llvm::to_vector(yieldOp.getOperands());
  newYieldOperands.push_back(updatedTensor);

  rewriter.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
  rewriter.eraseOp(yieldOp);

  // 6. Update uses of the original loop results (SSA Chain Rewriting).
  // The new loop returns the original results plus the appended one.
  for (auto [oldRes, newRes] :
       llvm::zip(forOp.getResults(), newForOp.getResults())) {
    rewriter.replaceAllUsesWith(oldRes, newRes);
  }

  // 7. Restore a valid dummy body to prevent verification failure.
  Block *ghostBlock = rewriter.createBlock(&forOp.getRegion());

  // Add IV and dummy iter_args matching the original loop signature.
  ghostBlock->addArgument(rewriter.getIndexType(), loc);
  for (Value init : forOp.getInits())
    ghostBlock->addArgument(init.getType(), loc);
  // Yield the dummy iter_args (all arguments except the IV at index 0).
  rewriter.setInsertionPointToEnd(ghostBlock);
  rewriter.create<scf::YieldOp>(loc, ghostBlock->getArguments().drop_front());
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

  auto genericOp = dyn_cast<linalg::GenericOp>(producerOp);
  if (!genericOp)
    return nullptr;

  Value newOutput = genericOp.getOutputs()[resultNumber];
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
  Value newLoopResult = newLoop->getResults().back();
  if (!dominatedUsers.empty() || hasCrossSubStageAttr) {
    rewriter.replaceUsesWithIf(
        producerResult, newLoopResult, [&](OpOperand &use) {
          Operation *owner = use.getOwner();
          return !newLoop->isAncestor(owner) && !owner->hasAttr(kHadFusedAttr);
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
  LDBG("Replacing subset op: " << *op << " ; replace by: " << tiledValue);

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
    // 如果你希望完全替换原来的 extract（把所有 uses 都指向
    // newSlice），取消注释下一行：
    rewriter.replaceOp(extract, newSlice);
  }
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
    diag.attachNote(producerOp->getLoc())
        << "producer is not a TilingInterface: " << *producerOp;
    return {};
  }

  // Identify valid subset users inside containingOp.
  SmallVector<Operation *> subsetOps;
  for (Operation *user : producerOp->getUsers()) {
    if (!containingOp->isProperAncestor(user) || !isSubsetOp(user))
      continue;
    LDBG("  Found candidate slice user: " << *user);
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

    FailureOr<TilingResult> result = tileableProducer.generateResultTileValue(
        rewriter, resultNumber, offsets, sizes);

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
    Operation *newContainingOp =
        appendLoopResultAndFuse(rewriter, diag, producerOp, currentContainingOp,
                                *result, resultNumber, offsets, sizes);

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

  SmallVector<Operation *> subsetOps;
  for (Operation *user : bbArg.getUsers()) {
    if (!containingOp->isProperAncestor(user) || !isSubsetOp(user))
      continue;
    LDBG("  Found candidate slice user: " << *user);
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
        << "could not find fusion opportunity for bbArg: " << bbArg;
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

  // If the producer has cross-substage users, cloning is not allowed because
  // it would break the requirement of maintaining a single consistent tensor
  // state.
  if (producerOp->hasAttr(kCrossTillUnitAttr)) {
    diag.attachNote(producerOp->getLoc())
        << "cloning fusion is prohibited for ops with cross-substage users; "
        << "use tiling-based fusion instead to maintain tensor state.";
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
    // Case 1: bufferization.to_buffer -> memref.subview
    if (auto toBufferOp = dyn_cast<bufferization::ToBufferOp>(producerOp)) {
      if (auto subViewOp = dyn_cast<memref::SubViewOp>(user)) {
        LDBG("Special Case: Pushing memref.subview across "
             "bufferization.to_buffer");

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
    // Case 2: bufferization.to_tensor -> tensor.extract_slice
    if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(producerOp)) {
      if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user)) {
        LDBG("Special Case: Pushing tensor.extract_slice across "
             "bufferization.to_tensor");

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
    // Default Case: Standard cloning for other ops
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