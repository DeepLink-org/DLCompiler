#include "dicp/Dialect/LinalgExt/Analysis/SliceParallelAnalysis.h"
#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "canonicalize-tileable-ops"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace dicp;
using namespace LinalgExt;
using namespace mlir::dicp::stage_attrs;

namespace mlir {
namespace dicp {
namespace LinalgExt {
#define GEN_PASS_DEF_CANONICALIZETILEABLEOPS
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace LinalgExt
} // namespace dicp
} // namespace mlir

namespace {

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

static StringRef findAttrNameWithPrefix(Operation *op, StringRef prefix) {
  if (!op)
    return {};

  for (const NamedAttribute &attr : op->getAttrs()) {
    StringRef attrName = attr.getName().strref();
    if (attrName.starts_with(prefix))
      return attrName;
  }
  return {};
}

static bool hasAttrWithPrefix(Operation *op, StringRef prefix) {
  return !findAttrNameWithPrefix(op, prefix).empty();
}

/// Converts a mixed bound/step operand to an index-typed OpFoldResult.
static FailureOr<OpFoldResult>
materializeIndexOFR(OpFoldResult ofr, PatternRewriter &rewriter, Location loc) {
  if (auto val = dyn_cast<Value>(ofr)) {
    if (val.getType().isIndex())
      return ofr;
    if (!val.getType().isSignlessInteger())
      return failure();
    return OpFoldResult(
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), val)
            .getResult());
  }

  if (auto attr = dyn_cast<Attribute>(ofr)) {
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (!intAttr)
      return failure();
    return intAttr.getType().isIndex()
               ? ofr
               : OpFoldResult(rewriter.getIndexAttr(intAttr.getInt()));
  }
  return failure();
}

/// Create empty tensors for linalg outputs if matching operands aren't found.
static SmallVector<Value>
getOrCreateOperandsMatchingResultTypes(OpBuilder &b, Operation *op) {
  assert(op && isConvertibleElementwiseOp(op) &&
         "expected convertible elementwise op");
  Location loc = op->getLoc();
  ValueRange operands = op->getOperands();
  SmallVector<Value> outputs;
  outputs.reserve(op->getNumResults());
  for (Type t : op->getResultTypes()) {
    auto it =
        llvm::find_if(operands, [&](Value v) { return v.getType() == t; });
    if (it != operands.end()) {
      outputs.push_back(*it);
      continue;
    }

    LDBG("[ElementwiseToGeneric] materialize empty output for type " << t);
    outputs.push_back(b.create<tensor::EmptyOp>(
        loc, tensor::getMixedSizes(b, loc, operands.front()),
        cast<RankedTensorType>(t).getElementType()));
  }
  return outputs;
}

//===----------------------------------------------------------------------===//
// Unroll Jam Pre-Processing
//===----------------------------------------------------------------------===//

/// Performs loop unroll-jamming based on stage tile metadata.
static void applyTileUnrollJam(func::FuncOp funcOp) {
  LDBG("[UnrollJam] start");
  DenseMap<StringRef, Operation *> fuseTagToAnchorOp;

  // 1. Build mapping from `kTileMetaFuseTag` to the anchor op (e.g.,
  // materialize_in_destination).
  funcOp.walk([&](Operation *op) {
    if (auto dict = op->getAttrOfType<DictionaryAttr>(kTileMetaTag)) {
      if (auto fuseTag = dict.getAs<StringAttr>(kTileMetaFuseTag)) {
        fuseTagToAnchorOp[fuseTag.getValue()] = op;
        LDBG("[UnrollJam] map fuse tag '" << fuseTag.getValue() << "' -> "
                                          << op->getName());
      }
    }
  });

  if (fuseTagToAnchorOp.empty()) {
    LDBG("[UnrollJam] no tile metadata, skip");
    return;
  }

  SmallVector<std::pair<scf::ForOp, uint64_t>> unrollCandidates;

  // 2. Discover target scf::ForOp and compute unroll factors.
  funcOp.walk([&](scf::ForOp forOp) {
    if (!forOp->hasAttr(kNPUStageAttrName) || forOp.getNumResults() > 1)
      return WalkResult::advance();

    StringRef foundFuseTag = findAttrNameWithPrefix(forOp, kTileMetaFuseTag);
    if (foundFuseTag.empty())
      return WalkResult::advance();

    auto it = fuseTagToAnchorOp.find(foundFuseTag);
    if (it == fuseTagToAnchorOp.end()) {
      LDBG("[UnrollJam] missing anchor for fuse tag '" << foundFuseTag << "'");
      return WalkResult::advance();
    }

    Operation *anchorOp = it->second;
    auto dict = anchorOp->getAttrOfType<DictionaryAttr>(kTileMetaTag);
    if (!dict)
      return WalkResult::advance();

    // Parse tile_sizes
    SmallVector<int64_t> tileSizes;
    if (auto denseArr = dict.getAs<DenseI64ArrayAttr>(kTileMetaTileSizesTag)) {
      tileSizes = llvm::to_vector(denseArr.asArrayRef());
    } else if (auto arr = dict.getAs<ArrayAttr>(kTileMetaTileSizesTag)) {
      for (auto a : arr) {
        if (auto intAttr = dyn_cast<IntegerAttr>(a))
          tileSizes.push_back(intAttr.getInt());
      }
    }

    if (tileSizes.empty())
      return WalkResult::advance();

    // Parse shaped type from operands or results
    ShapedType shapedType = nullptr;
    if (anchorOp->getNumOperands() > 0)
      shapedType = dyn_cast<ShapedType>(anchorOp->getOperand(0).getType());
    if (!shapedType && anchorOp->getNumResults() > 0)
      shapedType = dyn_cast<ShapedType>(anchorOp->getResult(0).getType());

    if (!shapedType || !shapedType.hasStaticShape()) {
      LDBG("[UnrollJam] skip dynamic/missing shape on " << anchorOp->getName());
      return WalkResult::advance();
    }

    ArrayRef<int64_t> shape = shapedType.getShape();
    if (shape.size() != tileSizes.size()) {
      LDBG("[UnrollJam] skip rank mismatch: shape="
           << shape.size() << ", tile_sizes=" << tileSizes.size());
      return WalkResult::advance();
    }

    // Compute total tile_num
    int64_t tileNum = 1;
    for (auto [dim, tileSize] : llvm::zip(shape, tileSizes)) {
      if (tileSize > 0)
        tileNum *= llvm::divideCeilSigned(dim, tileSize);
    }

    // Extract constant loop bounds
    llvm::APInt lbVal, ubVal, stepVal;
    if (!matchPattern(forOp.getLowerBound(), m_ConstantInt(&lbVal)) ||
        !matchPattern(forOp.getUpperBound(), m_ConstantInt(&ubVal)) ||
        !matchPattern(forOp.getStep(), m_ConstantInt(&stepVal)) ||
        stepVal.isZero()) {
      LDBG("[UnrollJam] skip non-constant loop bounds/step");
      return WalkResult::advance();
    }
    int64_t lbInt = lbVal.getSExtValue();
    int64_t ubInt = ubVal.getSExtValue();
    int64_t stepInt = stepVal.getSExtValue();
    int64_t tripCount = llvm::divideCeilSigned(ubInt - lbInt, stepInt);

    // Check if loop trip count is perfectly divisible by tile_num
    if (tripCount > 0 && tileNum > 0 && tripCount % tileNum == 0) {
      uint64_t unrollFactor = tripCount / tileNum;
      if (unrollFactor > 1) {
        LDBG("[UnrollJam] candidate tag="
             << foundFuseTag << ", trip_count=" << tripCount
             << ", tile_num=" << tileNum << ", factor=" << unrollFactor);
        unrollCandidates.push_back({forOp, unrollFactor});
      }
    }
    return WalkResult::advance();
  });

  // 3. Apply loop unroll jam
  for (auto &[forOp, factor] : unrollCandidates) {
    LDBG("[UnrollJam] apply factor=" << factor << " at " << forOp.getLoc());
    if (failed(mlir::loopUnrollByFactor(forOp, factor))) {
      LDBG("[UnrollJam] failed");
    } else {
      LDBG("[UnrollJam] success");
    }
  }
  LDBG("[UnrollJam] done");
}

//===----------------------------------------------------------------------===//
// Normalization Patterns
//===----------------------------------------------------------------------===//

/// Sink bufferization.to_tensor immediately after its alloc operand.
struct SinkToTensorToAlloc
    : public OpRewritePattern<bufferization::ToTensorOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(bufferization::ToTensorOp op,
                                PatternRewriter &rewriter) const override {
    Operation *allocOp = op.getOperand().getDefiningOp();
    if (!isa_and_nonnull<memref::AllocOp>(allocOp) ||
        op->getPrevNode() == allocOp)
      return failure();
    LDBG("SinkToTensorToAlloc: Sinking to_tensor after alloc " << *allocOp);
    rewriter.modifyOpInPlace(op, [&]() { op->moveAfter(allocOp); });
    return success();
  }
};

/// Convert elementwise operations to linalg.generic to enable tiling.
struct ConvertElementwiseToGenericPattern : public RewritePattern {
  ConvertElementwiseToGenericPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isConvertibleElementwiseOp(op))
      return failure();
    LDBG("ConvertElementwiseToGenericPattern: Converting op " << op->getName());
    auto rank = getRank(op);
    SmallVector<AffineMap> maps(op->getNumResults() + op->getNumOperands(),
                                rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iterTypes(rank,
                                               utils::IteratorType::parallel);
    auto outputs = getOrCreateOperandsMatchingResultTypes(rewriter, op);
    auto genericOp = rewriter.create<linalg::GenericOp>(
        op->getLoc(), op->getResultTypes(), op->getOperands(), outputs, maps,
        iterTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
          auto resTypes = llvm::map_to_vector(op->getResultTypes(), [](Type t) {
            return cast<TensorType>(t).getElementType();
          });
          auto *scalarOp =
              b.create(loc, op->getName().getIdentifier(),
                       args.take_front(op->getNumOperands()), resTypes);
          // Inherit inherent properties from the original tensor-level op
          // (e.g. `predicate` for arith.cmpi, `fastmath` for arith.addf).
          if (auto props = op->getPropertiesAsAttribute())
            (void)scalarOp->setPropertiesFromAttribute(
                props, [&]() { return scalarOp->emitError(); });
          b.create<linalg::YieldOp>(loc, scalarOp->getResults());
        });
    propagateDicpAttributes(op, genericOp);

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

/// Convert stage-tagged tensor.insert_slice into tensor.extract_slice +
/// linalg.copy + tensor.insert_slice.
///
/// This normalization wraps the slice write with a linalg.copy so that
/// subsequent tiling and fusion passes can operate on a tileable linalg op.
/// The entire rewrite stays in tensor semantics — no memref/bufferization
/// ops are introduced.
///
/// Before:
///   %r = tensor.insert_slice %src into %dest[off][sz][str]
/// After:
///   %ext = tensor.extract_slice %dest[off][sz][str]
///   %copied = linalg.copy ins(%src) outs(%ext) -> tensor<...>
///   %r = tensor.insert_slice %copied into %dest[off][sz][str]
struct ConvertInsertSliceToLinalgCopy
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp op,
                                PatternRewriter &rewriter) const override {
    // Only convert stage-tagged insert_slice ops.
    auto stageAttr = op->getAttrOfType<IntegerAttr>(kNPUStageAttrName);
    if (!stageAttr || !op->hasAttr(kTileMetaTag))
      return failure();

    Location loc = op->getLoc();
    Value src = op.getSource();
    Value dest = op.getDest();

    auto srcType = dyn_cast<RankedTensorType>(src.getType());
    auto destType = dyn_cast<RankedTensorType>(dest.getType());
    if (!srcType || !destType) {
      LDBG("ConvertInsertSliceToLinalgCopy: requires ranked tensor types, "
           "skipping "
           << *op);
      return failure();
    }

    LDBG("ConvertInsertSliceToLinalgCopy: converting " << *op);

    // Step 1: Extract the destination slice to serve as the linalg.copy init.
    auto extractSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, srcType, dest, op.getMixedOffsets(), op.getMixedSizes(),
        op.getMixedStrides());
    LDBG("  -> Created extract_slice: " << *extractSlice);

    // Step 2: Create linalg.copy in tensor semantics (src -> extracted slice).
    auto linalgCopy = rewriter.create<linalg::CopyOp>(
        loc, /*inputs=*/src, /*outputs=*/extractSlice.getResult());
    propagateDicpAttributes(op, linalgCopy);
    linalgCopy->setAttr(kOriginalOpNameAttr,
                        rewriter.getStringAttr(op->getName().getStringRef()));
    LDBG("  -> Created linalg.copy: " << *linalgCopy);

    // Step 3: Insert the copied result back into the destination tensor.
    // NOTE: kTileMetaTag is intentionally NOT propagated to the new
    // insert_slice to prevent this pattern from re-matching.
    auto insertSlice = rewriter.create<tensor::InsertSliceOp>(
        loc, linalgCopy.getResult(0), dest, op.getMixedOffsets(),
        op.getMixedSizes(), op.getMixedStrides());
    insertSlice->setAttr(kNPUStageAttrName, stageAttr);
    insertSlice->setAttr(kOriginalOpNameAttr,
                         rewriter.getStringAttr(op->getName().getStringRef()));
    LDBG("  -> Created insert_slice: " << *insertSlice);

    rewriter.replaceOp(op, insertSlice.getResult());
    return success();
  }
};

/// Lower various copy-like operations to linalg.copy to provide a unified
/// representation for subsequent tiling and fusion passes.
template <typename OpType>
struct ConvertCopyLikeOpToLinalgCopy : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    auto stageAttr = op->template getAttrOfType<IntegerAttr>(kNPUStageAttrName);
    if (!stageAttr)
      return failure();

    LDBG("Converting copy-like op to linalg.copy: " << op->getName());

    Value src, dst;
    if constexpr (std::is_same_v<OpType,
                                 bufferization::MaterializeInDestinationOp>) {
      src = op.getSource();
      dst = op.getDest();
    } else {
      src = op.getSource();
      dst = op.getTarget();
    }

    auto toMemref = [&](Value v) -> Value {
      auto tensorType = dyn_cast<RankedTensorType>(v.getType());
      if (!tensorType)
        return v;

      auto memrefType =
          MemRefType::get(tensorType.getShape(), tensorType.getElementType());
      auto res = rewriter.create<bufferization::ToBufferOp>(op->getLoc(),
                                                            memrefType, v);
      res->setAttr(kNPUStageAttrName, stageAttr);
      if (auto fuseTag = getProducerFuseTag(op)) {
        res->setAttr(
            getStageProducerToFuse(fuseTag->stage, fuseTag->sub, fuseTag->unit),
            UnitAttr::get(op.getContext()));
      }
      return res;
    };

    auto linalgCopy = rewriter.create<linalg::CopyOp>(
        op.getLoc(), toMemref(src), toMemref(dst));

    propagateDicpAttributes(op, linalgCopy);
    linalgCopy->setAttr(kOriginalOpNameAttr,
                        rewriter.getStringAttr(op->getName().getStringRef()));

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert stage-tagged fused `scf.for` loops to `scf.forall` when parallel
/// legality can be proven by LoopSliceParallelAnalysis.
struct ConvertTileFuseForToForallPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

private:
  static LogicalResult verifyIterArgFlow(SmallVector<scf::ForOp> nest) {
    if (nest.empty())
      return failure();
    if (!llvm::all_of(nest.front().getInitArgs(),
                      [](Value v) { return isa<TensorType>(v.getType()); })) {
      LDBG("[ForToForall] Reject: non-tensor init args are not supported.");
      return failure();
    }
    for (size_t i = 0; i < nest.size() - 1; ++i) {
      scf::ForOp parent = nest[i], child = nest[i + 1];
      if (!llvm::equal(parent.getRegionIterArgs(), child.getInitArgs())) {
        LDBG("[ForToForall] Reject: init args not forwarded at depth " << i);
        return failure();
      }
      auto yield = cast<scf::YieldOp>(parent.getBody()->getTerminator());
      if (!llvm::equal(yield.getOperands(), child.getResults())) {
        LDBG("[ForToForall] Reject: yields don't match child results at depth "
             << i);
        return failure();
      }
    }
    return success();
  }

  static FailureOr<SmallVector<tensor::InsertSliceOp>>
  analyzeInnermostYield(scf::ForOp innermostFor) {
    auto yieldOp = cast<scf::YieldOp>(innermostFor.getBody()->getTerminator());
    SmallVector<tensor::InsertSliceOp> insertOps;
    insertOps.reserve(yieldOp.getNumOperands());

    for (auto [iterArg, yielded] :
         llvm::zip(innermostFor.getRegionIterArgs(), yieldOp.getOperands())) {
      if (iterArg == yielded) {
        insertOps.push_back(nullptr); // Unmodified
        continue;
      }
      auto insertSlice = yielded.getDefiningOp<tensor::InsertSliceOp>();
      if (!insertSlice || insertSlice.getDest() != iterArg ||
          !insertSlice->hasOneUse()) {
        LDBG("[ForToForall] Reject: invalid innermost yield pattern."
             << innermostFor);
        return failure();
      }
      insertOps.push_back(insertSlice);
    }
    return insertOps;
  }

public:
  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (!forOp->hasAttr(kNPUStageAttrName) ||
        !hasAttrWithPrefix(forOp, kTileMetaFuseTag))
      return failure();

    LDBG("[ForToForall] Try convert staged fused scf.for at "
         << forOp.getLoc());

    if (forOp.getNumResults() > 1) {
      LDBG("[ForToForall] Reject: >1 results at " << forOp.getLoc());
      return failure();
    }

    if (auto parentFor = forOp->getParentOfType<scf::ForOp>()) {
      if (parentFor->hasAttr(kNPUStageAttrName) &&
          hasAttrWithPrefix(parentFor, kTileMetaFuseTag)) {
        SmallVector<scf::ForOp> parentNest;
        mlir::getPerfectlyNestedLoops(parentNest, parentFor);
        if (parentNest.size() > 1 && parentNest[1] == forOp) {
          LDBG("[ForToForall] Skip: deferred to valid parent loop.");
          return failure();
        }
      }
    }

    SmallVector<scf::ForOp> nest;
    mlir::getPerfectlyNestedLoops(nest, forOp);
    if (failed(verifyIterArgFlow(nest)))
      return failure();

    LDBG("[ForToForall] Candidate perfect nest depth: " << nest.size());
    LoopSliceParallelAnalysis loopParallelAnalysis;
    for (auto [depth, loop] : llvm::enumerate(nest)) {
      if (failed(loopParallelAnalysis.analyze(loop))) {
        LDBG("[ForToForall] Reject: ParallelAnalysis failed at depth "
             << depth);
        return failure();
      }
    }

    scf::ForOp innermostFor = nest.back();
    FailureOr<SmallVector<tensor::InsertSliceOp>> innermostInserts =
        analyzeInnermostYield(innermostFor);
    if (failed(innermostInserts))
      return failure();

    SmallVector<OpFoldResult> lowerBounds, upperBounds, steps;
    for (scf::ForOp loop : nest) {
      auto lb =
          materializeIndexOFR(loop.getLowerBound(), rewriter, forOp.getLoc());
      auto ub =
          materializeIndexOFR(loop.getUpperBound(), rewriter, forOp.getLoc());
      auto step = materializeIndexOFR(loop.getStep(), rewriter, forOp.getLoc());
      if (failed(lb) || failed(ub) || failed(step)) {
        LDBG(
            "[ForToForall] Reject: bounds/steps index materialization failed.");
        return failure();
      }
      lowerBounds.push_back(*lb);
      upperBounds.push_back(*ub);
      steps.push_back(*step);
    }

    auto forallOp = rewriter.create<scf::ForallOp>(
        forOp.getLoc(), lowerBounds, upperBounds, steps, forOp.getInitArgs(),
        rewriter.getArrayAttr({}));

    IRMapping mapper;
    Block *forallBody = forallOp.getBody();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forallBody);
    for (auto [idx, loop] : llvm::enumerate(nest)) {
      Value iv = forallBody->getArgument(idx);
      if (iv.getType() != loop.getInductionVar().getType()) {
        iv = rewriter.create<arith::IndexCastOp>(
            forOp.getLoc(), loop.getInductionVar().getType(), iv);
      }
      mapper.map(loop.getInductionVar(), iv);
    }

    ValueRange sharedOuts = forallBody->getArguments().drop_front(nest.size());
    for (scf::ForOp loop : nest) {
      for (auto [iterArg, sharedOut] :
           llvm::zip(loop.getRegionIterArgs(), sharedOuts))
        mapper.map(iterArg, sharedOut);
    }

    DenseSet<Operation *> skipOps;
    for (tensor::InsertSliceOp insert : *innermostInserts) {
      if (insert)
        skipOps.insert(insert);
    }

    auto inParallelOp = cast<scf::InParallelOp>(forallBody->getTerminator());
    rewriter.setInsertionPoint(inParallelOp);

    unsigned clonedOpCount = 0;
    for (Operation &nestedOp : innermostFor.getBody()->without_terminator()) {
      if (!skipOps.contains(&nestedOp)) {
        rewriter.clone(nestedOp, mapper);
        ++clonedOpCount;
      }
    }

    rewriter.setInsertionPointToEnd(inParallelOp.getBody());
    auto mapOFRs = [&](ArrayRef<OpFoldResult> ofrs) {
      return llvm::map_to_vector(ofrs, [&](OpFoldResult ofr) -> OpFoldResult {
        if (auto val = dyn_cast<Value>(ofr))
          return mapper.lookupOrDefault(val);
        return ofr;
      });
    };

    unsigned insertCount = 0;
    for (auto [idx, insert] : llvm::enumerate(*innermostInserts)) {
      if (!insert)
        continue;
      rewriter.create<tensor::ParallelInsertSliceOp>(
          forallOp.getLoc(), mapper.lookupOrDefault(insert.getSource()),
          sharedOuts[idx], mapOFRs(insert.getMixedOffsets()),
          mapOFRs(insert.getMixedSizes()), mapOFRs(insert.getMixedStrides()));
      ++insertCount;
    }

    propagateDicpAttributes(forOp, forallOp);
    rewriter.replaceOp(forOp, forallOp.getResults());

    LDBG("[ForToForall] Success: depth="
         << nest.size() << ", cloned_ops=" << clonedOpCount
         << ", parallel_inserts=" << insertCount);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Main Entry
//===----------------------------------------------------------------------===//

/// Canonicalization pass for tileable IR.
class CanonicalizeTileableOpsPass
    : public mlir::dicp::LinalgExt::impl::CanonicalizeTileableOpsBase<
          CanonicalizeTileableOpsPass> {
public:
  using CanonicalizeTileableOpsBase::CanonicalizeTileableOpsBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    // 1. Perform Loop Unroll Jam based on NPU tile constraints.
    // This is executed *before* `ConvertTileFuseForToForallPattern` transforms
    // perfect `scf.for` nests into `scf.forall`, guaranteeing correct
    // unrolling.
    // applyTileUnrollJam(funcOp);

    // 2. Early phase: convert eligible stage-tagged loops to scf.forall before
    // any other normalization rewrites.
    {
      RewritePatternSet earlyPatterns(ctx);
      earlyPatterns.add<ConvertTileFuseForToForallPattern>(ctx);
      if (failed(applyPatternsGreedily(funcOp, std::move(earlyPatterns)))) {
        signalPassFailure();
        return;
      }
    }

    // Normalize operations to a tileable form using greedy rewrites.
    RewritePatternSet patterns(ctx);
    patterns.add<SinkToTensorToAlloc, ConvertElementwiseToGenericPattern,
                 ConvertInsertSliceToLinalgCopy,
                 ConvertCopyLikeOpToLinalgCopy<
                     bufferization::MaterializeInDestinationOp>,
                 ConvertCopyLikeOpToLinalgCopy<memref::CopyOp>>(ctx);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::func::FuncOp>>
mlir::dicp::LinalgExt::createCanonicalizeTileableOpsPass() {
  return std::make_unique<CanonicalizeTileableOpsPass>();
}
