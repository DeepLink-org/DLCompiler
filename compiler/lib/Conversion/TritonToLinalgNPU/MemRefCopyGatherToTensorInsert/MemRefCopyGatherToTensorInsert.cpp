#include "dicp/Conversion/TritonToLinalgNPU/MemRefCopyGatherToTensorInsert/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::dicp::linked {
#define GEN_PASS_DEF_MEMREFCOPYGATHERTOTENSORINSERT
#include "dicp/Conversion/TritonToLinalgNPU/MemRefCopyGatherToTensorInsert/Passes.h.inc"
} // namespace mlir::dicp::linked

namespace {

/// Helper to convert OpFoldResult to Value.
/// If it is an attribute, creates an arith.constant.
static Value getValueOrCreateConstantIndexOp(PatternRewriter &rewriter,
                                             Location loc, OpFoldResult ofr) {
  if (auto val = dyn_cast<Value>(ofr))
    return val;
  return rewriter.create<arith::ConstantIndexOp>(
      loc, cast<IntegerAttr>(cast<Attribute>(ofr)).getInt());
}

/// Helper function to check if a Value is derived from a Tensor ExtractOp.
/// It supports two chains:
/// 1. ExtractOp -> Value (Target)
/// 2. ExtractOp -> IndexCastOp -> Value (Target)
/// Returns the defining ExtractOp if a match is found involving the loop IV.
static tensor::ExtractOp findSourceExtractOp(Value val, Value loopIV) {
  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return nullptr;

  // Case 1: Direct ExtractOp
  if (auto extractOp = dyn_cast<tensor::ExtractOp>(defOp)) {
    for (Value idx : extractOp.getIndices()) {
      if (idx == loopIV)
        return extractOp;
    }
    return nullptr;
  }

  // Case 2: ExtractOp -> IndexCastOp
  if (auto indexCastOp = dyn_cast<arith::IndexCastOp>(defOp)) {
    if (auto extractOp =
            indexCastOp.getIn().getDefiningOp<tensor::ExtractOp>()) {
      for (Value idx : extractOp.getIndices()) {
        if (idx == loopIV)
          return extractOp;
      }
    }
  }

  return nullptr;
}

/// Pattern to convert a specific memory gather loop into a tensor insertion
/// loop.
///
/// Source Pattern:
///   scf.for %iv ... {
///     %idx = tensor.extract[%iv]            <-- (Optional IndexCast)
///     %view = memref.reinterpret_cast %src to offset: [%idx] ...
///     %subview = memref.subview %alloc[%iv] ...
///     memref.copy %view, %subview
///   }
///
/// Target Pattern:
///   %res = scf.for %iv ... iter_args(%acc = %empty) {
///     %idx = tensor.extract[%iv]
///     %cast_idx = arith.index_cast %idx     <-- (If needed)
///     %view = memref.reinterpret_cast %src to offset: [%cast_idx] ...
///     %val = memref.load %view[0, 0, ...]   <-- Matches rank
///     %next = tensor.insert %val into %acc[%iv, ...] <-- Matches subview
///     offsets scf.yield %next
///   }
struct MemRefCopyGatherToTensorInsertPattern
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // 1. Analyze Loop Body: Must contain exactly one memref.copy
    memref::CopyOp copyOp;
    int copyCount = 0;
    forOp.getBody()->walk([&](memref::CopyOp op) {
      copyOp = op;
      copyCount++;
    });

    if (copyCount != 1 || !copyOp)
      return failure();

    // 2. Validate Target Semantics (Write Side)
    // Expectation: Copy -> SubView -> Alloc -> ToTensor
    auto subViewOp = copyOp.getTarget().getDefiningOp<memref::SubViewOp>();
    if (!subViewOp)
      return failure();

    auto allocOp = subViewOp.getSource().getDefiningOp<memref::AllocOp>();
    if (!allocOp)
      return failure();

    bufferization::ToTensorOp toTensorOp;
    for (Operation *user : allocOp->getUsers()) {
      if (auto op = dyn_cast<bufferization::ToTensorOp>(user)) {
        toTensorOp = op;
        break;
      }
    }

    if (!toTensorOp || toTensorOp->getBlock() != forOp->getBlock() ||
        !forOp->isBeforeInBlock(toTensorOp)) {
      return failure();
    }

    // 3. Validate Source Semantics (Read Side)
    // Expectation: (Extract -> Optional Cast) -> ReinterpretCast -> Copy Source
    auto reinterpretOp =
        copyOp.getSource().getDefiningOp<memref::ReinterpretCastOp>();
    if (!reinterpretOp)
      return failure();

    tensor::ExtractOp extractOp;
    bool patternFound = false;

    // Check dynamic offsets to find the one driven by the loop induction
    // variable.
    for (OpFoldResult ofr : reinterpretOp.getOffsets()) {
      if (auto val = dyn_cast<Value>(ofr)) {
        extractOp = findSourceExtractOp(val, forOp.getInductionVar());
        if (extractOp) {
          patternFound = true;
          break;
        }
      }
    }

    if (!patternFound)
      return failure();

    // ====================================================
    // Rewrite Phase
    // ====================================================
    Location loc = forOp.getLoc();

    // A. Prepare Accumulator (tensor.empty)
    auto resultType = cast<RankedTensorType>(toTensorOp.getResult().getType());
    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    // B. Create New Loop
    auto newForOp = rewriter.create<scf::ForOp>(
        loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
        ValueRange{initTensor});

    newForOp->setAttr("ExtractedLoadOrStore", rewriter.getUnitAttr());

    // C. Populate New Loop Body
    rewriter.setInsertionPointToStart(newForOp.getBody());

    Value iv = newForOp.getInductionVar();
    Value acc = newForOp.getRegionIterArgs()[0];

    // C.1. Recreate Index Calculation
    // We clone the indices from the original extractOp.
    // If an index was the old loop's IV, replace it with the new loop's IV.
    SmallVector<Value> extractIndices;
    for (Value idx : extractOp.getIndices()) {
      if (idx == forOp.getInductionVar())
        extractIndices.push_back(iv);
      else
        extractIndices.push_back(idx);
    }

    Value newExtract = rewriter.create<tensor::ExtractOp>(
        extractOp.getLoc(), extractOp.getTensor(), extractIndices);

    Value newOffsetIdx = newExtract;
    if (!newOffsetIdx.getType().isIndex()) {
      newOffsetIdx = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), newExtract);
    }

    // C.2. Recreate ReinterpretCast
    // Map the matched dynamic offset to our new calculated index.
    OpFoldResult newOffsetOfr = rewriter.getIndexAttr(0);

    if (!reinterpretOp.getMixedOffsets().empty()) {
      OpFoldResult oldOfr = reinterpretOp.getMixedOffsets()[0];

      // If the old offset matches our pattern, replace it.
      bool isTargetOffset = false;
      if (auto val = dyn_cast<Value>(oldOfr)) {
        if (findSourceExtractOp(val, forOp.getInductionVar())) {
          isTargetOffset = true;
        }
      }
      newOffsetOfr = isTargetOffset ? newOffsetIdx : oldOfr;
    }

    Value newSrcMemref = rewriter.create<memref::ReinterpretCastOp>(
        reinterpretOp.getLoc(), reinterpretOp.getType(),
        reinterpretOp.getSource(), newOffsetOfr, reinterpretOp.getMixedSizes(),
        reinterpretOp.getMixedStrides());

    // C.3. Load from Source
    // FIX: Ensure we provide indices for ALL dimensions of the reinterpreted
    // memref. We assume we are loading from the base (0, 0, ...).
    auto memRefType = cast<MemRefType>(newSrcMemref.getType());
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> loadIndices(memRefType.getRank(), c0);

    Value loadedVal =
        rewriter.create<memref::LoadOp>(loc, newSrcMemref, loadIndices);

    // C.4. Insert into Accumulator
    // FIX: Derive insertion indices from the original subview offsets.
    // This handles multi-dimensional tensors correctly by mapping the subview
    // logic.
    SmallVector<Value> insertIndices;
    for (OpFoldResult ofr : subViewOp.getMixedOffsets()) {
      if (auto val = dyn_cast<Value>(ofr)) {
        // If the offset is the old IV, use the new IV.
        if (val == forOp.getInductionVar())
          insertIndices.push_back(iv);
        else
          insertIndices.push_back(val);
      } else {
        // Materialize static offset as a constant index.
        insertIndices.push_back(
            getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
      }
    }

    Value nextTensor =
        rewriter.create<tensor::InsertOp>(loc, loadedVal, acc, insertIndices);

    // C.5. Yield
    auto yieldOp = rewriter.create<scf::YieldOp>(loc, nextTensor);
    yieldOp->setAttr("DiscreteMemAccess", rewriter.getUnitAttr());

    // D. Finalize
    rewriter.replaceOp(toTensorOp, newForOp.getResult(0));

    // Cleanup
    rewriter.eraseOp(forOp);
    rewriter.eraseOp(allocOp);

    return success();
  }
};

struct MemRefCopyGatherToTensorInsertPass
    : mlir::dicp::linked::impl::MemRefCopyGatherToTensorInsertBase<
          MemRefCopyGatherToTensorInsertPass> {

  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<MemRefCopyGatherToTensorInsertPattern>(&getContext());

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::dicp::linked::createMemRefCopyGatherToTensorInsertPass() {
  return std::make_unique<MemRefCopyGatherToTensorInsertPass>();
}