//===- PrepareVectorTiling.cpp - Early vector-tile preparation ------------===//
//
// Early preparation pass for the vector-tiling pipeline.
//
//===----------------------------------------------------------------------===//

#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "prepare-vector-tiling"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace mlir::dicp::LinalgExt;

namespace mlir::dicp::LinalgExt {
#define GEN_PASS_DEF_PREPAREVECTORTILING
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace mlir::dicp::LinalgExt

namespace {

static bool rejectCopyLayout(memref::CopyOp copyOp, const Twine &reason) {
  LDBG("Rejecting memref.copy: " << reason << ": " << copyOp);
  return false;
}

static bool hasStaticShapeAndUnitInnermostStride(MemRefType type,
                                                 StringRef debugName,
                                                 memref::CopyOp copyOp) {
  if (!type || !type.hasStaticShape())
    return rejectCopyLayout(copyOp, debugName + " memref is not static");

  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return rejectCopyLayout(copyOp, debugName + " memref has no strided layout "
                                                "metadata");

  if (strides.empty())
    return rejectCopyLayout(copyOp, debugName + " memref is rank-0");

  if (strides.back() != 1) {
    return rejectCopyLayout(copyOp, debugName + " memref innermost stride is " +
                                        Twine(strides.back()));
  }

  return true;
}

static bool isRelaxedReinterpretCastCopyLayout(memref::CopyOp copyOp,
                                               MemRefType srcType,
                                               MemRefType dstType) {
  auto reinterpretOp =
      copyOp.getSource().getDefiningOp<memref::ReinterpretCastOp>();
  if (!reinterpretOp)
    return rejectCopyLayout(
        copyOp, "relaxed layout requires a memref.reinterpret_cast source");

  if (srcType.getShape() != dstType.getShape())
    return rejectCopyLayout(copyOp,
                            "relaxed layout requires matching source/target "
                            "shapes");

  if (!hasStaticShapeAndUnitInnermostStride(srcType, "source", copyOp) ||
      !hasStaticShapeAndUnitInnermostStride(dstType, "target", copyOp)) {
    return rejectCopyLayout(
        copyOp, "relaxed layout requires static shapes and unit innermost "
                "strides on both memrefs");
  }

  LDBG("Accepted memref.copy through the relaxed reinterpret-cast layout rule: "
       << copyOp);
  return true;
}

static LogicalResult
validateVectorTileGlobalReshapePrecondition(Operation *op) {
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<tensor::ExpandShapeOp>([](tensor::ExpandShapeOp expandOp) {
        if (expandOp.getResultType().hasStaticShape())
          return success();
        expandOp.emitError("tensor.expand_shape must have a fully static "
                           "result shape for the vector-tile pipeline");
        return failure();
      })
      .Case<tensor::CollapseShapeOp>([](tensor::CollapseShapeOp collapseOp) {
        if (collapseOp.getSrcType().hasStaticShape())
          return success();
        collapseOp.emitError("tensor.collapse_shape must have a fully static "
                             "source shape for the vector-tile pipeline");
        return failure();
      })
      .Default([](Operation *) { return success(); });
}

static bool isSubsetLikeOpForVectorTile(Operation *op) {
  return op && isa<OffsetSizeAndStrideOpInterface>(op) &&
         (isa<SubsetExtractionOpInterface>(op) ||
          isa<SubsetInsertionOpInterface>(op));
}

static bool hasOnlyUnitStridesForVectorTile(Operation *op) {
  auto sliceLikeOp = cast<OffsetSizeAndStrideOpInterface>(op);
  return llvm::all_of(sliceLikeOp.getStaticStrides(),
                      [](int64_t stride) { return stride == 1; });
}

static bool hasCompatibleVectorTileCopyLayout(memref::CopyOp copyOp) {
  auto srcType = dyn_cast<MemRefType>(copyOp.getSource().getType());
  auto dstType = dyn_cast<MemRefType>(copyOp.getTarget().getType());
  if (!srcType || !dstType)
    return rejectCopyLayout(copyOp, "source/target are not memrefs");

  SmallVector<int64_t> srcStrides;
  SmallVector<int64_t> dstStrides;
  int64_t srcOffset = 0;
  int64_t dstOffset = 0;
  if (failed(srcType.getStridesAndOffset(srcStrides, srcOffset)) ||
      failed(dstType.getStridesAndOffset(dstStrides, dstOffset)))
    return rejectCopyLayout(copyOp,
                            "source/target strides cannot be extracted");

  if (srcStrides.size() != dstStrides.size()) {
    return rejectCopyLayout(copyOp, "source/target ranks differ");
  }

  for (auto [dim, srcStride, dstStride] :
       llvm::enumerate(srcStrides, dstStrides)) {
    if (srcStride == dstStride)
      continue;

    LDBG("memref.copy has a stride mismatch at dim "
         << dim << ". src=" << srcStride << ", dst=" << dstStride << ": "
         << copyOp);
    return isRelaxedReinterpretCastCopyLayout(copyOp, srcType, dstType);
  }

  LDBG("Accepted memref.copy with identical source/target strides: " << copyOp);
  return true;
}

static LogicalResult validateVectorTileGlobalPreconditions(Operation *root) {
  bool hasViolation = false;
  LDBG("Running global vector-tile legality precheck on root op: " << *root);

  root->walk([&](Operation *op) {
    if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
      if (hasCompatibleVectorTileCopyLayout(copyOp))
        return;

      hasViolation = true;
      copyOp.emitError("memref.copy must preserve layout for the vector-tile "
                       "pipeline; expected identical source/target strides, "
                       "or a reinterpret-cast source with matching static "
                       "shape and unit innermost stride");
      return;
    }

    if (isSubsetLikeOpForVectorTile(op)) {
      if (hasOnlyUnitStridesForVectorTile(op))
        return;

      hasViolation = true;
      op->emitError("subset operations must have unit strides on every "
                    "dimension for the vector-tile pipeline");
      LDBG("Global vector-tile legality violation: non-unit-stride subset op: "
           << *op);
      return;
    }

    if (failed(validateVectorTileGlobalReshapePrecondition(op))) {
      hasViolation = true;
      LDBG("Global vector-tile legality violation: reshape op with dynamic "
           "shape requirement breach: "
           << *op);
    }
  });

  if (hasViolation) {
    LDBG("Global vector-tile legality precheck failed");
    return failure();
  }

  LDBG("Global vector-tile legality precheck passed");
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern: linalg.fill memref → tensor
//===----------------------------------------------------------------------===//

/// Convert `linalg.fill` with memref output to tensor semantics, then convert
/// the tensor result back to memref via `bufferization.to_buffer` and replace
/// all downstream uses of the original memref.
///
/// Matches:
///   linalg.fill ins(%scalar) outs(%memref : memref<...>)
/// Produces:
///   %t   = bufferization.to_tensor %memref restrict writable
///   %r   = linalg.fill ins(%scalar) outs(%t) -> tensor<...>
///   %buf = bufferization.to_buffer %r : memref<...>
///   (all downstream uses of %memref are replaced with %buf)
struct ConvertFillBufferToTensorPattern
    : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    if (!fillOp.hasPureBufferSemantics())
      return failure();

    Location loc = fillOp.getLoc();
    Value memref = fillOp.getDpsInits()[0];
    auto memrefType = cast<MemRefType>(memref.getType());

    LDBG("Converting linalg.fill at " << loc);
    LDBG("  memref operand: " << memref << " : " << memrefType);

    // Bail out if the fill is nested in a region (e.g. scf.if) deeper than
    // the memref definition.  The to_buffer we create would not dominate
    // uses of the memref outside that nested region.
    if (fillOp->getParentRegion() != memref.getParentRegion()) {
      LDBG("  Skipping: fill is in a nested region relative to memref def");
      return failure();
    }

    // Step 1: Convert memref init to tensor.
    Type tensorType = memref::getTensorTypeFromMemRefType(memrefType);
    auto toTensorOp = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, memref, /*restrict=*/true, /*writable=*/true);
    LDBG("  Created to_tensor: " << *toTensorOp);

    // Step 2: Create tensor-semantics linalg.fill.
    Value scalarInput = fillOp.getDpsInputs()[0];
    auto newFill = rewriter.create<linalg::FillOp>(loc, scalarInput,
                                                   toTensorOp.getResult());
    LDBG("  Created tensor fill: " << *newFill);

    // Step 3: Convert tensor result back to memref via to_buffer.
    auto toBufferOp = rewriter.create<bufferization::ToBufferOp>(
        loc, memrefType, newFill.getResult(0));
    LDBG("  Created to_buffer: " << *toBufferOp);

    // Step 4: Replace uses of the original memref across all blocks.
    //  - If a use is an existing `bufferization.to_tensor`, replace its tensor
    //    *result* directly with the fill's tensor result (short-circuiting the
    //    redundant memref→tensor round-trip).
    //  - Otherwise, replace the memref use with the `to_buffer` result.
    Value fillTensor = newFill.getResult(0);
    Value newMemref = toBufferOp.getResult();

    SmallVector<bufferization::ToTensorOp> toTensorUsers;
    for (OpOperand &use : llvm::make_early_inc_range(memref.getUses())) {
      Operation *user = use.getOwner();
      if (user == toTensorOp.getOperation())
        continue; // Keep our own to_tensor intact.
      if (auto tt = dyn_cast<bufferization::ToTensorOp>(user)) {
        toTensorUsers.push_back(tt);
        continue;
      }
    }

    // Fold existing to_tensor ops: their results become the fill's tensor.
    for (auto tt : toTensorUsers) {
      LDBG("  Folding existing to_tensor: " << *tt << " -> fill tensor result");
      rewriter.replaceOp(tt, fillTensor);
    }

    // Replace remaining memref uses with the new buffer (cross-block safe).
    memref.replaceUsesWithIf(newMemref, [&](OpOperand &use) {
      return use.getOwner() != toTensorOp.getOperation();
    });
    LDBG("  Replaced remaining memref uses with to_buffer result.");

    rewriter.eraseOp(fillOp);
    LDBG("  Erased original buffer-semantics fill.");
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct PrepareVectorTilingPass
    : public mlir::dicp::LinalgExt::impl::PrepareVectorTilingBase<
          PrepareVectorTilingPass> {
  using PrepareVectorTilingBase::PrepareVectorTilingBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    LDBG("=== Running on function: " << funcOp.getName() << " ===");
    if (failed(validateVectorTileGlobalPreconditions(funcOp))) {
      LDBG("=== Global vector-tile legality precheck FAILED ===");
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertFillBufferToTensorPattern>(ctx);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      LDBG("=== Greedy pattern application FAILED ===");
      signalPassFailure();
      return;
    }

    LDBG("=== Completed successfully on: " << funcOp.getName() << " ===");
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::dicp::LinalgExt::createPrepareVectorTilingPass() {
  return std::make_unique<PrepareVectorTilingPass>();
}
