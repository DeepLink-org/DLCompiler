#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"
#include "dicp/Utils/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "prepare-input-bundle-scheduling"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace mlir::dicp::LinalgExt;
using namespace mlir::dicp;

namespace mlir::dicp::LinalgExt {
#define GEN_PASS_DEF_PREPAREINPUTBUNDLESCHEDULING
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace mlir::dicp::LinalgExt

namespace {

static LogicalResult rejectComposeSubview(memref::SubViewOp subview,
                                          Twine reason) {
  LDBG("[ComposeRCSubview] reject " << reason << ": " << subview);
  return failure();
}

/// Compose a `memref.subview` on top of a `memref.reinterpret_cast` into a
/// single `memref.reinterpret_cast` rooted at the original source. This is
/// intentionally restricted to rank-preserving subviews, which is sufficient
/// for the late cleanup patterns emitted by this pass.
struct ComposeSubviewOfReinterpretCastPattern
    : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp subview,
                                PatternRewriter &rewriter) const override {
    auto reinterpretOp =
        subview.getSource().getDefiningOp<memref::ReinterpretCastOp>();
    if (!reinterpretOp)
      return failure();

    MemRefType subviewType = subview.getType();
    MemRefType reinterpretType = reinterpretOp.getType();
    if (subviewType.getRank() != reinterpretType.getRank())
      return rejectComposeSubview(subview, "rank-reducing subview");

    SmallVector<OpFoldResult> subOffsets = subview.getMixedOffsets();
    SmallVector<OpFoldResult> subSizes = subview.getMixedSizes();
    SmallVector<OpFoldResult> subStrides = subview.getMixedStrides();
    SmallVector<OpFoldResult> reinterpretOffsets =
        reinterpretOp.getMixedOffsets();
    SmallVector<OpFoldResult> reinterpretStrides =
        reinterpretOp.getMixedStrides();
    if (reinterpretOffsets.empty())
      return rejectComposeSubview(subview, "missing reinterpret offset");
    if (subOffsets.size() != reinterpretStrides.size() ||
        subStrides.size() != reinterpretStrides.size()) {
      return rejectComposeSubview(subview, "mismatched offset/stride ranks");
    }

    Location loc = subview.getLoc();
    OpFoldResult newOffset = reinterpretOffsets.front();
    for (auto [subOffset, reinterpretStride] :
         llvm::zip_equal(subOffsets, reinterpretStrides)) {
      OpFoldResult delta = mulOfrs(rewriter, loc, subOffset, reinterpretStride);
      newOffset = addOfrs(rewriter, loc, newOffset, delta);
    }

    SmallVector<OpFoldResult> newStrides;
    newStrides.reserve(subStrides.size());
    for (auto [subStride, reinterpretStride] :
         llvm::zip_equal(subStrides, reinterpretStrides))
      newStrides.push_back(
          mulOfrs(rewriter, loc, subStride, reinterpretStride));

    auto newReinterpret = rewriter.create<memref::ReinterpretCastOp>(
        loc, subviewType, reinterpretOp.getSource(), newOffset, subSizes,
        newStrides);
    LDBG("[ComposeRCSubview] fold " << subview << " into " << newReinterpret);
    rewriter.replaceOp(subview, newReinterpret.getResult());
    return success();
  }
};

struct PrepareInputBundleSchedulingPass
    : public mlir::dicp::LinalgExt::impl::PrepareInputBundleSchedulingBase<
          PrepareInputBundleSchedulingPass> {
  using mlir::dicp::LinalgExt::impl::PrepareInputBundleSchedulingBase<
      PrepareInputBundleSchedulingPass>::PrepareInputBundleSchedulingBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    LDBG("[Pass] run input-bundle scheduling canonicalization preprocess");

    RewritePatternSet patterns(&getContext());
    // Add the custom pattern for composing subview of reinterpret_cast.
    patterns.add<ComposeSubviewOfReinterpretCastPattern>(&getContext());
    // Add standard MLIR memref alias folding patterns.
    memref::populateFoldMemRefAliasOpPatterns(patterns);

    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
mlir::dicp::LinalgExt::createPrepareInputBundleSchedulingPass() {
  return std::make_unique<PrepareInputBundleSchedulingPass>();
}
