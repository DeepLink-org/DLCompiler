#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"
#include "dicp/TransformOps/Transforms.h"
#include "dicp/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "de-linalgize"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace mlir::dicp;
using namespace mlir::dicp::stage_attrs;

namespace mlir::dicp::LinalgExt {
#define GEN_PASS_DEF_DELINALGIZE
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace mlir::dicp::LinalgExt

namespace {

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

/**
 * @brief Restores linalg.generic ops back to arith/math elementwise ops.
 *
 * This pattern targets ops previously converted to generics for fusion or
 * tiling purposes, restoring them to their high-level functional form for
 * backend-specific code generation.
 */
struct GenericToElementwisePattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp->hasAttr(kNPUStageAttrName))
      return failure();

    LDBG("[GenericRestore] analyze " << genericOp);

    if (!mlir::linalg::isElementwise(genericOp)) {
      LDBG("[GenericRestore] reject non-elementwise generic");
      return failure();
    }

    Block &body = genericOp.getRegion().front();
    if (body.getOperations().size() != 2) { // [ScalarOp, YieldOp]
      LDBG("[GenericRestore] reject body size " << body.getOperations().size()
                                                << ", expected 2");
      return failure();
    }

    Operation *scalarOp = &body.front();
    StringRef opName = scalarOp->getName().getStringRef();

    // Only restore arith and math dialects
    if (!opName.starts_with("arith.") && !opName.starts_with("math.")) {
      LDBG("[GenericRestore] reject inner op " << opName
                                               << " outside arith/math");
      return failure();
    }

    // Map block arguments back to generic operands
    SmallVector<Value> newOperands;
    for (Value operand : scalarOp->getOperands()) {
      auto arg = dyn_cast<BlockArgument>(operand);
      if (!arg || arg.getOwner() != &body) {
        LDBG("[GenericRestore] reject non-block-argument operand");
        return failure();
      }

      unsigned argIdx = arg.getArgNumber();
      if (argIdx >= genericOp.getNumDpsInputs()) {
        LDBG("[GenericRestore] reject output/accumulator operand use");
        return failure();
      }
      newOperands.push_back(genericOp.getDpsInputOperand(argIdx)->get());
    }

    LDBG("[GenericRestore] restore to " << opName);

    Operation *newOp = rewriter.create(
        genericOp.getLoc(), rewriter.getStringAttr(opName), newOperands,
        genericOp.getResultTypes(), scalarOp->getAttrs());

    rewriter.replaceOp(genericOp, newOp->getResults());
    return success();
  }
};

/// Composes two offset arrays element-wise: result[i] = outer[i] + inner[i].
/// Folds statically when both operands are constants; short-circuits when
/// either operand is zero.
static SmallVector<OpFoldResult> composeOffsets(ArrayRef<OpFoldResult> outer,
                                                ArrayRef<OpFoldResult> inner,
                                                OpBuilder &b, Location loc) {
  assert(outer.size() == inner.size() && "offset rank mismatch");
  SmallVector<OpFoldResult> result;
  result.reserve(outer.size());
  for (auto [o, i] : llvm::zip(outer, inner))
    result.push_back(addOfrs(b, loc, o, i));
  return result;
}

/**
 * @brief Restores linalg.copy to its original operation form.
 *
 * Handles three restoration targets based on the `kOriginalOpNameAttr` marker:
 *   1. bufferization.materialize_in_destination
 *   2. memref.copy
 *   3. tensor.insert_slice — with special forall-aware handling that folds
 *      the copy into the parallel_insert_slice and absorbs the outer
 *      extract_slice/insert_slice pair into the forall's shared_outs.
 */
struct LinalgCopyToOriginalPattern : public OpRewritePattern<linalg::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    auto originalNameAttr =
        copyOp->getAttrOfType<StringAttr>(kOriginalOpNameAttr);
    if (!originalNameAttr)
      return failure();

    StringRef originalName = originalNameAttr.getValue();
    Location loc = copyOp.getLoc();
    Value source = copyOp.getInputs().front();
    Value dest = copyOp.getOutputs().front();

    LDBG("[CopyRestore] restore linalg.copy to " << originalName);

    // Propagate attributes excluding our internal marker
    SmallVector<NamedAttribute> filteredAttrs;
    for (auto attr : copyOp->getAttrs()) {
      if (attr.getName() != kOriginalOpNameAttr)
        filteredAttrs.push_back(attr);
    }

    if (originalName ==
        bufferization::MaterializeInDestinationOp::getOperationName()) {
      Value tensorSource = recoverTensorSource(source, rewriter, loc);
      if (!tensorSource) {
        LDBG("[CopyRestore] reject missing tensor source for materialization");
        return failure();
      }

      auto matOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
          loc, TypeRange{}, tensorSource, dest, false, true);
      matOp->setAttrs(filteredAttrs);

      rewriter.eraseOp(copyOp);
      return success();
    }

    if (originalName == memref::CopyOp::getOperationName()) {
      auto memrefCopy = rewriter.create<memref::CopyOp>(loc, source, dest);
      memrefCopy->setAttrs(filteredAttrs);
      rewriter.eraseOp(copyOp);
      return success();
    }

    // Restore linalg.copy (from insert_slice) back to tensor.insert_slice.
    //
    // The normalize pass converted:
    //   %r = tensor.insert_slice %src into %dest[off][sz][str]
    // Into (tensor semantics):
    //   %ext = tensor.extract_slice %dest[off][sz][str]
    //   %copied = linalg.copy ins(%src) outs(%ext) -> tensor<...>
    //   %r = tensor.insert_slice %copied into %dest[off][sz][str]
    if (originalName == tensor::InsertSliceOp::getOperationName()) {
      if (copyOp->getNumResults() == 0) {
        LDBG("[CopyRestore] reject copy without tensor results");
        return failure();
      }

      // When the copy lives inside scf.forall, apply the forall-aware
      // transformation: fold the copy into parallel_insert_slice and
      // optionally absorb the outer extract/insert pair.
      if (auto forallOp = copyOp->getParentOfType<scf::ForallOp>())
        return restoreCopyInsideForall(copyOp, forallOp, source, rewriter, loc);

      LDBG("[CopyRestore] replace insert_slice-form copy with source");
      rewriter.replaceOp(copyOp, source);
      return success();
    }

    return failure();
  }

private:
  //===--------------------------------------------------------------------===//
  // ForallOp-aware copy restoration
  //===--------------------------------------------------------------------===//

  /// Restores a linalg.copy inside scf.forall to its original
  /// tensor.parallel_insert_slice form, then optionally folds the outer
  /// extract_slice → forall → insert_slice chain by promoting the forall's
  /// shared_outs to the base tensor.
  ///
  /// BEFORE:
  ///   %ext = tensor.extract_slice %base[outerOff][outerSz][outerStr]
  ///   %r = scf.forall ... shared_outs(%arg = %ext) -> SliceType {
  ///     %d = tensor.extract_slice %arg[innerOff][innerSz][innerStr]
  ///     %cp = linalg.copy ins(%src) outs(%d) {original = "insert_slice"}
  ///     scf.forall.in_parallel {
  ///       tensor.parallel_insert_slice %cp into %arg[innerOff]...
  ///     }
  ///   }
  ///   %final = tensor.insert_slice %r into %base[outerOff][outerSz]...
  ///
  /// AFTER:
  ///   %r = scf.forall ... shared_outs(%arg = %base) -> BaseType {
  ///     %ext_inner = tensor.extract_slice %arg[outerOff][outerSz][outerStr]
  ///     ... (old %arg uses replaced by %ext_inner) ...
  ///     scf.forall.in_parallel {
  ///       tensor.parallel_insert_slice %src into %arg[outer+inner]...
  ///     }
  ///   }
  ///   // %final's uses replaced by %r
  LogicalResult restoreCopyInsideForall(linalg::CopyOp copyOp,
                                        scf::ForallOp forallOp, Value source,
                                        PatternRewriter &rewriter,
                                        Location loc) const {
    Value copyResult = copyOp.getResult(0);

    LDBG("[CopyRestore] attempt forall-aware restoration");

    // --- Step 1: Locate the parallel_insert_slice consuming the copy result
    // ---
    tensor::ParallelInsertSliceOp targetInsert = nullptr;
    for (Operation &op : forallOp.getTerminator().getYieldingOps()) {
      if (auto pis = dyn_cast<tensor::ParallelInsertSliceOp>(&op);
          pis && pis.getSource() == copyResult) {
        targetInsert = pis;
        break;
      }
    }

    if (!targetInsert) {
      LDBG("    -> No parallel_insert_slice uses the copy result; "
           "falling back to simple replacement.");
      rewriter.replaceOp(copyOp, source);
      return success();
    }

    LDBG("[CopyRestore] found matching parallel_insert_slice: "
         << *targetInsert);

    // The parallel_insert_slice's dest must be a forall output block argument.
    auto blockArg = dyn_cast<BlockArgument>(targetInsert.getDest());
    if (!blockArg || blockArg.getOwner() != forallOp.getBody()) {
      LDBG("[CopyRestore] destination is not a forall output block arg");
      rewriter.replaceOp(copyOp, source);
      return success();
    }

    // --- Step 2: Replace parallel_insert_slice source with copy's input ---
    rewriter.modifyOpInPlace(
        targetInsert, [&] { targetInsert.getSourceMutable().assign(source); });
    rewriter.eraseOp(copyOp);
    LDBG("[CopyRestore] replaced parallel_insert_slice source");

    // --- Step 3: Try to fold the outer extract_slice/insert_slice pair ---
    unsigned numIVs = forallOp.getRank();
    assert(blockArg.getArgNumber() >= numIVs);
    unsigned outIdx = blockArg.getArgNumber() - numIVs;
    return foldOuterSlicePair(forallOp, outIdx, blockArg, rewriter, loc);
  }

  /// Absorbs the outer extract_slice/insert_slice pair wrapping a forall by
  /// promoting the forall's shared_outs from the extracted slice to the base
  /// tensor.  Returns success unconditionally (the copy was already handled).
  LogicalResult foldOuterSlicePair(scf::ForallOp forallOp, unsigned outIdx,
                                   BlockArgument blockArg,
                                   PatternRewriter &rewriter,
                                   Location loc) const {
    Value initValue = forallOp.getOutputs()[outIdx];
    Value forallResult = forallOp.getResult(outIdx);

    // The init must come from an extract_slice for us to fold.
    auto outerExtract = initValue.getDefiningOp<tensor::ExtractSliceOp>();
    if (!outerExtract) {
      LDBG("[CopyRestore] skip outer fold: init is not extract_slice");
      return success();
    }

    Value baseTensor = outerExtract.getSource();
    SmallVector<OpFoldResult> outerOffsets = outerExtract.getMixedOffsets();
    SmallVector<OpFoldResult> outerSizes = outerExtract.getMixedSizes();
    SmallVector<OpFoldResult> outerStrides = outerExtract.getMixedStrides();

    LDBG("[CopyRestore] outer extract_slice base: " << baseTensor);

    // Find the matching outer tensor.insert_slice with the marker attribute.
    tensor::InsertSliceOp outerInsert = nullptr;
    for (Operation *user : forallResult.getUsers()) {
      auto is = dyn_cast<tensor::InsertSliceOp>(user);
      if (!is)
        continue;
      auto nameAttr = is->getAttrOfType<StringAttr>(kOriginalOpNameAttr);
      if (!nameAttr ||
          nameAttr.getValue() != tensor::InsertSliceOp::getOperationName())
        continue;
      if (is.getDest() != baseTensor)
        continue;
      outerInsert = is;
      break;
    }

    if (!outerInsert) {
      LDBG("[CopyRestore] skip outer fold: no matching outer insert_slice");
      return success();
    }

    LDBG("[CopyRestore] found outer tensor.insert_slice: " << *outerInsert);
    LDBG("[CopyRestore] promote forall shared_outs to base tensor");

    // --- 3a: Promote the forall's init, block arg type, and result type ---
    rewriter.modifyOpInPlace(forallOp, [&] {
      forallOp.getOutputsMutable().slice(outIdx, 1).assign(baseTensor);
      blockArg.setType(baseTensor.getType());
      forallResult.setType(baseTensor.getType());
    });

    // --- 3b: Insert extract_slice at the body entry so that existing
    //         non-terminator uses of the block arg still see the original
    //         (smaller) tensor type ---
    Value innerExtractVal;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(forallOp.getBody());
      auto innerExtract = rewriter.create<tensor::ExtractSliceOp>(
          loc, outerExtract.getType(), blockArg, outerOffsets, outerSizes,
          outerStrides);
      innerExtractVal = innerExtract.getResult();
      LDBG("[CopyRestore] insert body-entry extract_slice: " << *innerExtract);
    }

    // Replace block arg uses, keeping:
    //  (a) the new extract_slice's own source (it *is* the block arg), and
    //  (b) parallel_insert_slice dest operands (must write to the full tensor).
    blockArg.replaceUsesWithIf(innerExtractVal, [&](OpOperand &use) {
      if (use.getOwner() == innerExtractVal.getDefiningOp())
        return false;
      if (isa<tensor::ParallelInsertSliceOp>(use.getOwner()) &&
          use.getOperandNumber() == 1) // dest operand
        return false;
      return true;
    });

    // --- 3c: Compose offsets for every parallel_insert_slice that writes
    //         to this block argument ---
    {
      OpBuilder::InsertionGuard guard(rewriter);
      // Place offset-composition ops in the forall body, before the terminator.
      rewriter.setInsertionPoint(forallOp.getTerminator());

      for (Operation &op : llvm::make_early_inc_range(
               forallOp.getTerminator().getYieldingOps())) {
        auto pis = dyn_cast<tensor::ParallelInsertSliceOp>(&op);
        if (!pis || pis.getDest() != blockArg)
          continue;

        SmallVector<OpFoldResult> composedOffsets =
            composeOffsets(outerOffsets, pis.getMixedOffsets(), rewriter, loc);
        LDBG("[CopyRestore] compose offsets for parallel_insert_slice: "
             << *pis);

        // Rebuild the parallel_insert_slice inside the terminator region.
        OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPoint(pis);
        rewriter.create<tensor::ParallelInsertSliceOp>(
            loc, pis.getSource(), blockArg, composedOffsets,
            pis.getMixedSizes(), pis.getMixedStrides());
        rewriter.eraseOp(pis);
      }
    }

    // --- 3d: Replace outer insert_slice result with the promoted forall ---
    LDBG("[CopyRestore] replace outer tensor.insert_slice with forall result");
    rewriter.replaceOp(outerInsert, forallResult);

    return success();
  }

  //===--------------------------------------------------------------------===//
  // Memref-to-tensor source recovery
  //===--------------------------------------------------------------------===//

  /// Iteratively traces a memref back to a tensor source.
  /// Handles SubView chains by creating corresponding ExtractSliceOps.
  Value recoverTensorSource(Value val, PatternRewriter &rewriter,
                            Location loc) const {
    if (auto toBuffer = val.getDefiningOp<bufferization::ToBufferOp>())
      return toBuffer.getTensor();

    if (auto subview = val.getDefiningOp<memref::SubViewOp>()) {
      Value parentTensor =
          recoverTensorSource(subview.getSource(), rewriter, loc);
      if (!parentTensor)
        return nullptr;

      return rewriter.create<tensor::ExtractSliceOp>(
          loc, parentTensor, subview.getMixedOffsets(), subview.getMixedSizes(),
          subview.getMixedStrides());
    }

    if (auto viewOp = val.getDefiningOp<ViewLikeOpInterface>())
      return recoverTensorSource(viewOp.getViewSource(), rewriter, loc);

    return nullptr;
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct DeLinalgizePass
    : public mlir::dicp::LinalgExt::impl::DeLinalgizeBase<DeLinalgizePass> {
  using DeLinalgizeBase::DeLinalgizeBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<GenericToElementwisePattern, LinalgCopyToOriginalPattern>(
        context);

    // Use GreedyPatternRewriteDriver to handle potential chains of restorations
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::dicp::LinalgExt::createDeLinalgizePass() {
  return std::make_unique<DeLinalgizePass>();
}
