#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"
#include "dicp/TransformOps/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "de-linalgize"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace mlir::dicp;

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

    LDBG("Analyzing generic op for restoration: " << genericOp);

    if (!mlir::linalg::isElementwise(genericOp)) {
      LDBG("  -> Failed: Not an identity-mapped elementwise generic.");
      return failure();
    }

    Block &body = genericOp.getRegion().front();
    if (body.getOperations().size() != 2) { // [ScalarOp, YieldOp]
      LDBG("  -> Failed: Body size is " << body.getOperations().size()
                                        << ", expected 2.");
      return failure();
    }

    Operation *scalarOp = &body.front();
    StringRef opName = scalarOp->getName().getStringRef();

    // Only restore arith and math dialects
    if (!opName.starts_with("arith.") && !opName.starts_with("math.")) {
      LDBG("  -> Failed: Inner op " << opName << " is not arith/math.");
      return failure();
    }

    // Map block arguments back to generic operands
    SmallVector<Value> newOperands;
    for (Value operand : scalarOp->getOperands()) {
      auto arg = dyn_cast<BlockArgument>(operand);
      if (!arg || arg.getOwner() != &body) {
        LDBG("  -> Failed: Operand is not a block argument of the generic.");
        return failure();
      }

      unsigned argIdx = arg.getArgNumber();
      if (argIdx >= genericOp.getNumDpsInputs()) {
        LDBG("  -> Failed: Operation uses output/accumulator as input.");
        return failure();
      }
      newOperands.push_back(genericOp.getDpsInputOperand(argIdx)->get());
    }

    LDBG("  -> Restoring to " << opName);

    // Filter attributes: keep scalar op attributes and carry over NPU stage
    SmallVector<NamedAttribute> newAttrs;
    for (auto attr : scalarOp->getAttrs())
      newAttrs.push_back(attr);

    Operation *newOp = rewriter.create(
        genericOp.getLoc(), rewriter.getStringAttr(opName), newOperands,
        genericOp.getResultTypes(), scalarOp->getAttrs());

    rewriter.replaceOp(genericOp, newOp->getResults());
    return success();
  }
};

/**
 * @brief Restores linalg.copy to memref.copy or materialization.
 *
 * Specifically handles the restoration of bufferization artifacts that were
 * lowered to linalg.copy for generic transformation passes.
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

    LDBG("Restoring linalg.copy to original op: " << originalName);

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
        LDBG("  -> Failed to recover tensor source for materialization.");
        return failure();
      }

      auto matOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
          loc, tensorSource, dest);
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

    return failure();
  }

private:
  /**
   * @brief Iteratively traces a memref back to a tensor source.
   * Handles SubView chains by creating corresponding ExtractSliceOps.
   */
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

    // Handle generic view-like interfaces if necessary
    if (auto viewOp = val.getDefiningOp<ViewLikeOpInterface>()) {
      return recoverTensorSource(viewOp.getViewSource(), rewriter, loc);
    }

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

    {
      PassManager pm(&getContext(), module.getOperationName());
      pm.addPass(createCSEPass());
      pm.addPass(createCanonicalizerPass());
      if (failed(runPipeline(pm, module))) {
        LDBG("Final cleanup pipeline failed.");
        signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::dicp::LinalgExt::createDeLinalgizePass() {
  return std::make_unique<DeLinalgizePass>();
}