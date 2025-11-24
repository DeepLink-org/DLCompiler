#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"
#include "dicp/Dialect/LinalgExt/Transforms/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

#define DEBUG_TYPE "linalg-generic-to-scf"

using namespace mlir;
using namespace dicp;
using namespace LinalgExt;

namespace mlir::dicp::LinalgExt {
#define GEN_PASS_DEF_LINALGGENERICTOSCF
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace mlir::dicp::LinalgExt

namespace {

/**
 * @brief Converts a multi-dimensional parallel linalg.generic (on tensors) into
 * nested scf.for loops.
 *
 * Handles cases with and without outputs.
 *
 * Case 1: With Outputs (Accumulator/Map style)
 * %res = linalg.generic ... ins(...) outs(%out)
 * ->
 * %res = scf.for ... iter_args(%curr = %out) {
 * ...
 * %new = tensor.insert ...
 * scf.yield %new
 * }
 *
 * Case 2: Without Outputs (Side-effecting/Void style)
 * linalg.generic ... ins(...) outs()
 * ->
 * scf.for ... {
 * ...
 * scf.yield
 * }
 */
struct LinalgGenericToScfForPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    // --- 0. Pre-check for specific IndexCast case ---
    auto linalgYield = mlir::dyn_cast_or_null<linalg::YieldOp>(
        linalgOp.getBody()->getTerminator());
    if (linalgYield && linalgYield.getNumOperands() == 1) {
      Value yieldValue = linalgYield.getOperand(0);
      if (llvm::isa_and_nonnull<arith::IndexCastOp>(
              yieldValue.getDefiningOp())) {
        return rewriter.notifyMatchFailure(
            linalgOp, "Conversion skipped: index increment detected.");
      }
    }

    // --- 1. Check LinalgOp Properties ---
    // Check: all iterator types must be "parallel"
    if (!llvm::all_of(linalgOp.getIteratorTypesArray(),
                      linalg::isParallelIterator)) {
      return rewriter.notifyMatchFailure(
          linalgOp, "Failed match: expected all parallel iterators");
    }

    // Check: all indexing maps must be identity
    for (AffineMap map : linalgOp.getIndexingMapsArray()) {
      if (!map.isIdentity()) {
        return rewriter.notifyMatchFailure(
            linalgOp, "Conversion failed: expected all maps to be identity.");
      }
    }

    // Check Output presence
    // We allow 0 outputs or exactly 1 output.
    if (linalgOp.getNumDpsInits() > 1) {
      return rewriter.notifyMatchFailure(
          linalgOp, "Conversion failed: >1 outputs not supported yet.");
    }

    bool hasOutput = (linalgOp.getNumDpsInits() == 1);
    Value outputTensor = nullptr;

    if (hasOutput) {
      outputTensor = linalgOp.getDpsInitOperand(0)->get();
      // Verify output comes from tensor.empty if present
      auto emptyOp = outputTensor.getDefiningOp<tensor::EmptyOp>();
      if (!emptyOp) {
        return rewriter.notifyMatchFailure(
            linalgOp,
            "Conversion failed: output must originate from tensor.empty.");
      }
    }

    // --- 2. Determine Loop Bounds ---
    // Use output shape if available, otherwise use the first input shape.
    ShapedType boundsType;
    if (hasOutput) {
      boundsType = mlir::cast<ShapedType>(outputTensor.getType());
    } else {
      if (linalgOp.getNumDpsInputs() == 0) {
        return rewriter.notifyMatchFailure(
            linalgOp,
            "Conversion failed: no inputs or outputs to determine bounds.");
      }
      Value firstInput = linalgOp.getDpsInputOperand(0)->get();
      boundsType = mlir::cast<ShapedType>(firstInput.getType());
    }

    if (!boundsType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(linalgOp,
                                         "Dynamic shapes are not supported.");
    }
    ArrayRef<int64_t> shape = boundsType.getShape();
    int64_t rank = boundsType.getRank();

    if (rank == 0) {
      return rewriter.notifyMatchFailure(linalgOp, "Rank 0 not supported.");
    }

    Location loc = linalgOp.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // --- 3. Build Nested Loops ---
    SmallVector<scf::ForOp> loops;
    SmallVector<Value> ivs;

    // The 'current' tensor being threaded through iter_args.
    Value currentThreadedTensor = hasOutput ? outputTensor : nullptr;

    // Create loops from outer (dim 0) to inner (dim rank-1).
    for (int64_t i = 0; i < rank; ++i) {
      Value ub = rewriter.create<arith::ConstantIndexOp>(loc, shape[i]);

      // Prepare iter_args for this specific loop level
      SmallVector<Value> iterArgs;
      if (hasOutput) {
        iterArgs.push_back(currentThreadedTensor);
      }

      auto forOp = rewriter.create<scf::ForOp>(loc, c0, ub, c1, iterArgs);

      loops.push_back(forOp);
      ivs.push_back(forOp.getInductionVar());

      rewriter.setInsertionPointToStart(forOp.getBody());

      // Update threaded tensor for the next inner loop
      if (hasOutput) {
        currentThreadedTensor = forOp.getRegionIterArg(0);
      }

      if (linalgOp->hasAttr("ExtractedLoadOrStore")) {
        forOp->setAttr("ExtractedLoadOrStore", rewriter.getUnitAttr());
      }
    }

    // --- 4. Build Innermost Body ---
    IRMapping map;
    Block *linalgBody = linalgOp.getBody();

    unsigned inputIdx = 0;
    for (BlockArgument &arg : linalgBody->getArguments()) {
      Value newSource;

      if (arg.getArgNumber() < linalgOp.getNumDpsInputs()) {
        // --- Map Input ---
        Value inputTensor = linalgOp.getDpsInputOperand(inputIdx++)->get();
        auto extractOp =
            rewriter.create<tensor::ExtractOp>(loc, inputTensor, ivs);
        newSource = extractOp;
      } else {
        // --- Map Output (Accumulator) ---
        auto extractOp =
            rewriter.create<tensor::ExtractOp>(loc, currentThreadedTensor, ivs);
        newSource = extractOp;
      }
      map.map(arg, newSource);
    }

    // Clone linalg.generic operations (excluding linalg.yield)
    for (Operation &op : linalgBody->without_terminator()) {
      rewriter.clone(op, map);
    }

    // --- 5. Handle Yield (Innermost) ---
    SmallVector<Value> innerYieldOperands;

    if (hasOutput) {
      // If there is an output, insert the calculated scalar back into the
      // tensor.
      Value computedScalar = map.lookup(linalgYield.getOperand(0));
      Value insertedTensor = rewriter.create<tensor::InsertOp>(
          loc, computedScalar, currentThreadedTensor, ivs);
      innerYieldOperands.push_back(insertedTensor);
    }
    if (!innerYieldOperands.empty())
      auto innerYield = rewriter.create<scf::YieldOp>(loc, innerYieldOperands);

    // --- 6. Handle Yields for Outer Loops ---
    // Walk back up the loop nest.
    for (int64_t i = rank - 2; i >= 0; --i) {
      scf::ForOp innerLoop = loops[i + 1];

      // Set insertion point to after the inner loop
      rewriter.setInsertionPointAfter(innerLoop);

      // Yield the result of the inner loop (if any)
      rewriter.create<scf::YieldOp>(loc, innerLoop.getResults());
    }

    // --- 7. Finalize Replacement ---
    if (hasOutput) {
      rewriter.replaceOp(linalgOp, loops[0].getResults());
    } else {
      rewriter.eraseOp(linalgOp);
    }

    return success();
  }
};

struct LinalgGenericToSCFPass
    : mlir::dicp::LinalgExt::impl::LinalgGenericToSCFBase<
          LinalgGenericToSCFPass> {

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();
    {
      RewritePatternSet patterns(context);
      patterns.add<LinalgGenericToScfForPattern>(context);
      populateRemoveSingleIterationLoopPattern(patterns);
      if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
        signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::dicp::LinalgExt::createLinalgGenericToSCFPass() {
  return std::make_unique<LinalgGenericToSCFPass>();
}