#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "scalar-to-1d-tensor"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::tensor;
using namespace mlir::memref;
using namespace mlir::arith;
using namespace mlir::func;

namespace mlir {
namespace dicp {
namespace LinalgExt {
#define GEN_PASS_DEF_SCALARTO1DTENSOR
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace LinalgExt
} // namespace dicp
} // namespace mlir

namespace {

/// Pattern: Rewrite `memref.store` of a scalar into a sequence using 1D tensors
/// and bufferization.
///
/// The scalar store is replaced with:
///  - tensor.empty<1 x elemTy> (at function entry)
///  - tensor.insert scalar into that empty tensor
///  - memref.reinterpret_cast (to create a 1D view at the store index)
///  - bufferization.materialize_in_destination (to write the tensor view)
///
/// Match conditions:
///  - The store uses a single index (1D logical access).
///  - The target memref is either:
///     (A) a function entry block argument.
///     (B) a memref.cast or memref.reinterpret_cast whose source is (A).
struct MemrefStoreToMaterializePattern
    : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = storeOp.getLoc();

    // Find the enclosing func op.
    auto func = storeOp->getParentOfType<FuncOp>();
    if (!func) {
      return rewriter.notifyMatchFailure(storeOp,
                                         "store not inside func, skipping");
    }

    // Only handle 1D index stores.
    if (storeOp.getIndices().size() != 1) {
      return rewriter.notifyMatchFailure(storeOp, "store not 1D, skipping");
    }
    Value storeIndex = storeOp.getIndices().front();
    Value storedValue = storeOp.getValue();
    Value memrefVal = storeOp.getMemRef();

    // Determine qualification (Case A or B)
    bool qualifies = false;
    Value castSrcOrArg = nullptr;
    Block *entry = &func.getBody().front();

    // Case A: direct block argument from function entry
    if (auto ba = dyn_cast_or_null<BlockArgument>(memrefVal)) {
      if (ba.getOwner() == entry) {
        qualifies = true;
        castSrcOrArg = memrefVal;
      }
    }

    // Case B: memref.cast or memref.reinterpret_cast whose source is func entry
    // block argument.
    if (!qualifies) {
      if (auto castOp = memrefVal.getDefiningOp<memref::CastOp>()) {
        Value src = castOp.getSource();
        if (auto srBa = dyn_cast_or_null<BlockArgument>(src)) {
          if (srBa.getOwner() == entry) {
            qualifies = true;
            castSrcOrArg = castOp.getSource(); // underlying source arg
          }
        }
      }
    }

    if (!qualifies) {
      if (auto castOp = memrefVal.getDefiningOp<memref::ReinterpretCastOp>()) {
        Value src = castOp.getSource();
        if (auto srBa = dyn_cast_or_null<BlockArgument>(src)) {
          if (srBa.getOwner() == entry) {
            qualifies = true;
            castSrcOrArg = castOp.getSource(); // underlying source arg
          }
        }
      }
    }

    if (!qualifies) {
      return rewriter.notifyMatchFailure(
          storeOp, "memref.store's memref is neither func arg "
                   "nor a cast from one; skipping");
    }

    // Determine element type and construct tensor<1 x elemTy>
    Type elemTy = storedValue.getType();
    RankedTensorType oneTensorTy = RankedTensorType::get({1}, elemTy);

    // Create tensor.empty at function entry insertion point.
    // We always create one, trading duplication for simplicity.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&func.getBody().front());
    Value emptyTensor =
        rewriter.create<tensor::EmptyOp>(loc, oneTensorTy, ValueRange{});

    // Reset insertion point to before the storeOp
    rewriter.setInsertionPoint(storeOp);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // Insert the scalar into the 1-element tensor
    Value inserted = rewriter.create<tensor::InsertOp>(
        loc, storedValue, emptyTensor, ValueRange{c0});

    // Prepare result memref type: memref<1 x elemTy>
    // MemRefType resultMemRefTy = MemRefType::get({1}, elemTy); // Unused

    // Use the non-casted source if available.
    Value reinterpretSource = memrefVal;
    if (auto castOp = memrefVal.getDefiningOp<memref::CastOp>())
      reinterpretSource = castOp.getSource();

    // Convert sizes/strides to OpFoldResult (using Attribute for static 1)
    SmallVector<OpFoldResult, 1> sizesOf;
    sizesOf.push_back(rewriter.getIndexAttr(1)); // Static size 1

    SmallVector<OpFoldResult, 1> stridesOf;
    stridesOf.push_back(rewriter.getIndexAttr(1)); // Static stride 1

    // Use the dynamic store index (storeIndex) as the offset OpFoldResult
    OpFoldResult offsetOf = storeIndex;

    // Call the ReinterpretCastOp creation method that accepts OpFoldResults.
    // This overload correctly sets static_sizes/static_strides attributes.
    Value reinterpretCast = rewriter.create<memref::ReinterpretCastOp>(
        loc, /*source*/ reinterpretSource, /*offset*/ offsetOf,
        /*sizes*/ sizesOf, /*strides*/ stridesOf,
        /*attrs*/ ArrayRef<NamedAttribute>{});

    // Materialize the inserted tensor into the 1D view of the memref.
    rewriter.setInsertionPoint(storeOp);
    rewriter.create<bufferization::MaterializeInDestinationOp>(
        loc, TypeRange{}, inserted, reinterpretCast, false, true);

    // Erase the original store.
    rewriter.eraseOp(storeOp);

    return success();
  }
};

/// The pass: ScalarTo1DTensorPass
struct ScalarTo1DTensorPass
    : public mlir::dicp::LinalgExt::impl::ScalarTo1DTensorBase<
          ScalarTo1DTensorPass> {
  ScalarTo1DTensorPass() = default;
  void runOnOperation() override {
    FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    // Build patterns
    RewritePatternSet patterns(context);
    patterns.add<MemrefStoreToMaterializePattern>(context);

    // Apply patterns greedily on this function
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::dicp::LinalgExt::createScalarTo1DTensorPass() {
  return std::make_unique<ScalarTo1DTensorPass>();
}