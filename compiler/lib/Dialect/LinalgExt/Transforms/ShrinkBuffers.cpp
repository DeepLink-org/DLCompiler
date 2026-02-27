#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"
#include "dicp/TransformOps/Transforms.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "buffer-shrink"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace dicp;
using namespace LinalgExt;

namespace mlir::dicp::LinalgExt {
#define GEN_PASS_DEF_SHRINKBUFFERS
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace mlir::dicp::LinalgExt

namespace {

//===----------------------------------------------------------------------===//
// Helper: Slice Parameter Analysis
//===----------------------------------------------------------------------===//

/// Holds extracted static parameters for a slice/view operation.
struct SliceParams {
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> strides;
  Type elementType;

  /// Checks if two SliceParams represent the exact same region and type.
  bool operator==(const SliceParams &other) const {
    return offsets == other.offsets && sizes == other.sizes &&
           strides == other.strides && elementType == other.elementType;
  }

  bool operator!=(const SliceParams &other) const { return !(*this == other); }
};

/// Tries to extract static slice parameters from an operation.
/// Returns failure if the op is not a supported slice/view or has dynamic
/// shapes.
LogicalResult getStaticSliceParams(Operation *op, SliceParams &params) {
  // Support both tensor.extract_slice and memref.subview
  auto iface = dyn_cast<OffsetSizeAndStrideOpInterface>(op);
  if (!iface)
    return failure();

  // 1. Check if all offsets, sizes, and strides are static.
  // We check the "static" arrays for any dynamic placeholders.
  auto staticOffsets = iface.getStaticOffsets();
  auto staticSizes = iface.getStaticSizes();
  auto staticStrides = iface.getStaticStrides();

  auto isDynamic = [](int64_t v) { return ShapedType::isDynamic(v); };

  if (llvm::any_of(staticOffsets, isDynamic) ||
      llvm::any_of(staticSizes, isDynamic) ||
      llvm::any_of(staticStrides, isDynamic)) {
    LLVM_DEBUG(llvm::dbgs()
               << "  [Shrink] Op has dynamic shapes: " << *op << "\n");
    return failure();
  }

  // 2. Use .assign() instead of '=' for SmallVector
  params.offsets.assign(staticOffsets.begin(), staticOffsets.end());
  params.sizes.assign(staticSizes.begin(), staticSizes.end());
  params.strides.assign(staticStrides.begin(), staticStrides.end());

  // Get element type for consistency checking
  if (auto shapedType = dyn_cast<ShapedType>(op->getResult(0).getType())) {
    params.elementType = shapedType.getElementType();
  } else {
    return failure();
  }

  return success();
}

/// Validates that the slice is contiguous (unit strides).
/// Shrinking a buffer with non-unit strides into a dense buffer changes data
/// layout, which breaks consumers expecting specific pointer arithmetic.
bool isContiguousSlice(const SliceParams &params) {
  return llvm::all_of(params.strides, [](int64_t s) { return s == 1; });
}


//===----------------------------------------------------------------------===//
// Pattern 1: ShrinkTensorEmpty
//===----------------------------------------------------------------------===//

/// Pattern to shrink `tensor.empty` size based on its usage.
///
/// Matches:
///   %0 = tensor.empty() : tensor<100x100xf32>
///   %1 = tensor.extract_slice %0[0,0][10,10][1,1] ...
///
/// And/Or:
///   %0 = tensor.empty() ...
///   %buf = bufferization.to_memref %0 ...
///   %view = memref.subview %buf[0,0][10,10][1,1] ...
///
/// Requirement: All users must point to the *same* static contiguous
/// sub-region.
struct ShrinkTensorEmpty : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern<tensor::EmptyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::EmptyOp emptyOp,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs()
               << "Analyze ShrinkTensorEmpty: " << emptyOp << "\n");

    SliceParams commonParams;
    bool isFirst = true;

    SmallVector<Operation *> extractSliceUsers;
    SmallVector<Operation *> subviewUsers;

    // 1. Analyze all direct users of tensor.empty
    for (Operation *user : emptyOp->getUsers()) {
      // Case A: Direct use by tensor.extract_slice
      if (isa<tensor::ExtractSliceOp>(user)) {
        SliceParams params;
        if (failed(getStaticSliceParams(user, params)))
          return failure();

        if (isFirst) {
          commonParams = params;
          isFirst = false;
        } else if (commonParams != params) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  [Shrink] Inconsistent extract_slice params.\n");
          return failure();
        }
        extractSliceUsers.push_back(user);
        continue;
      }

      // Case B: Direct use by bufferization.to_memref (or to_buffer)
      // Note: We check for ToBufferOp. If your dialect uses a different cast
      // (e.g., CastOp), adapt here.
      if (isa<bufferization::ToBufferOp>(user)) {
        // Check users of the buffer cast
        for (Operation *bufUser : user->getUsers()) {
          if (isa<memref::SubViewOp>(bufUser)) {
            SliceParams params;
            if (failed(getStaticSliceParams(bufUser, params)))
              return failure();

            if (isFirst) {
              commonParams = params;
              isFirst = false;
            } else if (commonParams != params) {
              LLVM_DEBUG(
                  llvm::dbgs()
                  << "  [Shrink] Inconsistent subview params via to_memref.\n");
              return failure();
            }
            subviewUsers.push_back(bufUser);
          } else {
            LLVM_DEBUG(llvm::dbgs() << "  [Shrink] to_memref has invalid user: "
                                    << *bufUser << "\n");
            return failure();
          }
        }
        continue;
      }

      // Invalid user found
      LLVM_DEBUG(llvm::dbgs()
                 << "  [Shrink] Unhandled user: " << *user << "\n");
      return failure();
    }

    if (isFirst) {
      LLVM_DEBUG(llvm::dbgs() << "  [Shrink] No valid users found.\n");
      return failure();
    }

    // 2. Safety Check: Ensure the slice is contiguous (stride 1).
    if (!isContiguousSlice(commonParams)) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "  [Shrink] Slice is not contiguous. Cannot shrink safely.\n");
      return failure();
    }

    // 3. Rewrite
    // Create new smaller tensor.empty
    // The result type is inferred from the slice size and element type.
    auto newRankedType =
        RankedTensorType::get(commonParams.sizes, commonParams.elementType);

    // We replace the consumers first, then the producer.
    rewriter.setInsertionPoint(emptyOp);
    auto newEmptyOp = rewriter.create<tensor::EmptyOp>(
        emptyOp.getLoc(), newRankedType, ValueRange{}); // No dynamic sizes

    // Replace all extract_slice users with the new empty tensor
    // (Since the new tensor *is* the slice, the extract operation is
    // redundant/identity)
    for (Operation *op : extractSliceUsers) {
      if (op->getResult(0).getType() != newRankedType) {
        // This implies rank-reduction or type mismatch, which we must handle or
        // bail. For simplicity in this pattern, we assume exact match or let
        // verification fail if complex. But usually extract_slice result type
        // == empty tensor of that size.
      }
      rewriter.replaceOp(op, newEmptyOp);
    }

    // Handle to_memref -> subview chains
    // We need a new to_memref converting the NEW empty tensor to a NEW memref
    if (!subviewUsers.empty()) {
      auto newMemRefType =
          MemRefType::get(commonParams.sizes, commonParams.elementType);
      auto newToMemref = rewriter.create<bufferization::ToBufferOp>(
          emptyOp.getLoc(), newMemRefType, newEmptyOp);

      for (Operation *subview : subviewUsers) {
        // Check if we need a cast. The subview result might have a specific
        // layout map (offset: ?). The new alloc is canonical (offset: 0).
        Value replacement = newToMemref;
        if (subview->getResult(0).getType() != newMemRefType) {
          replacement = rewriter.create<memref::CastOp>(
              subview->getLoc(), subview->getResult(0).getType(), newToMemref);
        }
        rewriter.replaceOp(subview, replacement);
      }
    }

    // tensor.empty itself doesn't have side effects, so if users are gone, it's
    // dead. However, we must ensure we don't leave dangling users of the
    // intermediate to_memref. The standard DCE will handle the old to_memref
    // and emptyOp if they have no users.
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 2: ShrinkMemRefAlloc
//===----------------------------------------------------------------------===//

/// Pattern to shrink `memref.alloc` size based on `subview` usage.
///
/// Matches:
///   %alloc = memref.alloc() : memref<128xf32>
///   %view = memref.subview %alloc[0][64][1] : ...
///
/// Requirements:
///   - Users of %alloc are ONLY subviews.
///   - All subviews access the *same* static contiguous region.
struct ShrinkMemRefAlloc : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs()
               << "Analyze ShrinkMemRefAlloc: " << allocOp << "\n");

    SliceParams commonParams;
    bool isFirst = true;
    SmallVector<memref::SubViewOp> subviewUsers;

    // 1. Analyze users
    for (Operation *user : allocOp->getUsers()) {
      auto subview = dyn_cast<memref::SubViewOp>(user);
      if (!subview) {
        LLVM_DEBUG(llvm::dbgs() << "  [Shrink] Alloc has non-subview user: "
                                << *user << "\n");
        return failure();
      }

      SliceParams params;
      if (failed(getStaticSliceParams(subview, params)))
        return failure();

      if (isFirst) {
        commonParams = params;
        isFirst = false;
      } else if (commonParams != params) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  [Shrink] Inconsistent subview definitions.\n");
        return failure();
      }
      subviewUsers.push_back(subview);
    }

    if (isFirst) {
      return failure(); // No users
    }

    // 2. Safety Check: Contiguous
    if (!isContiguousSlice(commonParams)) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "  [Shrink] Subview is not contiguous. Cannot shrink alloc.\n");
      return failure();
    }

    // 3. Rewrite
    // Construct the new MemRef type (Canonical layout, because it's a fresh
    // alloc)
    auto newMemRefType =
        MemRefType::get(commonParams.sizes, commonParams.elementType);

    rewriter.setInsertionPoint(allocOp);
    auto newAlloc =
        rewriter.create<memref::AllocOp>(allocOp.getLoc(), newMemRefType);

    // Propagate attributes (e.g., memory space, alignment) if necessary
    if (allocOp.getAlignment().has_value()) {
      newAlloc.setAlignment(allocOp.getAlignment().value());
    }
    // Note: We deliberately drop extra attributes like "dicp.npu.stage" unless
    // we know they remain valid. However, usually alloc attributes should be
    // preserved. rewriter.replaceOpWithNewOp handles result replacement, but
    // here we are changing types.

    for (auto subview : subviewUsers) {
      Value replacement = newAlloc;

      // If the subview result type differs from the new canonical alloc type
      // (e.g. strict stride info), we must cast the new alloc to the old
      // subview type to satisfy consumers.
      if (subview.getType() != newMemRefType) {
        replacement = rewriter.create<memref::CastOp>(
            subview.getLoc(), subview.getType(), newAlloc);
      }

      rewriter.replaceOp(subview, replacement);
    }

    // The old alloc has no users now (we replaced the subviews, which were the
    // only users).
    rewriter.eraseOp(allocOp);

    return success();
  }
};

struct ShrinkBuffersPass
    : public mlir::dicp::LinalgExt::impl::ShrinkBuffersBase<ShrinkBuffersPass> {
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);

    // Insert your two patterns. The pattern constructors below assume they take
    // MLIRContext* or PatternRewriter/OperationContext as appropriate.
    patterns.add<ShrinkMemRefAlloc, ShrinkTensorEmpty>(ctx);

    // Optionally: add canonicalization / folding patterns if helpful.
    // apply patterns greedily to the function.
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::dicp::LinalgExt::createShrinkBuffersPass() {
  return std::make_unique<ShrinkBuffersPass>();
}
