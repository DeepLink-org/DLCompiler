#ifndef DICP_LINALGEXT_TRANSFORMS_H_
#define DICP_LINALGEXT_TRANSFORMS_H_

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir::dicp::LinalgExt {

using ForControlFnRef = llvm::function_ref<bool(scf::ForOp)>;

/// Insert pattern to remove single iteration loop. The pattern will detect
/// single iteration loops based on the range returned ValueBoundsOpInterface.
void populateRemoveSingleIterationLoopPattern(
    RewritePatternSet &patterns, ForControlFnRef controlFn = nullptr);

/**
 * Populates the given pattern set with rewrite patterns that lift
 * scalar arithmetic select operations out of Linalg generic ops.
 *
 * This typically includes patterns like:
 * - LiftScalarSelectToTensorPattern: Converts a scalar arith.select
 * with loop-invariant operands inside a linalg.generic to a
 * tensor arith.select placed outside the generic op.
 * - LiftYieldSelectOutPattern: Converts an arith.select yielded by
 * the linalg.generic body into an outside tensor arith.select.
 *
 */
void populateLinalgLiftSelectPattern(RewritePatternSet &patterns);

/// This pattern detects a chain of `tensor::InsertSliceOp` that together
/// implement an interleave write: multiple source tensors are inserted into the
/// same destination along the last dimension using different static offsets.
/// Once detected, the pattern normalizes this chain into a canonical form
/// where the last-dimension offsets are [0..channelNum-1] and the stride is
/// `channelNum`, making the interleave structure explicit and ready for further
/// fusion or replacement by a dedicated Interleave op.
void populateNormalizeInsertSliceOpInInterleavePattern(RewritePatternSet &patterns);

} // namespace mlir::dicp::LinalgExt

#endif