#ifndef MLIR_ANALYSIS_SLICEPARALLELANALYSIS_H
#define MLIR_ANALYSIS_SLICEPARALLELANALYSIS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir::dicp {

//===----------------------------------------------------------------------===//
// MemoryRegion
//===----------------------------------------------------------------------===//

/// A best-effort region abstraction for buffer-typed SSA values.
///
/// `baseRoot` is the underlying storage root after stripping buffer/tensor
/// adapters and view-like wrappers. When `hasPreciseSlice` is true, the
/// `(offsets, sizes, strides)` triple describes the accessed slice relative to
/// `baseRoot`. Otherwise, the region conservatively represents the entire root
/// buffer.
struct MemoryRegion {
  Value baseRoot;
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  bool hasPreciseSlice = false;

  bool isValid() const { return static_cast<bool>(baseRoot); }
};

/// Returns a best-effort memory region for the given value.
std::optional<MemoryRegion> getMemoryRegion(Value value);

/// Proves whether two precise regions on the same base root are disjoint.
/// Returns failure when either region lacks precise slice information or the
/// regions are not comparable.
FailureOr<bool> proveMemoryRegionsDisjoint(const MemoryRegion &lhs,
                                           const MemoryRegion &rhs);

//===----------------------------------------------------------------------===//
// SliceRegion
//===----------------------------------------------------------------------===//

/// Represents a unified view of a slice operation.
/// Abstracts over `tensor.insert_slice`, `tensor.extract_slice`, and
/// `memref.subview`.
struct SliceRegion {
  Value baseTensor;
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  Operation *op;
  bool isWrite;
};

//===----------------------------------------------------------------------===//
// SliceRegionAnalysis
//===----------------------------------------------------------------------===//

/// An analysis dedicated to extracting unified `SliceRegion`s from various IR
/// ops.
class SliceRegionAnalysis {
public:
  SliceRegionAnalysis() = default;

  /// Unifies standard slicing operators into a common SliceRegion
  /// representation.
  std::optional<SliceRegion> getSliceRegion(Operation *op) const;
};

//===----------------------------------------------------------------------===//
// DisjointSliceAnalysis
//===----------------------------------------------------------------------===//

/// This analysis proves whether a `memref.alloc` is exclusively and safely
/// partitioned into a set of non-overlapping `memref.subview` operations.
/// It leverages `SliceAnalysis` for transitive escape detection and
/// `ValueBoundsConstraintSet` for rigorous geometric disjointness proofs.
/// The analysis currently expects slice-like users rooted in
/// `OffsetSizeAndStrideOpInterface` and deduplicates equivalent subviews
/// before proving pairwise disjointness.
class DisjointSliceAnalysis {
public:
  DisjointSliceAnalysis() = default;

  /// Analyzes a root value and returns a list of disjoint slice operations.
  FailureOr<SmallVector<OffsetSizeAndStrideOpInterface>>
  analyze(Value rootValue) const;

  /// Analyzes the single result of an operation for disjoint slices.
  FailureOr<SmallVector<OffsetSizeAndStrideOpInterface>>
  analyze(Operation *op) const;

private:
  LogicalResult
  collectSlices(Value root,
                SmallVectorImpl<OffsetSizeAndStrideOpInterface> &slices) const;

  LogicalResult
  checkTransitiveEscape(ArrayRef<OffsetSizeAndStrideOpInterface> slices) const;

  LogicalResult
  proveDisjointness(ArrayRef<OffsetSizeAndStrideOpInterface> slices) const;

  bool areDisjoint(OffsetSizeAndStrideOpInterface a,
                   OffsetSizeAndStrideOpInterface b) const;
};

//===----------------------------------------------------------------------===//
// Shared Utilities
//===----------------------------------------------------------------------===//

/// Check whether two 1D ranges are provably disjoint.
///
/// The proof is delegated to MLIR's value-bounds infrastructure, so this can
/// handle both static and value-bounded dynamic ranges.
bool areRangesDisjoint1D(OpFoldResult offsetA, OpFoldResult sizeA,
                         OpFoldResult strideA, OpFoldResult offsetB,
                         OpFoldResult sizeB, OpFoldResult strideB);

/// Compare two OpFoldResult values for equality.
///
/// Equality is proven through MLIR's value-bounds infrastructure. If equality
/// cannot be established, this function conservatively returns false.
bool isEqualOFR(OpFoldResult a, OpFoldResult b);

/// Convert each OpFoldResult to a static IndexAttr. Returns failure if any
/// value is not a compile-time constant.
FailureOr<SmallVector<OpFoldResult>>
toStaticIndexAttrs(OpBuilder &b, ArrayRef<OpFoldResult> ofrs);

/// Verifies that disjoint tensor partitions are independently computed:
///   1. No partition's forward compute slice writes to a forbidden memref
///      target (cross-partition write independence).
///   2. No partition's forward compute slice escapes to a function call.
///
/// \param partitionRoots       partitionRoots[i] = tensor values seeding
///                             partition i's computation (extract_slice
///                             results)
/// \param forbiddenWriteTargets memref values that no partition's compute
///                              is allowed to write to (e.g. alloc + subviews)
LogicalResult
verifyPartitionIsolation(ArrayRef<SmallVector<Value>> partitionRoots,
                         const DenseSet<Value> &forbiddenWriteTargets);

//===----------------------------------------------------------------------===//
// InsertSliceChainAnalysis
//===----------------------------------------------------------------------===//

/// Represents a disjoint partition of a tensor buffer identified by its
/// static offset / size / stride tuple.
struct SlicePartition {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  RankedTensorType sliceType;
};

/// Semantic partitions discovered from direct slice users of a single root.
struct DirectSlicePartitionInfo {
  SmallVector<SlicePartition> partitions;
  DenseMap<Operation *, unsigned> sliceToPartIdx;
};

/// Find the index of the partition whose offsets/sizes/strides match.
/// Returns -1 if no match is found.
int findMatchingPartition(ArrayRef<SlicePartition> partitions,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes,
                          ArrayRef<OpFoldResult> strides);

/// Group direct slice users by semantic slice equivalence instead of op
/// identity.
FailureOr<DirectSlicePartitionInfo>
buildDirectSlicePartitionInfo(ArrayRef<OffsetSizeAndStrideOpInterface> slices);

/// Verify that semantic partitions are pairwise disjoint.
LogicalResult
verifyDirectSlicePartitionsDisjoint(ArrayRef<SlicePartition> partitions);

/// Preserve the legacy direct-slice conservatism: direct slice results must
/// not escape to calls.
LogicalResult
verifyDirectSliceNoCallEscape(ArrayRef<OffsetSizeAndStrideOpInterface> slices);

/// Analyzes a tensor.insert_slice chain rooted at a given value and
/// decomposes it into disjoint partitions.
///
/// A **chain** is the set of SSA values formed by the root and the results
/// of every tensor.insert_slice whose `dest` is a chain value.  All uses of
/// a chain value must be one of:
///   - tensor.extract_slice  (source == chain value)
///   - tensor.insert_slice   (dest   == chain value)
///   - a supported pure tensor op such as elementwise/broadcast/reduce/matmul
///   - An operation accepted by the caller-provided `allowedUser` callback
///
/// If the chain can be decomposed into >= 2 pairwise-disjoint partitions,
/// analysis succeeds and returns the partition information.
class InsertSliceChainAnalysis {
public:
  struct Result {
    /// The identified disjoint partitions.
    SmallVector<SlicePartition> partitions;
    /// Maps each slice op in the chain to its partition index.
    DenseMap<Operation *, unsigned> sliceToPartIdx;
    /// All SSA values on the chain (root + insert_slice results).
    DenseSet<Value> chainValues;
    /// All extract_slice / insert_slice ops collected on the chain.
    SmallVector<OffsetSizeAndStrideOpInterface> slices;
  };

  InsertSliceChainAnalysis() = default;

  /// Analyze the insert_slice chain rooted at `root`.
  /// `allowedUser` is called for any non-chain user; return true to accept.
  FailureOr<Result>
  analyze(Value root,
          function_ref<bool(Operation *)> allowedUser = nullptr) const;
};

//===----------------------------------------------------------------------===//
// LoopSliceParallelAnalysis
//===----------------------------------------------------------------------===//

/// Analyzes an scf::ForOp to determine if it is semantically safe to be
/// converted into an scf::forall, ensuring memory & execution independence.
class LoopSliceParallelAnalysis {
public:
  LoopSliceParallelAnalysis() = default;

  /// Entry point to analyze the given `scf.for` loop for `scf.forall` validity.
  LogicalResult analyze(scf::ForOp loop) const;

private:
  SliceRegionAnalysis sliceAnalyzer;

  /// Phase 1: Side Effect Verification.
  LogicalResult
  collectAndVerifySideEffects(scf::ForOp loop,
                              SmallVectorImpl<SliceRegion> &writes,
                              SmallVectorImpl<SliceRegion> &reads) const;

  /// Phase 2: Cross-Iteration Geometric Disjointness.
  LogicalResult
  verifyCrossIterationDisjointness(scf::ForOp loop,
                                   ArrayRef<SliceRegion> writes) const;

  /// Phase 3: No Cross-Iteration RAW/WAR Checks.
  LogicalResult
  verifyNoCrossIterationDependencies(scf::ForOp loop,
                                     ArrayRef<SliceRegion> writes,
                                     ArrayRef<SliceRegion> reads) const;

  // --- Utility Methods ---

  Value getRootValue(Value val, scf::ForOp loop) const;
  bool isDerivedFromSubview(Value val) const;
  bool areSlicesIdentical(const SliceRegion &a, const SliceRegion &b) const;
  bool isEqualOFR(OpFoldResult a, OpFoldResult b) const;

  /// Statically determines the linear relationship multiplier `a` mapping the
  /// IV to a given value.
  std::optional<int64_t> getIVMultiplier(Value val, scf::ForOp loop) const;

  /// Recurses through MLIR Affine Expressions to dynamically compose the total
  /// linear step mapping.
  std::optional<int64_t> evaluateAffineMultiplier(AffineExpr expr,
                                                  ValueRange operands,
                                                  AffineMap map,
                                                  scf::ForOp loop) const;
};

} // namespace mlir::dicp

#endif // MLIR_ANALYSIS_SLICEPARALLELANALYSIS_H
