#ifndef DICP_DIALECT_LINALGEXT_TRANSFORMS_DIMANALYZER_H
#define DICP_DIALECT_LINALGEXT_TRANSFORMS_DIMANALYZER_H

#include "dicp/Dialect/LinalgExt/Analysis/StageDependencyAnalyzer.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <numeric>
#include <queue>
#include <vector>

namespace mlir {
namespace dicp {

/// Classification of a dimension's role in the computation graph.
/// This helps determine if a dimension is safe to tile or parallelize.
enum class DimKind {
  Unknown,   // No specific property inferred yet.
  Parallel,  // Dimension implies independent iterations (safe to tile).
  Reduction, // Dimension is collapsed/reduced (requires accumulation).
  Broadcast, // Dimension is replicated (data invariant along this axis).
  Complex    // Dimension undergoes complex transformation (e.g., non-affine
             // reshape).
};

std::string toString(DimKind k);

/// Disjoint Set Union (DSU) for tracking dimension equivalence and properties.
///
/// This class implements a Disjoint Set data structure (Union-Find)
/// specifically designed for Tensor/MemRef dimensions. It serves two main
/// purposes:
/// 1. **Equivalence Tracking**: Determines which dimensions across different
/// values
///    represent the same logical axis (e.g., the 'N' dimension in a Matmul
///    propagating through element-wise adds).
/// 2. **Property Propagation**: Merges semantic properties (DimKind) when
/// dimensions
///    are unified. For example, if a dimension is used as a Reduction iterator
///    in one operation, that property propagates to all equivalent dimensions
///    in the set.
class DimensionDisjointSet {
public:
  explicit DimensionDisjointSet(size_t size = 0) { resize(size); }

  /// Allocates `n` new dimension IDs in the set.
  /// \return The ID of the first allocated dimension.
  int64_t allocate(size_t n = 1);

  /// Finds the representative (root) ID for the set containing dimension `i`.
  /// Implements path compression for amortized constant time lookups.
  int64_t find(int64_t i);

  /// Merges the sets containing dimensions `i` and `j`.
  /// This also merges the `DimKind` properties of both roots using
  /// `mergeKinds`.
  void unionSets(int64_t i, int64_t j);

  /// Updates the DimKind property for the set containing dimension `i`.
  /// The new kind is merged with the existing kind to ensure safety (e.g.,
  /// Reduction is sticky).
  void setKind(int64_t i, DimKind k);

  /// Retrieves the DimKind property of the set containing dimension `i`.
  DimKind getKind(int64_t i);

private:
  /// Resizes the internal storage to accommodate `n` dimensions.
  void resize(size_t n);

  /// Defines the logic for combining two dimension kinds.
  /// Hierarchy of "stickiness": Complex > Reduction > Broadcast/Parallel.
  DimKind mergeKinds(DimKind a, DimKind b);

  std::vector<int64_t> parent; // Parent pointers for DSU.
  std::vector<DimKind> kind;   // Properties associated with each root.
};

/// DimAnalyzer:
/// Analyzes a specific execution stage (StageInfo) to determine tiling
/// strategies.
///
/// The analyzer constructs a constraint graph where nodes are tensor dimensions
/// and edges represent data flow relationships. It uses a Breadth-First Search
/// (BFS) approach to traverse operations and propagate dimension IDs.
///
/// Algorithm Overview:
/// 1. **Initialization**: Seeds the analysis with stage inputs (operands
/// defined outside the stage).
/// 2. **BFS Propagation**: Traverses the def-use chains. For each operation, it
/// uses specific handlers (e.g., processMatmulOp) to bind input dimensions to
/// output dimensions.
/// 3. **Anchor Heuristic**: Identifies the "Anchor" operation (typically the
/// final LinalgOp) to interpret the resulting loops.
/// 4. **Tiling Selection**: Checks the properties of the Anchor's loops in the
///   DSU to recommend outermost parallel loops for tiling.
class DimAnalyzer {
public:
  explicit DimAnalyzer(const StageInfo &stage);

  /// Analyzes the stage operations and returns indices of loops recommended for
  /// tiling. The indices correspond to the loop nest of the "Anchor" operation.
  SmallVector<int64_t> analyzeAndGetTilingDims();

private:
  const StageInfo &stage_;
  // Quick lookup for ops belonging to this stage.
  DenseSet<Operation *> stageOps_;
  DimensionDisjointSet dsu_;
  // Maps SSA Value -> [Dim IDs]
  DenseMap<Value, std::vector<int64_t>> valueDims_;

  // BFS State passed to handlers to allow them to enqueue new values.
  using BFSQueue = std::queue<Value>;
  using VisitedSet = DenseSet<Value>;

  /// Drives the traversal of the data flow graph.
  void processBFS();

  /// Dispatches the operation to the appropriate handler.
  /// \return true if the operation was handled, false otherwise.
  bool processOperation(Operation *op, Value current, BFSQueue &q,
                        VisitedSet &v);

  /// Lazily retrieves or allocates unique IDs for the dimensions of a Value.
  std::vector<int64_t> getOrAllocateDims(Value v);

  /// Helper to strictly bind all dimensions of v1 to v2 (1-to-1 mapping).
  /// Used for Elementwise, Copy, etc.
  void bindDimensions(Value v1, Value v2);

  // --- Op Handlers ---
  // Each handler interprets the semantics of the op to union input/output
  // dimensions correctly.

  void processElementwise(Operation *op, Value current);
  void processMatmulOp(linalg::MatmulOp op);
  void processReduceOp(linalg::ReduceOp op);
  void processTransposeOp(linalg::TransposeOp op);
  void processBroadcastOp(linalg::BroadcastOp op);
  void processLinalgOpGeneric(linalg::LinalgOp op);
  void processReshapeOp(Operation *op);
  void processConcatOp(tensor::ConcatOp op);
  void processPadOp(tensor::PadOp op);
  void processExtractSliceOp(tensor::ExtractSliceOp op);
  void processInsertSliceOp(tensor::InsertSliceOp op);

  // Handlers that may need to continue BFS propagation explicitly
  void processMemrefCopyOp(memref::CopyOp op, Value current, BFSQueue &q,
                           VisitedSet &v);
  void processMemrefCastOp(Operation *op);
  void processBufferizationToTensor(bufferization::ToTensorOp op);
  void processMaterializeOp(bufferization::MaterializeInDestinationOp op,
                            Value current, BFSQueue &q, VisitedSet &v);
};

} // namespace dicp
} // namespace mlir

#endif