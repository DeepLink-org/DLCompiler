#ifndef DICP_DIALECT_LINALGEXT_TRANSFORMS_STAGEDEPENDENCYANALYZER_H
#define DICP_DIALECT_LINALGEXT_TRANSFORMS_STAGEDEPENDENCYANALYZER_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SetVector.h"

#include <set>
#include <vector>

namespace mlir {
namespace dicp {

/// Defines the execution unit type for the stage.
enum class StageType {
  Vector, // Default: General vector or scalar operations
  Cube    // Matrix operations (e.g., Matmul)
};

/// Represents a single pipeline stage.
/// A stage is a sequence of operations that execute together.
/// Synchronization operations (SyncBlockWaitOp) typically delimit stage
/// boundaries.
struct StageInfo {
  int id = -1;
  SmallVector<Operation *, 8> ops;
  // IDs of stages that this stage depends on
  std::set<int> preds;
  // IDs of stages that depend on this stage
  std::set<int> succs;
  bool hasSync = false;
  // Stage execution type
  StageType type = StageType::Vector;
};

// StageDependencyAnalyzer:
// 1. Partitioning a block into "stages" based on synchronization primitives
//    (hivm::SyncBlockWaitOp).
// 2. Building a dependency graph between these stages considering both:
//    - SSA Data Flow (Producer-Consumer relationships).
//    - Memory Dependencies (Read-After-Write via AliasAnalysis).
// 3. Computing a topological ordering (levels) to detect cycles and determine
//    a valid execution schedule.
// 4. Physically reordering the IR operations to match the valid schedule.
//
class StageDependencyAnalyzer {
public:
  StageDependencyAnalyzer(Block *block, AliasAnalysis &aliasAnalysis)
      : block(block), aliasAnalysis(aliasAnalysis) {}

  /// Runs the analysis, computes the topological sort, and physically reorders
  /// the operations in the block.
  /// Returns the ordered list of StageInfo on success, or failure if a cycle is
  /// detected.
  FailureOr<std::vector<StageInfo>> runAndReorder(RewriterBase &rewriter);

  /// Scans the block to populate the `stages` vector.
  FailureOr<std::vector<StageInfo>> collectStages();

private:
  /// Internal node structure for the dependency graph.
  struct StageNode {
    int id;
    StageInfo *stageInfo;
    int level = 0; // Topological level (depth)

    // Memory dependencies
    llvm::SetVector<Value> readValues;
    llvm::SetVector<Value> writeValues;

    // SSA Value dependencies
    llvm::SetVector<Value> producedValues; // Values defined in this stage
    llvm::SetVector<Value> consumedValues; // Values used in this stage
  };

  Block *block;
  AliasAnalysis &aliasAnalysis;
  std::vector<StageInfo> stages;
  std::vector<StageNode> nodes;

  /// Collects SSA definitions/uses and Memory Read/Write effects for each
  /// stage.
  void collectEffects();

  /// Builds the directed graph edges based on SSA and Memory conflicts.
  void buildDependencyGraph();

  /// Computes the topological level of each node using DFS.
  /// Returns failure if a cycle is detected.
  LogicalResult computeStageLevels();

  /// Sorts the `stages` vector based on the computed topological levels.
  void reorderStagesLogical();

  /// Moves the operations in the IR to match the logical order of `stages`.
  void materializeScheduleToIR();
};

} // namespace dicp
} // namespace mlir

#endif // DICP_DIALECT_LINALGEXT_TRANSFORMS_STAGEDEPENDENCYANALYZER_H