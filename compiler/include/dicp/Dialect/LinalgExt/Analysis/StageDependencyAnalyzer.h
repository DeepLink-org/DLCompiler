#ifndef DICP_DIALECT_LINALGEXT_TRANSFORMS_STAGEDEPENDENCYANALYZER_H
#define DICP_DIALECT_LINALGEXT_TRANSFORMS_STAGEDEPENDENCYANALYZER_H

#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace dicp {

/// StageDependencyAnalyzer partitions a Block into "Stages" (delimited by sync
/// ops) and reorders them based on a topological sort of their data and memory
/// dependencies.
class StageDependencyAnalyzer {
public:
  StageDependencyAnalyzer(Block *block, AliasAnalysis &aliasAnalysis)
      : block(block), aliasAnalysis(aliasAnalysis) {}

  /// High-level entry point: Collects stages from the block, analyzes
  /// dependencies, and physically reorders operations.
  FailureOr<SmallVector<StageInfo, 4>> runAndReorder(RewriterBase &rewriter);

  /// Scans the block to partition operations into initial StageInfo objects
  /// based on hivm::SyncBlockWaitOp boundaries.
  static FailureOr<SmallVector<StageInfo, 4>> collectStages(Block *block);

  /// Performs dependency analysis and physical IR reordering for a given
  /// set of stages. This allows re-analysis of externally provided stages.
  FailureOr<SmallVector<StageInfo, 4>>
  analyzeAndReorderStages(SmallVector<StageInfo, 4> &&initialStages);

private:
  /// Internal helper to represent the analysis state of a stage.
  struct StageNode {
    int level = 0;
    StageInfo *stageInfo;

    // SSA Value dependencies
    llvm::SetVector<Value> producedValues;
    llvm::SetVector<Value> consumedValues;

    // Memory dependencies
    llvm::SetVector<Value> readBuffers;
    llvm::SetVector<Value> writeBuffers;
  };

  /// Populates SSA definitions/uses and Memory effects for each stage.
  void collectEffects(SmallVector<StageInfo, 4> &stages);

  /// Identifies edges between stages based on SSA and Memory Alias Analysis.
  void buildDependencyGraph();

  /// Computes topological levels for nodes. Returns failure if a cycle exists.
  LogicalResult computeStageLevels();

  Block *block;
  AliasAnalysis &aliasAnalysis;
  std::vector<StageNode> nodes;
};

} // namespace dicp
} // namespace mlir

#endif // DICP_DIALECT_LINALGEXT_TRANSFORMS_STAGEDEPENDENCYANALYZER_H