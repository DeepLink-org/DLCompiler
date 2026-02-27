#ifndef DICP_LINALGEXT_STAGEUTILS_H
#define DICP_LINALGEXT_STAGEUTILS_H

#include "dicp/Dialect/LinalgExt/Analysis/StageDependencyAnalyzer.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace dicp {
namespace LinalgExt {

/// Represents a subset of operations within a stage, bounded by synchronization
/// points (e.g., HIVM Sync ops).
struct SubStage {
  unsigned index = 0;
  int stageId = -1;
  SmallVector<Operation *, 8> ops;

  bool isValid() const { return !ops.empty(); }
};

/// Represents a logical execution stage consisting of multiple SubStages.
struct Stage {
  int id = -1;
  SmallVector<SubStage, 4> subStages;
  StageType type = StageType::Vector;
  Stage() = default;
  Stage(int id, StageType type) : id(id), type(type) {
    subStages.push_back(SubStage{0, id, {}});
  }

  void addOp(Operation *op) {
    if (subStages.empty())
      subStages.push_back(SubStage{0, id, {}});
    subStages.back().ops.push_back(op);
  }
  bool isValid() const { return id != -1 && !subStages.empty(); }

  /// Returns the total number of operations across all substages.
  size_t getTotalOpCount() const {
    size_t count = 0;
    for (const auto &ss : subStages)
      count += ss.ops.size();
    return count;
  }
};

/// Check if an operation belongs to a specific stage ID.
bool isOpInStage(Operation *op, int stageId);

/// Shared utility class for analyzing blocks and partitioning them into stages.
class StagePartitioner {
public:
  /// Identifies all blocks containing HIVM sync operations.
  static SmallVector<Block *> findBlocksWithHivmSyncOps(ModuleOp module);

  /// Analyzes a block for stage dependencies and tags operations with stage
  /// attributes. \returns success if analysis succeeds, failure if a cycle is
  /// detected.
  static LogicalResult analyzeAndTagBlock(Block *block, MLIRContext *ctx,
                                          bool &anyStageFound);

  /// Extracts all unique stage IDs present in the block in deterministic order.
  static SetVector<int> getStageIdsInBlock(Block *block);

  /// Partitions the block for the given stageId into a Stage object containing
  /// SubStages. This is the primary entry point for stage-based decomposition.
  static Stage partition(Block *block, int stageId);

  /// Collects all stages present in the block.
  static SmallVector<Stage, 4> getAllStagesInBlock(Block *block);
};


class CubeVectorSplitter {
public:
  /// Segments the provided block into a sequence of Cube and Vector stages.
  ///
  /// \param block The block to analyze.
  /// \param stages Output vector to hold the resulting stages.
  /// \return failure() if validation fails (e.g., illegal nesting).
  static LogicalResult splitBlock(Block &block,
                                  llvm::SmallVectorImpl<Stage> &stages);

  /// Finds the core computation block based on the "mix_mode" attribute.
  /// Uses a maximum-density strategy for "mix" mode, and an outermost-level
  /// strategy for "aiv" / "aic" single modes.
  static Block *findTargetBlock(func::FuncOp funcOp);
};


} // namespace LinalgExt
} // namespace dicp
} // namespace mlir

#endif // DICP_LINALGEXT_STAGEUTILS_H