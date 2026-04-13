//===- StagePartition.h - DICP Stage Partition Pipeline --------*- C++ -*-===//
//
// Declares the stage partition pipeline that discovers partitionable blocks,
// groups operations into StageInfo objects, and optionally reorders stages.
//
//===----------------------------------------------------------------------===//

#ifndef DICP_DIALECT_LINALGEXT_ANALYSIS_STAGEPARTITION_H
#define DICP_DIALECT_LINALGEXT_ANALYSIS_STAGEPARTITION_H

#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::dicp {

/// Result of partitioning a single block into stages.
struct StagePartitionResult {
  Block *block = nullptr;
  SmallVector<StageInfo, 4> stages;
  DenseMap<Operation *, int> opToStageId;
  bool containsSync = false;
};

/// Configuration options for stage partitioning.
struct StagePartitionOptions {
  bool reorderEnabled = false;
  bool tileAllBlocks = false;
};

/// Unified pipeline for stage partitioning within a function.
class StagePartition {
public:
  explicit StagePartition(func::FuncOp func,
                          const StagePartitionOptions &opts = {})
      : funcOp(func), options(opts) {}

  /// Runs the stage partitioning algorithm and returns one result per block.
  FailureOr<SmallVector<StagePartitionResult, 2>> run(RewriterBase &rewriter);

private:
  LogicalResult collectTargetBlocks(SmallVectorImpl<Block *> &blocks);
  LogicalResult collectAllTargetBlocks(SmallVectorImpl<Block *> &blocks);
  void renumberStageIds(SmallVectorImpl<StagePartitionResult> &results);
  LogicalResult partitionBlock(Block *block, StagePartitionResult &result);
  LogicalResult partitionSyncMode(Block *block, StagePartitionResult &result);
  LogicalResult partitionRecoverMode(Block *block,
                                     StagePartitionResult &result);
  LogicalResult partitionCubeVectorMode(Block *block,
                                        StagePartitionResult &result);
  FailureOr<SmallVector<StageInfo, 4>>
  scheduleBlock(StagePartitionResult &result, RewriterBase &rewriter);

  func::FuncOp funcOp;
  StagePartitionOptions options;
};

} // namespace mlir::dicp

#endif // DICP_DIALECT_LINALGEXT_ANALYSIS_STAGEPARTITION_H
