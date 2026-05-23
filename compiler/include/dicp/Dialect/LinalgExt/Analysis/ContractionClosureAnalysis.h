//===- ContractionClosureAnalysis.h - Contraction Closure Analysis -*- C++
//-*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#ifndef DICP_DIALECT_LINALGEXT_ANALYSIS_CONTRACTIONCLOSURE_H
#define DICP_DIALECT_LINALGEXT_ANALYSIS_CONTRACTIONCLOSURE_H

#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>
#include <string>

namespace mlir {
namespace dicp {

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

struct SubStage;
class DimAnalyzer;
// TilingUnit is defined in StageUtils.h

//===----------------------------------------------------------------------===//
// Data Structures
//===----------------------------------------------------------------------===//

/// Represents the semantic role of an operation in the feeder chain.
enum class FeederRole {
  Root,      // memref.alloc, bufferization.alloc_tensor, tensor.empty
  Bridge,    // bufferization.to_tensor, materialize_in_destination
  Transform, // linalg.transpose, memref.cast, expand/collapse_shape
  Copy,      // memref.copy, linalg.copy
  Other,     // Unknown or irrelevant
  Terminator // Stop tracing (e.g., scf.for, func.call)
};

/// Information about an operation in the feeder chain.
struct ChainOpInfo {
  Operation *op = nullptr;
  FeederRole role = FeederRole::Other;
  bool affectsUntiledDims = true;
};

/// Represents a storage root and its shared status.
struct RootEquivalenceClass {
  Operation *root = nullptr;
  SmallVector<struct FeederClosure *> closures;
  bool isShared() const { return closures.size() > 1; }
};

/// Represents the tile semantics of a contraction operation.
struct ContractionTileSemantics {
  Operation *contractionOp = nullptr;
  const TilingUnit *tileUnit = nullptr;

  // Derived from the owning unit selection and the contraction semantics.
  SmallVector<int64_t> tiledLoopDims;
  SmallVector<int64_t> parallelDims;
  SmallVector<int64_t> reductionDims;
  SmallVector<int64_t> tiledParallelDims;
  SmallVector<int64_t> tiledReductionDims;

  // Operand -> bound semantic dimensions
  DenseMap<Value, SmallVector<int64_t>> operandToDims;
  DenseMap<Value, SmallVector<int64_t>> operandRelevantPositions;

  bool isTiledLoopDim(int64_t dim) const {
    return llvm::is_contained(tiledLoopDims, dim);
  }
  bool hasTiledDims() const {
    return !tiledParallelDims.empty() || !tiledReductionDims.empty();
  }
};

/// Represents an exit point from a closure.
struct ClosureExit {
  Value exitValue;
  Operation *exitOp = nullptr;
  bool isExternal = false;
};

/// A feeder closure for a specific operand of contraction.
struct FeederClosure {
  Operation *contractionOp = nullptr;
  Value operand;
  SmallVector<int64_t> relevantOperandDims;
  SmallVector<ChainOpInfo> chainOps;
  SmallVector<Operation *> storageRoots;
  SmallVector<ClosureExit> exits;
  bool isExclusive = false;
  bool isSupported = true;
  std::string nonExclusiveReason;
};

/// Suppression decision for a tiling unit.
struct SuppressionDecision {
  enum Action { NoAction, MarkNoTile } action = NoAction;

  Operation *anchorOp = nullptr;
  std::string reason;
  SmallVector<Operation *> suppressedOps;
  SmallVector<Operation *> suppressedRoots;
  SmallVector<FeederClosure *> evidence;
};

//===----------------------------------------------------------------------===//
// Main Analysis Class
//===----------------------------------------------------------------------===//

class ContractionClosureAnalysis {
public:
  /// This analysis must run after the entire current substage has
  /// materialized all provisional tiling attrs needed by the analysis.
  explicit ContractionClosureAnalysis(const SubStage &subStageInfo,
                                      ArrayRef<TilingUnit> tilingUnits);
  ~ContractionClosureAnalysis();

  LogicalResult analyze();
  ArrayRef<SuppressionDecision> getDecisions() const;

private:
  // Phase 1: Discovery
  void discoverContractions();
  bool isContractionAnchor(Operation *op) const;

  // Phase 2: Tile Semantics
  FailureOr<std::optional<ContractionTileSemantics>>
  parseTileSemantics(Operation *op);
  void computeOperandDimBindings(ContractionTileSemantics &sem);
  LogicalResult initializeProvenanceAnalysis();
  FailureOr<SmallVector<int64_t, 4>>
  resolveContractionTiledLoopDims(linalg::LinalgOp contraction,
                                  const TilingUnit &tileUnit) const;

  // Phase 3: Dimension-Aware Backward Slice
  FeederClosure traceFeederClosure(Value operand,
                                   ArrayRef<int64_t> relevantPositions,
                                   ArrayRef<int64_t> relevantOperandDims,
                                   Operation *contractionOp) const;
  FeederRole classifyFeederRole(Operation *op) const;

  // Phase 4: Root Equivalence & Exclusivity
  void buildRootEquivalenceClasses();
  bool checkExclusivity(FeederClosure &closure);
  bool analyzeClosureExits(FeederClosure &closure);
  bool isInScope(Operation *op) const;

  // Phase 5: Decision Generation
  void generateDecisions();
  bool isUnitAnchorInClosure(const TilingUnit &unit,
                             const FeederClosure &closure) const;

  // Members
  const SubStage &subStageInfo;
  ArrayRef<TilingUnit> tilingUnits;

  DenseSet<Operation *> scopeOpSet;
  SmallVector<Operation *, 16> provenanceOps;
  std::unique_ptr<DimAnalyzer> provenanceAnalyzer;
  SmallVector<Operation *> contractions;
  SmallVector<ContractionTileSemantics, 4> semantics;
  SmallVector<FeederClosure, 4> closures;
  DenseMap<Operation *, RootEquivalenceClass> rootClasses;
  DenseMap<Operation *, size_t> decisionIndexByAnchor;
  SmallVector<SuppressionDecision, 4> decisions;
};

} // namespace dicp
} // namespace mlir

#endif // DICP_DIALECT_LINALGEXT_ANALYSIS_CONTRACTIONCLOSURE_H
