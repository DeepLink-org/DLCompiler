//===- StageUtils.h - DICP Stage Analysis Utilities ------------*- C++ -*-===//
//
// Stage-specific utilities for DICP compiler infrastructure. This header
// defines the shared stage model, temporary stage attribute protocol, and
// tiling metadata helpers used across analyses and transforms.
//
//===----------------------------------------------------------------------===//

#ifndef DICP_LINALGEXT_STAGEUTILS_H
#define DICP_LINALGEXT_STAGEUTILS_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include <optional>
#include <string>

namespace mlir {
namespace dicp {

//===----------------------------------------------------------------------===//
// Stage Attribute Name Constants
//===----------------------------------------------------------------------===//

/// Prefix for all temporary DICP stage attributes.
/// These attributes are used during transformation and should be removed
/// after the transformation pipeline completes.
inline constexpr llvm::StringLiteral kDicpStageAttrPrefix = "dicp.tmp.stage.";

/// Attribute names for stage metadata, grouped in namespace for clarity.
namespace stage_attrs {

/// Dictionary attribute containing tile metadata.
inline constexpr llvm::StringLiteral kTileMetaTag = "dicp.tmp.stage.tile_meta";

/// Anchor operation tile tag (identifies which tiling unit an op belongs to).
inline constexpr llvm::StringLiteral kTileMetaAnchorTag =
    "dicp.tmp.stage.anchor_op_to_tile_tag";

/// Producer fusion tag (identifies producers to fuse with anchor).
inline constexpr llvm::StringLiteral kTileMetaFuseTag =
    "dicp.tmp.stage.producer_to_fuse_tag";

/// Tile sizes attribute key.
inline constexpr llvm::StringLiteral kTileMetaTileSizesTag = "tile_sizes";

/// Memory producer allocation attribute.
inline constexpr llvm::StringLiteral kStageProducerAllocToFuseAttr =
    "dicp.tmp.stage.alloc_producer";

/// Cross-unit user marker (indicates producer has users outside its unit).
inline constexpr llvm::StringLiteral kCrossTillUnitAttr =
    "dicp.tmp.stage.till_unit_has_cross_user";

/// Fusion completed marker.
inline constexpr llvm::StringLiteral kHadFusedAttr =
    "dicp.tmp.stage.op_had_fused";

/// Stage ID attribute name.
inline constexpr llvm::StringLiteral kNPUStageAttrName = "dicp.tmp.stage.id";

/// Original operation name (for debugging).
inline constexpr llvm::StringLiteral kOriginalOpNameAttr =
    "dicp.tmp.stage.original_op_name";

/// No-tile suppression marker.
inline constexpr llvm::StringLiteral kNoTileAttr = "dicp.tmp.stage.no_tile";

} // namespace stage_attrs

//===----------------------------------------------------------------------===//
// Stage Tag Generation Functions (inline for header-only use)
//===----------------------------------------------------------------------===//

/// Generates a unique anchor tag string for the given stage/sub/unit.
/// Format: "dicp.tmp.stage.anchor_op_to_tile_tag.stage_<id>.sub_<id>.u_<id>"
inline std::string getStageOpToTile(int64_t stage, int64_t sub, int64_t unit) {
  return llvm::formatv("{0}.stage_{1}.sub_{2}.u_{3}",
                       stage_attrs::kTileMetaAnchorTag, stage, sub, unit)
      .str();
}

/// Generates a unique producer fusion tag string.
/// Format: "dicp.tmp.stage.producer_to_fuse_tag.stage_<id>.sub_<id>.u_<id>"
inline std::string getStageProducerToFuse(int64_t stage, int64_t sub,
                                          int64_t unit) {
  return llvm::formatv("{0}.stage_{1}.sub_{2}.u_{3}",
                       stage_attrs::kTileMetaFuseTag, stage, sub, unit)
      .str();
}

//===----------------------------------------------------------------------===//
// Stage Data Structures
//===----------------------------------------------------------------------===//

/// Parsed representation of a stage/sub/unit tag.
struct StageSubUnitTag {
  int64_t stage = -1;
  int64_t sub = -1;
  int64_t unit = -1;

  bool isValid() const { return stage >= 0 && sub >= 0 && unit >= 0; }
};

/// Classification of stage execution type.
enum class StageType {
  Vector, ///< General vector or scalar operations
  Cube    ///< Matrix operations (e.g., Matmul) using Tensor Cores
};

/// A contiguous group of operations within a Stage, bounded by sync points.
struct SubStage {
  unsigned index = 0;
  int stageId = -1;
  SmallVector<Operation *, 8> ops;

  bool empty() const { return ops.empty(); }
};

/// A logical stage representing a partitioned group of operations.
/// Owns its SubStages and tracks inter-stage dependencies.
struct StageInfo {
  int id = -1;
  int level = -1;
  bool hasSync = false;
  StageType type = StageType::Vector;

  SmallVector<SubStage, 4> subStages;
  DenseSet<int> preds;
  DenseSet<int> succs;

  bool empty() const { return subStages.empty(); }

  /// Returns total operation count across all substages.
  size_t getTotalOpCount() const {
    size_t total = 0;
    for (const auto &ss : subStages)
      total += ss.ops.size();
    return total;
  }

  /// Flattens all operations in this stage.
  SmallVector<Operation *, 16> getOps() const {
    SmallVector<Operation *, 16> flatOps;
    for (const auto &ss : subStages)
      flatOps.append(ss.ops.begin(), ss.ops.end());
    return flatOps;
  }
};

//===----------------------------------------------------------------------===//
// Operation Classification Predicates
//===----------------------------------------------------------------------===//

/// Returns true if op is a SIMD-like compute operation (elementwise,
/// reduction).
bool isSIMDLikeOp(Operation *op);

/// Returns true if op is a matrix multiplication operation.
bool isMatMulOp(Operation *op);

/// Returns true if op performs a memory write (copy, store, materialize).
bool isWriteOp(Operation *op);

/// Returns true if op is a structured control-flow operation with regions.
/// Uses RegionBranchOpInterface to detect scf.for, scf.while, scf.forall,
/// scf.if. Terminators are excluded as they are structural markers, not
/// executable ops.
bool isStructuredControlFlowOp(Operation *op);

/// Returns true if op is any control-flow operation (structured or
/// unstructured). Includes BranchOpInterface, RegionBranchOpInterface, and
/// terminators.
bool isControlFlowOp(Operation *op);

/// Returns true if op is a storage root (alloc, empty, alloc_tensor).
bool isRootedTileClosureSeedOp(Operation *op);

/// Returns true if op is a partition-propagatable tensor operation.
bool isPartitionPropagatableTensorOp(Operation *op);

/// Returns true if op is convertible to a generic elementwise operation.
bool isConvertibleElementwiseOp(Operation *op);

/// Returns true if op has dynamic shapes or is a scalar operation.
bool isDynamicOrScalarOp(Operation *op);

//===----------------------------------------------------------------------===//
// Operation Utilities
//===----------------------------------------------------------------------===//

/// Returns the rank (loop nesting depth) of an operation.
int64_t getRank(Operation *op);

/// Propagates DICP-specific attributes from srcOp to dstOp.
void propagateDicpAttributes(Operation *srcOp, Operation *dstOp);

/// Returns the stage id attached to the operation, if present.
std::optional<int64_t> getStageId(Operation *op);

/// Parses and returns the producer fusion tag from an operation.
std::optional<StageSubUnitTag> getProducerFuseTag(Operation *op);

/// Parses and returns the anchor tile tag from an operation.
std::optional<StageSubUnitTag> getAnchorTileTag(Operation *op);

/// Returns the best available stage/sub/unit tag attached to the operation.
/// Producer tags take precedence over anchor tags because they describe the
/// ownership of movable producer-side bundles.
std::optional<StageSubUnitTag> getStageSubUnitTag(Operation *op);

/// Returns true when both tags belong to the same stage/substage, ignoring the
/// unit component.
bool isSameStageAndSubstage(const StageSubUnitTag &lhs,
                            const StageSubUnitTag &rhs);

/// Returns the tile metadata dictionary attached to the anchor op, if any.
DictionaryAttr getTileMeta(Operation *op);

/// Parses the staged tile sizes from the anchor op tile metadata.
FailureOr<SmallVector<int64_t, 4>> getTileSizes(Operation *op);

/// Attaches the canonical tile metadata dictionary to the anchor op.
void setTileMeta(Operation *op, StringRef anchorTag, StringRef producerFuseTag,
                 ArrayRef<int64_t> tileSizes);

/// Removes temporary tiling ownership/tagging attributes while preserving
/// stage membership and any permanent suppression markers.
void clearTilingUnitAttrs(Operation *op);

/// Clears cross-unit user attributes from an operation.
void clearCrossUnitUserAttrs(Operation *op);

/// Removes all temporary stage attributes from the module.
void removeTmpStageAttributes(ModuleOp moduleOp);

/// Resolves the underlying memref.alloc from a producer operation.
/// Handles direct AllocOp and bufferization.to_tensor wrappers.
memref::AllocOp resolveUnderlyingAlloc(Operation *producer);

/// Returns the shaped value whose dimensions drive tiling for anchorOp.
Value getTilingReferenceValue(Operation *anchorOp);

/// Returns true when the stage satisfies all tiling eligibility rules.
bool isStageEligibleForTiling(const StageInfo &stage);

//===----------------------------------------------------------------------===//
// Tiling Unit Definition
//===----------------------------------------------------------------------===//

/// Metadata for a single tiling group with lazy-loaded attributes.
struct TilingUnit {
  // Phase 1: Discovery (Skeleton)
  Operation *anchorOp = nullptr;
  SmallVector<Operation *, 16> producerOps;
  SmallVector<Operation *, 16> ownedProducerOps;
  SmallVector<Operation *, 16> borrowedProducerOps;
  int64_t rank = 0;

  DenseMap<Operation *, bool> isMemProducer;
  DenseMap<Operation *, bool> hasCrossUsers;
  bool anchorHasCrossUsers = false;
  bool anchorNeedsProducerTag = false;
  unsigned priority = 0;
  size_t irOrder = 0;

  // Phase 2: Analysis (Computed values)
  SmallVector<int64_t, 4> candidateDims;
  std::optional<int64_t> tilingDimIndex;
  SmallVector<int64_t, 4> tileSizes;

  // Generated tag strings
  std::string anchorTag;
  std::string producerComputeTag;
  std::string producerAllocTag;
  std::string crossUserTag;
};

} // namespace dicp
} // namespace mlir

#endif // DICP_LINALGEXT_STAGEUTILS_H
