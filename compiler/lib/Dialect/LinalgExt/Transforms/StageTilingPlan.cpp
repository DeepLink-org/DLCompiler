#include "dicp/Dialect/LinalgExt/Analysis/BufferEscapeAnalysis.h"
#include "dicp/Dialect/LinalgExt/Analysis/ContractionClosureAnalysis.h"
#include "dicp/Dialect/LinalgExt/Analysis/DimAnalyzer.h"
#include "dicp/Dialect/LinalgExt/Analysis/StagePartition.h"
#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"
#include "dicp/Utils/Utils.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "stage-tiling-plan"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace dicp;
using namespace LinalgExt;

namespace mlir {
namespace dicp {
namespace LinalgExt {
#define GEN_PASS_DEF_STAGETILINGPLAN
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace LinalgExt
} // namespace dicp
} // namespace mlir

namespace {

//===----------------------------------------------------------------------===//
// Shape & Tile Calculus
//===----------------------------------------------------------------------===//

/// Calculate the tile size for a given dimension and trip count.
/// Returns std::nullopt if the size cannot be determined or is not perfectly
/// divisible.
static std::optional<int64_t> calculateTileSizeForUnit(Operation *anchorOp,
                                                       int64_t dimIdx,
                                                       int64_t tripCount) {
  if (tripCount <= 0) {
    LDBG("  [TileCalc] Reject dim " << dimIdx << ": invalid trip count "
                                    << tripCount);
    return std::nullopt;
  }

  auto getDimSize = [&](Value v) -> std::optional<int64_t> {
    auto type = dyn_cast_or_null<ShapedType>(v.getType());
    if (!type || !type.hasRank() || dimIdx >= type.getRank() ||
        type.isDynamicDim(dimIdx))
      return std::nullopt;
    return type.getDimSize(dimIdx);
  };

  std::optional<int64_t> totalSize =
      getDimSize(getTilingReferenceValue(anchorOp));
  if (!totalSize || *totalSize <= 0) {
    LDBG("  [TileCalc] Reject dim " << dimIdx
                                    << ": static size is unavailable");
    return std::nullopt;
  }
  if (*totalSize % tripCount != 0) {
    LDBG("  [TileCalc] Reject dim " << dimIdx << ": size " << *totalSize
                                    << " is not divisible by " << tripCount);
    return std::nullopt;
  }

  LDBG("  [TileCalc] Accept dim " << dimIdx << ": size " << *totalSize
                                  << " -> tile size "
                                  << (*totalSize / tripCount));
  return *totalSize / tripCount;
}

//===----------------------------------------------------------------------===//
// Forward Escape Analysis
//===----------------------------------------------------------------------===//

/// Returns true if the write op targets a storage root that has been
/// suppressed by contraction no-tile hints (kNoTileAttr on root).
static bool isSuppressedWriteAnchor(Operation *op) {
  if (!isWriteOp(op))
    return false;
  return llvm::any_of(op->getOperands(), [](Value v) {
    Operation *def = v.getDefiningOp();
    return def && def->hasAttr(stage_attrs::kNoTileAttr) &&
           isRootedTileClosureSeedOp(def);
  });
}

/// Constructs a BackwardSliceOptions configured for anchor-based discovery.
/// Shared by Phase 0 (seed hints) and Phase 1 (skeleton extraction) to
/// ensure consistent slicing behavior.
static BackwardSliceOptions makeAnchorSliceOptions(Block *block) {
  BackwardSliceOptions opt;
  opt.omitBlockArguments = true;
  opt.filter = [block](Operation *op) {
    if (op->getBlock() != block)
      return false;
    if (isa<memref::ReinterpretCastOp, memref::CastOp>(op) ||
        isa<BranchOpInterface, RegionBranchOpInterface>(op))
      return false;
    return !isDynamicOrScalarOp(op);
  };
  return opt;
}

/// Selects the first tiling dimension from sorted candidates that yields a
/// perfectly divisible tile size. Shared by Phase 0 and Phase 2.
static std::optional<int64_t> selectFirstValidDim(Operation *anchor,
                                                  ArrayRef<int64_t> candidates,
                                                  int64_t tripCount) {
  for (int64_t dim : candidates) {
    if (calculateTileSizeForUnit(anchor, dim, tripCount))
      return dim;
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Data Structures
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Local Type Aliases (using dicp::TilingUnit from StageUtils.h)
//===----------------------------------------------------------------------===//

using TilingUnit = dicp::TilingUnit;

/// Represents a sub-stage containing multiple tiling units.
struct TiledSubStage {
  SubStage &base;
  const StageInfo &stage;
  SmallVector<TilingUnit, 4> units;
  explicit TiledSubStage(SubStage &s, const StageInfo &st)
      : base(s), stage(st) {}
};

/// Stable, stage-scoped cache for legality facts reused across many candidate
/// units in the same substage. Only unit-invariant facts belong here.
class SubStageAnalysisCache {
public:
  SubStageAnalysisCache(const TiledSubStage &ts, AliasAnalysis *aliasAnalysis)
      : aliasAnalysis(aliasAnalysis) {
    for (Operation *op : ts.stage.getOps())
      stageOpSet.insert(op);
  }

  bool hasStaticShape(Operation *op) const {
    auto [it, inserted] = staticShapeCache.try_emplace(op, false);
    if (!inserted)
      return it->second;

    bool hasStatic = succeeded(verifyStaticShape(op));
    it->second = hasStatic;
    LDBG("    [Cache:StaticShape] " << (hasStatic ? "accept " : "reject ")
                                    << op->getName());
    return hasStatic;
  }

  bool isMemProducer(Operation *op) const {
    auto [it, inserted] = memProducerCache.try_emplace(op, false);
    if (!inserted)
      return it->second;

    bool isMem =
        TypeSwitch<Operation *, bool>(op)
            .Case<tensor::EmptyOp, memref::AllocOp>([](auto) { return true; })
            .Case<bufferization::ToTensorOp>([](auto toTensor) {
              return isa_and_nonnull<tensor::EmptyOp, memref::AllocOp>(
                  toTensor.getOperand().getDefiningOp());
            })
            .Default(false);
    it->second = isMem;
    LDBG("    [Cache:MemProducer] " << (isMem ? "accept " : "reject ")
                                    << op->getName());
    return isMem;
  }

  DenseMap<Operation *, BufferEscapeSummary>
  collectBufferEscapeSummaries(const TilingUnit &unit) const {
    DenseMap<Operation *, BufferEscapeSummary> summaries;
    for (Operation *producer : unit.producerOps) {
      auto allocOp = resolveUnderlyingAlloc(producer);
      if (!allocOp || summaries.contains(allocOp.getOperation()))
        continue;

      auto [it, inserted] =
          stageEscapeSummaryCache.try_emplace(allocOp.getOperation());
      if (inserted) {
        auto isInStage = [&](Operation *op) { return stageOpSet.contains(op); };
        it->second = analyzeBufferEscape(allocOp, isInStage, aliasAnalysis);
        LDBG("    [Cache:BufferEscape] Materialized summary for alloc "
             << *allocOp << " with " << it->second.externalUsers.size()
             << " external user(s)");
      } else {
        LDBG("    [Cache:BufferEscape] Reusing cached summary for alloc "
             << *allocOp);
      }

      summaries.try_emplace(allocOp.getOperation(), it->second);
    }
    return summaries;
  }

private:
  AliasAnalysis *aliasAnalysis = nullptr;
  DenseSet<Operation *> stageOpSet;

  mutable DenseMap<Operation *, bool> staticShapeCache;
  mutable DenseMap<Operation *, bool> memProducerCache;
  mutable DenseMap<Operation *, BufferEscapeSummary> stageEscapeSummaryCache;
};

enum class AnchorPrecheckResult {
  Allow,
  SkipSuppressed,
};

//===----------------------------------------------------------------------===//
// Unit Eligibility Rules (Chain-of-Rules Pattern)
//===----------------------------------------------------------------------===//
//
// Each unit rule is a pure predicate over UnitRuleContext.
// Returns `true` when the unit is eligible, `false` to skip it (soft reject).
//
// To add a new rule:
//   1. Define a static function: `bool fn(const UnitRuleContext &)`.
//   2. Append it to `kUnitRules`.
// No other code needs to change.
//===----------------------------------------------------------------------===//

/// Context passed to every unit eligibility rule.
struct UnitRuleContext {
  const TilingUnit &unit;
  const StageInfo &stage;
  const DenseMap<Operation *, BufferEscapeSummary> &bufferEscapeSummaries;
  const SubStageAnalysisCache *analysisCache = nullptr;
  bool tileAllBlocks = false;
  std::optional<int64_t> tilingDimIndex;
  DimAnalyzer *recurrenceAnalyzer = nullptr;
};

/// Signature for a unit eligibility rule.
using UnitRule = bool (*)(const UnitRuleContext &);

enum class EscapePolicyResult {
  Allow,
  SkipUnit,
  FailPass,
};

static const BufferEscapeSummary *
getBufferEscapeSummary(const UnitRuleContext &ctx, memref::AllocOp allocOp) {
  auto it = ctx.bufferEscapeSummaries.find(allocOp.getOperation());
  if (it == ctx.bufferEscapeSummaries.end())
    return nullptr;
  return &it->second;
}

/// All ops in the unit must have fully static shapes.
/// Dynamic shapes cannot be perfectly tiled with static tile sizes.
static bool hasStaticShapes(const UnitRuleContext &ctx) {
  auto hasStaticShape = [&](Operation *op) {
    return ctx.analysisCache ? ctx.analysisCache->hasStaticShape(op)
                             : succeeded(verifyStaticShape(op));
  };

  if (!hasStaticShape(ctx.unit.anchorOp)) {
    LDBG("    [UnitRule:StaticShape] Anchor has dynamic shape: "
         << *ctx.unit.anchorOp);
    return false;
  }
  for (Operation *p : ctx.unit.producerOps) {
    if (!hasStaticShape(p)) {
      LDBG("    [UnitRule:StaticShape] Producer has dynamic shape: " << *p);
      return false;
    }
  }
  return true;
}

/// Evaluates how alloc-backed producers may interact with operations outside
/// the current stage.
///
/// Policy:
///   - `tileAllBlocks=true`: align with SliceParallelAnalysis. Only real
///     storage escapes (calls or writes to the buffer alias subtree) are
///     fatal. Pure tensor propagation is deferred to post-dimension recurrence
///     checks.
///   - `tileAllBlocks=false`: preserve the original strict single-block mode.
///     Any use outside the current stage causes the unit to be skipped, but
///     the pass continues without failure.
static EscapePolicyResult
evaluateBufferEscapePolicy(const UnitRuleContext &ctx) {
  auto moduleOp = ctx.unit.anchorOp->getParentOfType<ModuleOp>();
  if (!moduleOp || !isAscendBackend(moduleOp))
    return EscapePolicyResult::Allow;

  for (Operation *p : ctx.unit.producerOps) {
    auto allocOp = resolveUnderlyingAlloc(p);
    if (!allocOp)
      continue;

    const BufferEscapeSummary *summary = getBufferEscapeSummary(ctx, allocOp);
    if (!summary) {
      LDBG("    [UnitRule:ConfinedAlloc] Missing escape summary for alloc: "
           << *allocOp);
      return EscapePolicyResult::FailPass;
    }

    if (summary->hasCallEscape()) {
      LDBG("    [UnitRule:ConfinedAlloc] Cube stage allowing call escape: "
           << *allocOp);
      for (Operation *op : summary->callEscapes)
        LDBG("      [CallEscape] " << *op);
      if (ctx.stage.type != StageType::Cube) {
        return ctx.tileAllBlocks ? EscapePolicyResult::FailPass
                                 : EscapePolicyResult::SkipUnit;
      }
    }

    if (!summary->isReadOnlyOutsideScope()) {
      LDBG("    [UnitRule:ConfinedAlloc] Cube stage allowing modifying escape: "
           << *allocOp);
      for (Operation *op : summary->modifyingEscapes)
        LDBG("      [ModifyingEscape] " << *op);
      if (ctx.stage.type != StageType::Cube) {
        return ctx.tileAllBlocks ? EscapePolicyResult::FailPass
                                 : EscapePolicyResult::SkipUnit;
      }
    }

    if (!ctx.tileAllBlocks && !summary->externalUsers.empty()) {
      LDBG("    [UnitRule:ConfinedAlloc] Strict single-block mode forbids any "
           "use outside the current stage. Skipping unit for alloc: "
           << *allocOp);
      for (Operation *externalOp : summary->externalUsers)
        LDBG("      [StageEscape] " << *externalOp);
      return EscapePolicyResult::SkipUnit;
    }
  }
  return EscapePolicyResult::Allow;
}

static bool isTiledRootParallelInYieldedValue(Value yieldedValue,
                                              int64_t selectedRoot,
                                              DimAnalyzer &analyzer) {
  auto shapedType = dyn_cast<ShapedType>(yieldedValue.getType());
  if (!shapedType || !shapedType.hasRank()) {
    LDBG("      [Recurrence] Yielded value has no ranked shaped dims: "
         << yieldedValue);
    return true;
  }

  bool matchedSelectedRoot = false;
  for (int64_t dim = 0, e = shapedType.getRank(); dim < e; ++dim) {
    auto root = analyzer.getDimRoot(yieldedValue, dim);
    if (!root || *root != selectedRoot)
      continue;

    matchedSelectedRoot = true;
    DimKind kind = analyzer.getDimKind(yieldedValue, dim);
    LDBG("      [Recurrence] Yielded dim " << dim << " has kind "
                                           << toString(kind));
    if (kind != DimKind::Parallel && kind != DimKind::Broadcast)
      return false;
  }

  if (!matchedSelectedRoot)
    LDBG("      [Recurrence] Yielded value no longer carries the tiled root");
  return true;
}

/// After the tiling dimension is selected, every yielded value that depends on
/// the alloc closure must either keep the tiled root on parallel dimensions or
/// consume that root entirely inside the current iteration.
static bool hasTileIndependentRecurrences(const UnitRuleContext &ctx) {
  auto moduleOp = ctx.unit.anchorOp->getParentOfType<ModuleOp>();
  if (!moduleOp || !isAscendBackend(moduleOp))
    return true;
  if (!ctx.tilingDimIndex || !ctx.recurrenceAnalyzer) {
    LDBG("    [UnitRule:Recurrence] Missing recurrence analysis context.");
    return false;
  }

  Value tilingRef = getTilingReferenceValue(ctx.unit.anchorOp);
  if (!tilingRef) {
    LDBG("    [UnitRule:Recurrence] Could not resolve tiling reference value "
         "for anchor: "
         << *ctx.unit.anchorOp);
    return false;
  }

  auto selectedRoot =
      ctx.recurrenceAnalyzer->getDimRoot(tilingRef, *ctx.tilingDimIndex);
  if (!selectedRoot) {
    LDBG("    [UnitRule:Recurrence] Failed to resolve selected tiled root for "
         "anchor dim "
         << *ctx.tilingDimIndex << ": " << *ctx.unit.anchorOp);
    return false;
  }

  for (Operation *p : ctx.unit.producerOps) {
    auto allocOp = resolveUnderlyingAlloc(p);
    if (!allocOp)
      continue;

    const BufferEscapeSummary *summary = getBufferEscapeSummary(ctx, allocOp);
    if (!summary)
      return false;

    for (Operation *externalOp : summary->externalUsers) {
      auto yieldOp = dyn_cast<scf::YieldOp>(externalOp);
      if (!yieldOp)
        continue;
      if (yieldOp->getBlock() != allocOp->getBlock()) {
        LDBG("    [UnitRule:Recurrence] Rejecting nested/outer yield escape "
             "for alloc: "
             << *allocOp << " -> " << *yieldOp);
        return false;
      }

      for (Value operand : yieldOp.getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (!defOp || !summary->forwardSlice.contains(defOp))
          continue;

        if (!isTiledRootParallelInYieldedValue(operand, *selectedRoot,
                                               *ctx.recurrenceAnalyzer)) {
          LDBG("    [UnitRule:Recurrence] Rejecting yielded value with "
               "non-parallel tiled root: "
               << operand);
          LDBG("      [Producer] " << *defOp);
          return false;
        }
      }
    }
  }
  return true;
}

static constexpr UnitRule kPreDimUnitRules[] = {
    hasStaticShapes,
};

static constexpr UnitRule kPostDimUnitRules[] = {
    hasTileIndependentRecurrences,
};

static bool runUnitRules(ArrayRef<UnitRule> rules, const UnitRuleContext &ctx) {
  return llvm::all_of(rules, [&](UnitRule rule) { return rule(ctx); });
}

//===----------------------------------------------------------------------===//
// TilingProcessor
//===----------------------------------------------------------------------===//

/// A Two-Phase processor to identify tiling anchors and assign unique tags.
/// Phase 1: Discovery (Read-only, topology skeleton extraction).
/// Phase 2: Analyze & Tag (Per-unit dimension analysis, tile size calc, IR
/// mutation).
class TilingProcessor {
public:
  explicit TilingProcessor(int64_t tripCount,
                           AliasAnalysis *aliasAnalysis = nullptr,
                           bool tileAllBlocks = false,
                           bool enableFallback = true)
      : tripCount(tripCount), aliasAnalysis(aliasAnalysis),
        tileAllBlocks(tileAllBlocks), enableFallback(enableFallback) {}

  /// Main entry point: Executes the Two-Phase decoupling pipeline.
  LogicalResult process(TiledSubStage &ts) const {
    LDBG("[TilingProcessor] Processing SubStage "
         << ts.base.index << " in Stage " << ts.base.stageId);
    SubStageAnalysisCache analysisCache(ts, aliasAnalysis);

    DenseSet<Operation *> suppressedAnchors;
    for (size_t attempt = 0, maxAttempts = ts.base.ops.size();
         attempt < maxAttempts; ++attempt) {
      SmallVector<TilingUnit, 4> skeletons =
          discoverUnits(ts, suppressedAnchors, analysisCache);
      if (skeletons.empty()) {
        if (suppressedAnchors.empty()) {
          LDBG("  [Info] No tiling skeletons found for this substage. "
               << "Treating it as a legal no-op.");
        } else {
          LDBG("  [Info] No remaining tiling skeletons found after "
               << suppressedAnchors.size() << " suppression step(s).");
        }
        return success();
      }

      SmallVector<TilingUnit, 4> plannedUnits;
      if (failed(analyzeUnits(ts, skeletons, plannedUnits, analysisCache))) {
        LDBG("  [Error] Unit planning failed.");
        ts.base.ops.front()->emitError("Phase 2: Analysis failed for SubStage ")
            << ts.base.index << " in Stage " << ts.base.stageId;
        return failure();
      }
      if (plannedUnits.empty()) {
        LDBG("  [Info] No legal tiling plan survived analysis for this "
             "substage.");
        return success();
      }

      SmallVector<TilingUnit, 4> activeUnits;
      selectNonOverlappingUnits(plannedUnits, activeUnits);
      if (activeUnits.empty()) {
        LDBG("  [Info] All planned units were deferred due to overlap.");
        return success();
      }

      LDBG("  [Info] Materializing a full provisional substage snapshot "
           "before contraction-aware suppression.");
      materializeProvisionalUnitAttrs(activeUnits);

      ContractionClosureAnalysis analysis(ts.base, activeUnits);
      if (failed(analysis.analyze())) {
        clearProvisionalUnitAttrs(activeUnits);
        ts.base.ops.front()->emitError(
            "ContractionClosureAnalysis failed on the provisional substage "
            "snapshot");
        return failure();
      }

      if (analysis.getDecisions().empty()) {
        LDBG("  [Info] No contraction-driven suppression was produced. "
             "Committing the current substage tiling plan.");
        llvm::append_range(ts.units, activeUnits);
        return success();
      }

      SmallVector<Operation *, 4> newlySuppressedAnchors;
      collectSuppressedAnchors(analysis, newlySuppressedAnchors);
      LDBG("  [Info] Contraction analysis suppressed "
           << newlySuppressedAnchors.size()
           << " anchor(s); clearing provisional placement attrs and "
              "restarting substage planning.");
      applySuppressionMarkers(analysis);
      clearProvisionalUnitAttrs(activeUnits);
      suppressedAnchors.insert(newlySuppressedAnchors.begin(),
                               newlySuppressedAnchors.end());
    }

    ts.base.ops.front()->emitError(
        "Exceeded suppression rediscovery limit while tagging tiling units");
    return failure();
  }

private:
  int64_t tripCount;
  AliasAnalysis *aliasAnalysis;
  bool tileAllBlocks;
  bool enableFallback;

  /// Lower numeric value means higher discovery/selection priority.
  ///
  /// Keep the legacy ordering intact:
  ///   1. normalized write anchors,
  ///   2. yielded compute anchors,
  ///   3. fallback compute anchors.
  ///
  /// This preserves the original greedy ownership model: the first accepted
  /// unit claims the shared closure, and later overlapping units are deferred.
  enum class Priority { Normalized = 1, Yield = 2, Fallback = 3 };
  struct Candidate {
    Operation *op;
    Priority prio;
    size_t irIdx;
  };

  static StringRef getPriorityName(unsigned priority) {
    switch (static_cast<Priority>(priority)) {
    case Priority::Normalized:
      return "Normalized";
    case Priority::Yield:
      return "Yield";
    case Priority::Fallback:
      return "Fallback";
    }
    llvm_unreachable("unexpected unit priority");
  }

  static bool hasHigherPriority(Priority lhsPrio, size_t lhsIrIdx,
                                Priority rhsPrio, size_t rhsIrIdx) {
    return std::tie(lhsPrio, lhsIrIdx) < std::tie(rhsPrio, rhsIrIdx);
  }

  static bool hasHigherPriority(const Candidate &lhs, const Candidate &rhs) {
    return hasHigherPriority(lhs.prio, lhs.irIdx, rhs.prio, rhs.irIdx);
  }

  static void recordBestCandidate(DenseMap<Operation *, Candidate> &best,
                                  Operation *op, Priority priority,
                                  size_t irIdx) {
    Candidate candidate{op, priority, irIdx};
    auto [it, inserted] = best.try_emplace(op, candidate);
    if (inserted || hasHigherPriority(candidate, it->second))
      it->second = candidate;
  }

  static void markCrossUnitOwner(TilingUnit &ownerUnit, Operation *sharedOp,
                                 Operation *borrowingAnchor) {
    if (ownerUnit.anchorOp == sharedOp) {
      ownerUnit.anchorHasCrossUsers = true;
      ownerUnit.anchorNeedsProducerTag = true;
      LDBG("    [Select] Shared op is owner anchor; preserving anchor "
           "ownership and adding cross-unit producer semantics. owner="
           << *ownerUnit.anchorOp << ", borrower=" << *borrowingAnchor);
      return;
    }

    ownerUnit.hasCrossUsers[sharedOp] = true;
    LDBG("    [Select] Shared producer stays with first owner. producer="
         << *sharedOp << ", owner=" << *ownerUnit.anchorOp
         << ", borrower=" << *borrowingAnchor);
  }

  static bool
  tryClaimUnitWithSharedProducers(TilingUnit &unit,
                                  SmallVectorImpl<TilingUnit> &activeUnits,
                                  DenseMap<Operation *, unsigned> &opOwners) {
    auto ownerIt = opOwners.find(unit.anchorOp);
    if (ownerIt != opOwners.end()) {
      LDBG("    [Select] Deferring unit because anchor was already claimed by "
           "unit "
           << ownerIt->second << ": " << *unit.anchorOp);
      return false;
    }

    unit.ownedProducerOps.clear();
    unit.borrowedProducerOps.clear();

    const unsigned unitIndex = activeUnits.size();
    opOwners[unit.anchorOp] = unitIndex;
    LDBG("    [Select] Claimed anchor for unit " << unitIndex << ": "
                                                 << *unit.anchorOp);

    for (Operation *producer : unit.producerOps) {
      auto [it, inserted] = opOwners.try_emplace(producer, unitIndex);
      if (inserted) {
        unit.ownedProducerOps.push_back(producer);
        LDBG("      [Select] Claimed producer for unit " << unitIndex << ": "
                                                         << *producer);
        continue;
      }

      unit.borrowedProducerOps.push_back(producer);
      TilingUnit &ownerUnit = activeUnits[it->second];
      markCrossUnitOwner(ownerUnit, producer, unit.anchorOp);
      LDBG("      [Select] Borrowed shared producer from unit "
           << it->second << ": " << *producer);
    }

    return true;
  }

  //===--------------------------------------------------------------------===//
  // Phase 1: Topology Discovery
  //===--------------------------------------------------------------------===//

  AnchorPrecheckResult
  runAnchorPrecheck(Operation *anchor,
                    const DenseSet<Operation *> &forbiddenAnchors) const {
    if (forbiddenAnchors.contains(anchor) ||
        anchor->hasAttr(stage_attrs::kNoTileAttr)) {
      LDBG("  [AnchorPrecheck] Skip suppressed anchor: " << *anchor);
      return AnchorPrecheckResult::SkipSuppressed;
    }

    return AnchorPrecheckResult::Allow;
  }

  TilingUnit
  buildUnitSkeleton(const TiledSubStage &ts, Operation *anchor,
                    const Candidate &candidate,
                    const SubStageAnalysisCache &analysisCache) const {
    SetVector<Operation *> slice;
    auto opt = makeAnchorSliceOptions(anchor->getBlock());
    (void)getBackwardSlice(anchor, &slice, opt);
    slice.remove(anchor);

    SmallVector<Operation *, 4> producers;
    for (Operation *producer : slice) {
      if (!llvm::is_contained(ts.base.ops, producer))
        continue;
      if (producer->hasAttr(stage_attrs::kNoTileAttr)) {
        LDBG("    [UnitBuilder] Excluding no_tile producer from skeleton: "
             << *producer);
        continue;
      }
      producers.push_back(producer);
    }

    TilingUnit unit;
    unit.anchorOp = anchor;
    unit.producerOps = std::move(producers);
    unit.rank = getRank(anchor);
    unit.priority = static_cast<unsigned>(candidate.prio);
    unit.irOrder = candidate.irIdx;

    SetVector<Operation *> currentUnitOps(unit.producerOps.begin(),
                                          unit.producerOps.end());
    currentUnitOps.insert(anchor);

    for (Operation *producer : unit.producerOps) {
      bool hasCross = llvm::any_of(producer->getUsers(), [&](Operation *user) {
        return !currentUnitOps.contains(user);
      });
      unit.hasCrossUsers[producer] = hasCross;
      unit.isMemProducer[producer] = analysisCache.isMemProducer(producer);

      if (hasCross) {
        LDBG("    [UnitBuilder] Producer has cross-unit users: " << *producer);
      }
    }

    return unit;
  }

  /// Discovers all potential TilingUnits (Skeletons) based on Def-Use chains.
  SmallVector<TilingUnit, 4>
  discoverUnits(const TiledSubStage &ts,
                const DenseSet<Operation *> &forbiddenAnchors,
                const SubStageAnalysisCache &analysisCache) const {
    SmallVector<TilingUnit, 4> units;
    DenseSet<Operation *> seenAnchors;
    auto candidates = collectCandidates(ts.base);

    auto tryExtractSkeleton = [&](const Candidate &candidate) {
      Operation *anchor = candidate.op;
      if (seenAnchors.contains(anchor)) {
        LDBG("  [Phase1] Skip: Anchor " << *anchor << " already visited.");
        return;
      }
      seenAnchors.insert(anchor);

      if (runAnchorPrecheck(anchor, forbiddenAnchors) !=
          AnchorPrecheckResult::Allow) {
        return;
      }
      TilingUnit unit = buildUnitSkeleton(ts, anchor, candidate, analysisCache);

      LDBG("  [Phase1] Created Skeleton for Anchor "
           << *anchor << " with " << unit.producerOps.size() << " producers.");

      units.push_back(std::move(unit));
    };

    // 1. Process High-Priority Candidates
    for (const auto &cand : candidates) {
      tryExtractSkeleton(cand);
    }

    // 2. Process Fallback Candidates
    if (enableFallback) {
      for (Operation *op : llvm::reverse(ts.base.ops)) {
        if (seenAnchors.contains(op) || isDynamicOrScalarOp(op))
          continue;
        if (isa<TilingInterface>(op) || isConvertibleElementwiseOp(op)) {
          auto it = llvm::find(ts.base.ops, op);
          size_t irIdx =
              static_cast<size_t>(std::distance(ts.base.ops.begin(), it));
          tryExtractSkeleton(Candidate{op, Priority::Fallback, irIdx});
        }
      }
    }

    return units;
  }

  SmallVector<Candidate> collectCandidates(const SubStage &ss) const {
    DenseMap<Operation *, Candidate> best;

    for (auto [i, op] : llvm::enumerate(ss.ops)) {
      if (isWriteOp(op)) {
        if (isSuppressedWriteAnchor(op)) {
          LDBG("    [collectCandidates] Suppressed write anchor: " << *op);
          continue;
        }
        recordBestCandidate(best, op, Priority::Normalized, i);
      }
    }

    if (!ss.ops.empty()) {
      if (auto yield = dyn_cast<scf::YieldOp>(
              ss.ops.back()->getBlock()->getTerminator())) {
        for (Value v : yield.getOperands()) {
          Operation *def = v.getDefiningOp();
          if (!def || !llvm::is_contained(ss.ops, def))
            continue;

          bool feedsNorm = llvm::any_of(
              def->getUsers(), [](Operation *u) { return isWriteOp(u); });
          if ((isa<TilingInterface>(def) || isConvertibleElementwiseOp(def)) &&
              !feedsNorm) {
            auto it = llvm::find(ss.ops, def);
            recordBestCandidate(best, def, Priority::Yield,
                                std::distance(ss.ops.begin(), it));
          }
        }
      }
    }

    SmallVector<Candidate> res;
    for (auto &kv : best)
      res.push_back(kv.second);
    llvm::sort(res, [](const Candidate &a, const Candidate &b) {
      return std::tie(a.prio, a.irIdx) < std::tie(b.prio, b.irIdx);
    });
    LDBG("  [collectCandidates] number of candidates: " << res.size());
    return res;
  }

  //===--------------------------------------------------------------------===//
  // Phase 2: Analyze & Tag
  //===--------------------------------------------------------------------===//

  SmallVector<Operation *, 16>
  buildRecurrenceAnalysisOps(const TilingUnit &unit,
                             const DenseMap<Operation *, BufferEscapeSummary>
                                 &bufferEscapeSummaries) const {
    DenseSet<Operation *> localOps(unit.producerOps.begin(),
                                   unit.producerOps.end());
    localOps.insert(unit.anchorOp);

    DenseSet<Operation *> seen;
    SmallVector<Operation *, 16> externalOps;
    for (const auto &it : bufferEscapeSummaries) {
      for (Operation *op : it.second.forwardSlice) {
        if (localOps.contains(op) || isControlFlowOp(op) ||
            !seen.insert(op).second) {
          continue;
        }
        externalOps.push_back(op);
      }
    }
    llvm::sort(externalOps, [](Operation *lhs, Operation *rhs) {
      return lhs->isBeforeInBlock(rhs);
    });

    SmallVector<Operation *, 16> analysisOps(externalOps.begin(),
                                             externalOps.end());
    analysisOps.append(unit.producerOps.begin(), unit.producerOps.end());
    analysisOps.push_back(unit.anchorOp);
    llvm::sort(analysisOps, [](Operation *lhs, Operation *rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
    return analysisOps;
  }

  /// Performs dimension analysis, size calculation, and IR tagging per Unit.
  LogicalResult analyzeUnits(TiledSubStage &ts,
                             SmallVector<TilingUnit, 4> &skeletons,
                             SmallVectorImpl<TilingUnit> &plannedUnits,
                             const SubStageAnalysisCache &analysisCache) const {
    for (auto [idx, unit] : llvm::enumerate(skeletons)) {
      LDBG("  [Phase2] Analyzing Unit "
           << idx << " (Anchor: " << *(unit.anchorOp) << ")");
      DenseMap<Operation *, BufferEscapeSummary> bufferEscapeSummaries =
          analysisCache.collectBufferEscapeSummaries(unit);

      UnitRuleContext preDimRuleCtx{
          unit,           ts.stage,      bufferEscapeSummaries,
          &analysisCache, tileAllBlocks, std::nullopt,
          nullptr,
      };
      switch (evaluateBufferEscapePolicy(preDimRuleCtx)) {
      case EscapePolicyResult::Allow:
        break;
      case EscapePolicyResult::SkipUnit:
        LDBG("    [Skip] Unit "
             << idx << " skipped due to stage-escape policy. Anchor: "
             << *unit.anchorOp);
        unit.anchorOp->emitWarning("Skipping unit because its alloc closure "
                                   "escapes the current stage");
        continue;
      case EscapePolicyResult::FailPass:
        LDBG("    [Fail] Unit " << idx << " rejected by escape policy. Anchor: "
                                << *unit.anchorOp);
        unit.anchorOp->emitError(
            "Unit buffer escape policy violated, aborting tiling");
        return failure();
      }

      if (!runUnitRules(kPreDimUnitRules, preDimRuleCtx)) {
        LDBG("    [Fail] Unit " << idx << " rejected by eligibility rules. "
                                << "Anchor: " << *unit.anchorOp);
        unit.anchorOp->emitError(
            "Unit eligibility rules violated, aborting tiling");
        return failure();
      }
      // Extract ops and sort them by module IR order to form a local context
      SmallVector<Operation *, 16> localOps = unit.producerOps;
      localOps.push_back(unit.anchorOp);
      llvm::sort(localOps, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });

      // Run local dimension analysis to choose the unit's tiling dimension
      // from the anchor's own iteration space.
      DimAnalyzer localAnalyzer(localOps);
      auto dims = localAnalyzer.analyzeAndGetTilingDims();
      if (dims.empty()) {
        LDBG("    [Error] DimAnalyzer found no valid dims. Failing unit.");
        unit.anchorOp->emitError("DimAnalyzer found no valid tiling dims for "
                                 "anchor op");
        return failure();
      }

      // Sort to ensure determinism in fallback attempts
      llvm::sort(dims);
      unit.candidateDims.assign(dims.begin(), dims.end());

      // Attempt to find a valid tile size with fallback support
      auto selectedDim =
          selectFirstValidDim(unit.anchorOp, unit.candidateDims, tripCount);
      if (!selectedDim) {
        LDBG("    [Error] No perfect tiling possible for any candidate dims. "
             "Failing unit.");
        unit.anchorOp->emitError("No perfect tiling possible for any "
                                 "candidate dims (candidates: ")
            << llvm::join(
                   llvm::map_range(unit.candidateDims,
                                   [](int64_t d) { return std::to_string(d); }),
                   ", ")
            << "), tripCount=" << tripCount;
        return failure();
      }

      // Populate the selected tiling dimension and tile sizes.
      unit.tilingDimIndex = *selectedDim;
      std::optional<int64_t> tileSize =
          calculateTileSizeForUnit(unit.anchorOp, *selectedDim, tripCount);
      if (!tileSize) {
        unit.anchorOp->emitError("Failed to materialize tile size for the "
                                 "selected tiling dimension");
        return failure();
      }
      unit.tileSizes.resize(unit.rank, 0);
      unit.tileSizes[*selectedDim] = *tileSize;

      SmallVector<Operation *, 16> recurrenceAnalysisOps =
          buildRecurrenceAnalysisOps(unit, bufferEscapeSummaries);
      DimAnalyzer recurrenceAnalyzer(recurrenceAnalysisOps);
      recurrenceAnalyzer.runAnalysis();

      UnitRuleContext postDimRuleCtx{
          unit,
          ts.stage,
          bufferEscapeSummaries,
          &analysisCache,
          tileAllBlocks,
          unit.tilingDimIndex,
          &recurrenceAnalyzer,
      };
      if (!runUnitRules(kPostDimUnitRules, postDimRuleCtx)) {
        LDBG("    [Fail] Unit "
             << idx << " rejected by post-dimension legality rules. Anchor: "
             << *unit.anchorOp);
        unit.anchorOp->emitError(
            "Unit legality rules violated after tiling dimension selection");
        return failure();
      }

      // Populate string tags for MLIR attribute injection
      unit.anchorTag = getStageOpToTile(ts.base.stageId, ts.base.index, idx);
      unit.producerComputeTag =
          getStageProducerToFuse(ts.base.stageId, ts.base.index, idx);
      unit.producerAllocTag = stage_attrs::kStageProducerAllocToFuseAttr.str();
      unit.crossUserTag = stage_attrs::kCrossTillUnitAttr.str();

      plannedUnits.push_back(std::move(unit));
    }

    return success();
  }

  void
  selectNonOverlappingUnits(ArrayRef<TilingUnit> plannedUnits,
                            SmallVectorImpl<TilingUnit> &activeUnits) const {
    SmallVector<TilingUnit, 4> ordered(plannedUnits.begin(),
                                       plannedUnits.end());
    llvm::sort(ordered, [](const TilingUnit &lhs, const TilingUnit &rhs) {
      if (lhs.priority != rhs.priority)
        return lhs.priority < rhs.priority;
      return lhs.irOrder > rhs.irOrder;
    });

    DenseMap<Operation *, unsigned> opOwners;
    for (TilingUnit &unit : ordered) {
      if (!tryClaimUnitWithSharedProducers(unit, activeUnits, opOwners)) {
        continue;
      }

      activeUnits.push_back(unit);
      LDBG("    [Select] Accepted disjoint unit anchored at "
           << *unit.anchorOp << " (priority=" << getPriorityName(unit.priority)
           << ", irOrder=" << unit.irOrder
           << ", ownedProducers=" << unit.ownedProducerOps.size()
           << ", borrowedProducers=" << unit.borrowedProducerOps.size() << ")");
    }
  }

  void materializeProvisionalUnitAttrs(ArrayRef<TilingUnit> units) const {
    for (const TilingUnit &unit : units) {
      commitUnit(unit);
      LDBG("    [Tag] Materialized provisional tiling attrs for unit anchor "
           << *unit.anchorOp);
    }
  }

  void clearProvisionalUnitAttrs(ArrayRef<TilingUnit> units) const {
    DenseSet<Operation *> visited;
    for (const TilingUnit &unit : units) {
      auto clearIfNeeded = [&](Operation *op) {
        if (!op || !visited.insert(op).second)
          return;
        clearTilingUnitAttrs(op);
        LDBG("    [Tag] Cleared provisional tile placement attrs from " << *op);
      };

      clearIfNeeded(unit.anchorOp);
      for (Operation *producer : unit.producerOps)
        clearIfNeeded(producer);
      for (Operation *producer : unit.ownedProducerOps)
        clearIfNeeded(producer);
      for (Operation *producer : unit.borrowedProducerOps)
        clearIfNeeded(producer);
    }
  }

  void collectSuppressedAnchors(
      const ContractionClosureAnalysis &analysis,
      SmallVectorImpl<Operation *> &suppressedAnchors) const {
    for (const SuppressionDecision &decision : analysis.getDecisions()) {
      if (!decision.anchorOp)
        continue;
      suppressedAnchors.push_back(decision.anchorOp);
      LDBG("    [Suppression] " << decision.reason << ": "
                                << *decision.anchorOp);
    }
  }

  void
  applySuppressionMarkers(const ContractionClosureAnalysis &analysis) const {
    MLIRContext *ctx = nullptr;
    for (const SuppressionDecision &decision : analysis.getDecisions()) {
      if (!ctx && decision.anchorOp)
        ctx = decision.anchorOp->getContext();
    }
    if (!ctx)
      return;

    UnitAttr marker = UnitAttr::get(ctx);
    for (const SuppressionDecision &decision : analysis.getDecisions()) {
      if (decision.anchorOp &&
          !decision.anchorOp->hasAttr(stage_attrs::kNoTileAttr))
        decision.anchorOp->setAttr(stage_attrs::kNoTileAttr, marker);

      for (Operation *op : decision.suppressedOps) {
        if (op && !op->hasAttr(stage_attrs::kNoTileAttr))
          op->setAttr(stage_attrs::kNoTileAttr, marker);
      }
      for (Operation *root : decision.suppressedRoots) {
        if (root && !root->hasAttr(stage_attrs::kNoTileAttr))
          root->setAttr(stage_attrs::kNoTileAttr, marker);
      }
    }
  }

  /// Injects dictionary attributes and tags into the IR safely.
  /// Decouples attribute preparation logic from the actual IR mutation.
  void commitUnit(const TilingUnit &u) const {
    MLIRContext *ctx = u.anchorOp->getContext();
    OpBuilder b(ctx);

    LDBG("    [Tag] Preparing tags for anchor " << *u.anchorOp << " with "
                                                << u.anchorTag);
    UnitAttr unitAttr = b.getUnitAttr();

    // Struct to hold pending attribute mutations
    struct PendingTag {
      Operation *op;
      StringRef attrName;
      Attribute attrValue;
    };
    SmallVector<PendingTag, 8> pendingMutations;

    // 1. Prepare Anchor op tags
    setTileMeta(u.anchorOp, u.anchorTag, u.producerComputeTag, u.tileSizes);
    pendingMutations.push_back({u.anchorOp, u.anchorTag, unitAttr});
    if (u.anchorNeedsProducerTag) {
      pendingMutations.push_back({u.anchorOp, u.producerComputeTag, unitAttr});
      LDBG("      [Tag] Anchor also exports producer ownership for shared use: "
           << *u.anchorOp);
    }
    if (u.anchorHasCrossUsers) {
      pendingMutations.push_back({u.anchorOp, u.crossUserTag, unitAttr});
      LDBG(
          "      [Tag] Anchor marked as cross-unit shared op: " << *u.anchorOp);
    }

    // 2. Prepare Producer ops tags
    ArrayRef<Operation *> ownedProducers = u.ownedProducerOps;
    if (ownedProducers.empty() && u.borrowedProducerOps.empty())
      ownedProducers = u.producerOps;

    for (Operation *p : ownedProducers) {
      if (u.hasCrossUsers.lookup(p)) {
        pendingMutations.push_back({p, u.crossUserTag, unitAttr});
      }

      StringRef tag =
          u.isMemProducer.lookup(p) ? u.producerAllocTag : u.producerComputeTag;
      pendingMutations.push_back({p, tag, unitAttr});
    }

    // --- Phase B: Execute Final IR Mutations ---
    for (const auto &mutation : pendingMutations) {
      if (isControlFlowOp(mutation.op)) {
        LDBG("      Skipping control-flow op (forbidden target): "
             << *mutation.op);
        continue;
      }
      mutation.op->setAttr(mutation.attrName, mutation.attrValue);
    }
    LDBG("    [Tag] Materialized " << pendingMutations.size()
                                   << " provisional attr mutation(s) for "
                                   << *u.anchorOp);
  }
};

//===----------------------------------------------------------------------===//
// Pass Main Entry
//===----------------------------------------------------------------------===//

/// Tagging pass that builds the stage tiling plan.
class StageTilingPlanPass
    : public mlir::dicp::LinalgExt::impl::StageTilingPlanBase<
          StageTilingPlanPass> {
public:
  using StageTilingPlanBase::StageTilingPlanBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
    int64_t tripCount = static_cast<int64_t>(tiledMixVectorLoopNumber);
    LDBG("Run StageTilingPlanPass with tripCount="
         << tripCount
         << ", tileAllBlocks=" << (tileAllBlocks ? "true" : "false"));
    IRRewriter rewriter(funcOp.getContext());
    StagePartitionOptions options;
    options.reorderEnabled = true;
    options.tileAllBlocks = tileAllBlocks;
    StagePartition pipeline(funcOp, options);
    auto partitionResults = pipeline.run(rewriter);
    // Pipeline encapsulates Block discovery, Partitioning, and Materialization.
    if (failed(partitionResults)) {
      funcOp.emitError("Failed to successfully partition and schedule stages.");
      return;
    }

    // Tag generated stages before tiling analysis.
    for (auto &partitionRes : partitionResults.value()) {
      for (StageInfo &stageInfo : partitionRes.stages) {
        for (auto op : stageInfo.getOps()) {
          if (isControlFlowOp(op)) {
            LDBG("  [StageTag] Skipping control-flow op (forbidden target): "
                 << *op);
            continue;
          }
          op->setAttr(stage_attrs::kNPUStageAttrName,
                      rewriter.getI32IntegerAttr(stageInfo.id));
        }
      }
    }

    // Analyze stages and assign transform tags.
    TilingProcessor processor(tripCount, &aliasAnalysis, tileAllBlocks);
    for (auto &partitionRes : partitionResults.value()) {
      for (StageInfo &stageInfo : partitionRes.stages) {
        // Keep the stage eligibility gate enabled until the backend can safely
        // consume every staged pattern emitted by this pipeline.
        if (!isStageEligibleForTiling(stageInfo)) {
          funcOp.emitError("Stage ")
              << stageInfo.id << " skipped: failed eligibility check";
          return signalPassFailure();
        }
        for (auto &subStage : stageInfo.subStages) {
          TiledSubStage ts(subStage, stageInfo);
          if (failed(processor.process(ts))) {
            funcOp.emitError("Processor failed to analyze SubStage ")
                << ts.base.index << " of Stage " << stageInfo.id;
            return signalPassFailure();
          }
        }
      }
    }
    LDBG("StageTilingPlanPass completed successfully.");
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::func::FuncOp>>
mlir::dicp::LinalgExt::createStageTilingPlanPass(
    const StageTilingPlanOptions &options) {
  return std::make_unique<StageTilingPlanPass>(options);
}

std::unique_ptr<OperationPass<mlir::func::FuncOp>>
mlir::dicp::LinalgExt::createStageTilingPlanPass(unsigned vectorTile) {
  StageTilingPlanOptions opt;
  opt.tiledMixVectorLoopNumber = vectorTile;
  return std::make_unique<StageTilingPlanPass>(opt);
}
