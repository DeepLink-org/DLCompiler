//===- ContractionClosureAnalysis.cpp - Contraction Closure Analysis ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "dicp/Dialect/LinalgExt/Analysis/ContractionClosureAnalysis.h"
#include "dicp/Dialect/LinalgExt/Analysis/DimAnalyzer.h"
#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "contraction-closure-analysis"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

namespace mlir {
namespace dicp {

using namespace llvm;

namespace {

struct ValueDimsWorkItem {
  Value value;
  SmallVector<int64_t, 4> relevantPositions;
};

static std::string formatDims(ArrayRef<int64_t> dims) {
  return llvm::formatv("[{0}]",
                       llvm::join(llvm::map_range(dims,
                                                  [](int64_t dim) {
                                                    return std::to_string(dim);
                                                  }),
                                  ", "))
      .str();
}

static void sortAndUniquePositions(SmallVectorImpl<int64_t> &positions) {
  llvm::sort(positions);
  positions.erase(std::unique(positions.begin(), positions.end()),
                  positions.end());
}

static bool
mergeRelevantPositions(DenseMap<Value, SmallVector<int64_t, 4>> &knownPositions,
                       Value value, ArrayRef<int64_t> newPositions,
                       SmallVectorImpl<ValueDimsWorkItem> &worklist) {
  if (newPositions.empty())
    return false;

  SmallVector<int64_t, 4> &known = knownPositions[value];
  bool changed = false;
  for (int64_t pos : newPositions) {
    if (!llvm::is_contained(known, pos)) {
      known.push_back(pos);
      changed = true;
    }
  }
  if (!changed)
    return false;

  sortAndUniquePositions(known);
  worklist.push_back(ValueDimsWorkItem{value, known});
  return true;
}

static bool addUniqueOp(SmallVectorImpl<ChainOpInfo> &chainOps,
                        Operation *candidate, FeederRole role) {
  if (llvm::any_of(chainOps, [&](const ChainOpInfo &info) {
        return info.op == candidate;
      })) {
    return false;
  }
  chainOps.push_back(ChainOpInfo{candidate, role, /*affectsUntiledDims=*/true});
  return true;
}

static bool addUniqueRoot(SmallVectorImpl<Operation *> &roots,
                          Operation *root) {
  if (llvm::is_contained(roots, root))
    return false;
  roots.push_back(root);
  return true;
}

static bool sameRankShaped(Value lhs, Value rhs) {
  auto lhsType = dyn_cast<ShapedType>(lhs.getType());
  auto rhsType = dyn_cast<ShapedType>(rhs.getType());
  return lhsType && rhsType && lhsType.hasRank() && rhsType.hasRank() &&
         lhsType.getRank() == rhsType.getRank();
}

static SmallVector<int64_t, 4>
mapTransposeResultPositionsToInput(ArrayRef<int64_t> resultPositions,
                                   ArrayRef<int64_t> permutation) {
  SmallVector<int64_t, 4> inputPositions;
  inputPositions.reserve(resultPositions.size());
  for (int64_t resultPos : resultPositions) {
    if (resultPos < 0 || resultPos >= static_cast<int64_t>(permutation.size()))
      continue;
    inputPositions.push_back(permutation[resultPos]);
  }
  sortAndUniquePositions(inputPositions);
  return inputPositions;
}

static bool isWriteIntoValue(Operation *op, Value target) {
  return TypeSwitch<Operation *, bool>(op)
      .Case<memref::CopyOp>(
          [&](memref::CopyOp copyOp) { return copyOp.getTarget() == target; })
      .Case<memref::StoreOp>([&](memref::StoreOp storeOp) {
        return storeOp.getMemref() == target;
      })
      .Case<linalg::CopyOp>([&](linalg::CopyOp copyOp) {
        return llvm::is_contained(copyOp.getOutputs(), target);
      })
      .Case<bufferization::MaterializeInDestinationOp>(
          [&](bufferization::MaterializeInDestinationOp materializeOp) {
            return materializeOp.getDest() == target;
          })
      .Case<tensor::InsertSliceOp>([&](tensor::InsertSliceOp insertSliceOp) {
        return insertSliceOp.getDest() == target;
      })
      .Default([](Operation *) { return false; });
}

template <typename Predicate>
static const TilingUnit *findTilingUnit(ArrayRef<TilingUnit> tilingUnits,
                                        Predicate &&predicate) {
  auto it = llvm::find_if(
      tilingUnits, [&](const TilingUnit &unit) { return predicate(unit); });
  return it == tilingUnits.end() ? nullptr : &*it;
}

static void appendUniqueSignedDims(SmallVectorImpl<int64_t> &dst,
                                   ArrayRef<unsigned> src) {
  for (unsigned dim : src)
    dst.push_back(static_cast<int64_t>(dim));
  sortAndUniquePositions(dst);
}

static SmallVector<int64_t, 4>
mapResultPositionsToLoopDims(linalg::LinalgOp contraction, OpResult result,
                             ArrayRef<int64_t> resultPositions) {
  SmallVector<int64_t, 4> loopDims;
  AffineMap resultMap = contraction.getIndexingMapMatchingResult(result);
  for (int64_t resultPos : resultPositions) {
    if (resultPos < 0 ||
        resultPos >= static_cast<int64_t>(resultMap.getNumResults()))
      continue;
    if (auto dimExpr =
            dyn_cast<AffineDimExpr>(resultMap.getResult(resultPos))) {
      loopDims.push_back(dimExpr.getPosition());
    }
  }
  sortAndUniquePositions(loopDims);
  return loopDims;
}

} // namespace

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//

ContractionClosureAnalysis::ContractionClosureAnalysis(
    const SubStage &subStageInfo, ArrayRef<TilingUnit> tilingUnits)
    : subStageInfo(subStageInfo), tilingUnits(tilingUnits) {
  for (Operation *op : subStageInfo.ops)
    scopeOpSet.insert(op);
}

ContractionClosureAnalysis::~ContractionClosureAnalysis() = default;

//===----------------------------------------------------------------------===//
// Main Analysis Entry
//===----------------------------------------------------------------------===//

LogicalResult ContractionClosureAnalysis::analyze() {
  LDBG("=== Starting ContractionClosureAnalysis ===");

  contractions.clear();
  semantics.clear();
  closures.clear();
  rootClasses.clear();
  decisionIndexByAnchor.clear();
  decisions.clear();
  provenanceOps.clear();
  provenanceAnalyzer.reset();

  if (failed(initializeProvenanceAnalysis()))
    return failure();

  discoverContractions();
  if (contractions.empty())
    return success();

  for (Operation *contractionOp : contractions) {
    FailureOr<std::optional<ContractionTileSemantics>> semanticsOr =
        parseTileSemantics(contractionOp);
    if (failed(semanticsOr))
      return failure();
    if (!*semanticsOr)
      continue;
    semantics.push_back(std::move(**semanticsOr));
  }

  if (semantics.empty()) {
    LDBG("No contraction with resolvable tiling provenance found");
    return success();
  }

  for (const ContractionTileSemantics &sem : semantics) {
    if (!sem.hasTiledDims())
      continue;

    for (const auto &[operand, operandLoopDims] : sem.operandToDims) {
      if (operandLoopDims.empty())
        continue;

      bool touchesTiledDim = llvm::any_of(operandLoopDims, [&](int64_t dim) {
        return sem.isTiledLoopDim(dim);
      });
      if (touchesTiledDim)
        continue;

      FeederClosure closure = traceFeederClosure(
          operand, sem.operandRelevantPositions.lookup(operand),
          operandLoopDims, sem.contractionOp);
      if (!closure.isSupported) {
        LDBG("  [Skip] Unsupported feeder closure for contraction "
             << *sem.contractionOp << ": " << closure.nonExclusiveReason);
        continue;
      }
      if (closure.storageRoots.empty())
        continue;
      closures.push_back(std::move(closure));
    }
  }

  buildRootEquivalenceClasses();

  for (FeederClosure &closure : closures)
    (void)checkExclusivity(closure);

  generateDecisions();
  LDBG("=== ContractionClosureAnalysis Complete ===");
  LDBG("Generated " << decisions.size() << " suppression decision(s)");
  return success();
}

LogicalResult ContractionClosureAnalysis::initializeProvenanceAnalysis() {
  provenanceOps.assign(subStageInfo.ops.begin(), subStageInfo.ops.end());
  llvm::sort(provenanceOps, [](Operation *lhs, Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });

  provenanceAnalyzer = std::make_unique<DimAnalyzer>(provenanceOps);
  provenanceAnalyzer->runAnalysis();
  return success();
}

//===----------------------------------------------------------------------===//
// Phase 1: Contraction Discovery
//===----------------------------------------------------------------------===//

void ContractionClosureAnalysis::discoverContractions() {
  DenseSet<Operation *> seen;
  for (Operation *op : subStageInfo.ops) {
    if (!seen.insert(op).second)
      continue;
    if (isContractionAnchor(op))
      contractions.push_back(op);
  }
}

bool ContractionClosureAnalysis::isContractionAnchor(Operation *op) const {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  return linalgOp && linalg::isaContractionOpInterface(linalgOp);
}

//===----------------------------------------------------------------------===//
// Phase 2: Tile Semantics Parsing
//===----------------------------------------------------------------------===//

FailureOr<std::optional<ContractionTileSemantics>>
ContractionClosureAnalysis::parseTileSemantics(Operation *op) {
  auto contraction = dyn_cast<linalg::LinalgOp>(op);
  if (!contraction)
    return std::optional<ContractionTileSemantics>();

  const TilingUnit *tileUnit = nullptr;
  Operation *tileAnchorOp = nullptr;
  bool hasInlineTileMeta = static_cast<bool>(getTileMeta(op));

  if (hasInlineTileMeta) {
    tileAnchorOp = op;
    tileUnit = findTilingUnit(tilingUnits, [&](const TilingUnit &unit) {
      return unit.anchorOp == op;
    });
    if (!tileUnit) {
      op->emitError()
          << "Failed to resolve owning tiling unit for contraction anchor";
      return failure();
    }
  }

  if (auto fuseTag = getProducerFuseTag(op)) {
    std::string fuseTagName =
        getStageProducerToFuse(fuseTag->stage, fuseTag->sub, fuseTag->unit);
    const TilingUnit *fusedUnit =
        findTilingUnit(tilingUnits, [&](const TilingUnit &unit) {
          return unit.producerComputeTag == fuseTagName;
        });
    if (!fusedUnit) {
      op->emitError() << "Failed to resolve producer_to_fuse tag '"
                      << fuseTagName << "' to an owning tiling unit";
      return failure();
    }

    if (!hasInlineTileMeta) {
      tileUnit = fusedUnit;
      tileAnchorOp = fusedUnit->anchorOp;
    } else if (fusedUnit != tileUnit) {
      op->emitError()
          << "Contraction carries conflicting inline tile metadata and "
             "producer_to_fuse ownership";
      return failure();
    }
  }

  if (!tileAnchorOp || !tileUnit) {
    LDBG("Skipping contraction without tile metadata or producer-to-fuse "
         "ownership: "
         << *op);
    return std::optional<ContractionTileSemantics>();
  }

  if (failed(getTileSizes(tileAnchorOp))) {
    tileAnchorOp->emitError()
        << "Resolved tiling anchor lacks finalized tile metadata";
    return failure();
  }

  FailureOr<SmallVector<int64_t, 4>> tiledLoopDims =
      resolveContractionTiledLoopDims(contraction, *tileUnit);
  if (failed(tiledLoopDims))
    return failure();

  ContractionTileSemantics sem;
  sem.contractionOp = op;
  sem.tileUnit = tileUnit;
  sem.tiledLoopDims.assign(tiledLoopDims->begin(), tiledLoopDims->end());
  sortAndUniquePositions(sem.tiledLoopDims);
  for (int64_t loopDim : sem.tiledLoopDims) {
    if (loopDim < 0 || loopDim >= contraction.getNumLoops()) {
      op->emitError() << "Resolved tiled contraction loop dim out of range: "
                      << loopDim;
      return failure();
    }
  }

  FailureOr<linalg::ContractionDimensions> dims =
      linalg::inferContractionDims(contraction);
  if (failed(dims)) {
    op->emitError()
        << "Failed to infer contraction dimensions from indexing maps";
    return failure();
  }

  appendUniqueSignedDims(sem.parallelDims, dims->batch);
  appendUniqueSignedDims(sem.parallelDims, dims->m);
  appendUniqueSignedDims(sem.parallelDims, dims->n);
  appendUniqueSignedDims(sem.reductionDims, dims->k);

  for (int64_t dim : sem.parallelDims) {
    if (sem.isTiledLoopDim(dim))
      sem.tiledParallelDims.push_back(dim);
  }
  for (int64_t dim : sem.reductionDims) {
    if (sem.isTiledLoopDim(dim))
      sem.tiledReductionDims.push_back(dim);
  }

  computeOperandDimBindings(sem);

  LDBG("[TileSemantics] contraction="
       << *op << " tile-anchor=" << *tileAnchorOp
       << " owning-unit-dim=" << *tileUnit->tilingDimIndex
       << " tiled-loop-dims=" << formatDims(sem.tiledLoopDims)
       << " tiled-parallel=" << formatDims(sem.tiledParallelDims)
       << " tiled-reduction=" << formatDims(sem.tiledReductionDims));
  return std::optional<ContractionTileSemantics>(std::move(sem));
}

FailureOr<SmallVector<int64_t, 4>>
ContractionClosureAnalysis::resolveContractionTiledLoopDims(
    linalg::LinalgOp contraction, const TilingUnit &tileUnit) const {
  if (!provenanceAnalyzer) {
    contraction->emitError()
        << "ContractionClosureAnalysis provenance analyzer is not initialized";
    return failure();
  }
  if (!tileUnit.tilingDimIndex) {
    contraction->emitError()
        << "ContractionClosureAnalysis requires a selected tiling "
           "dimension on the owning unit";
    return failure();
  }

  Value anchorValue = getTilingReferenceValue(tileUnit.anchorOp);
  if (!anchorValue) {
    contraction->emitError()
        << "Resolved tiling anchor has no shaped reference value";
    return failure();
  }

  std::optional<int64_t> trackedRoot =
      provenanceAnalyzer->getDimRoot(anchorValue, *tileUnit.tilingDimIndex);
  if (!trackedRoot) {
    contraction->emitError()
        << "Failed to resolve anchor dimension root with DimAnalyzer";
    return failure();
  }

  if (contraction->getNumResults() == 0) {
    contraction->emitError()
        << "DimAnalyzer-based tile provenance requires a tensor result";
    return failure();
  }

  Value resultValue = contraction->getResult(0);
  auto resultType = dyn_cast<ShapedType>(resultValue.getType());
  if (!resultType || !resultType.hasRank()) {
    contraction->emitError()
        << "Contraction result is not a ranked shaped value";
    return failure();
  }

  SmallVector<int64_t, 4> resultPositions;
  for (int64_t dim = 0, e = resultType.getRank(); dim < e; ++dim) {
    std::optional<int64_t> resultRoot =
        provenanceAnalyzer->getDimRoot(resultValue, dim);
    if (resultRoot && *resultRoot == *trackedRoot)
      resultPositions.push_back(dim);
  }
  if (resultPositions.empty()) {
    contraction->emitError()
        << "DimAnalyzer failed to map the owning anchor dimension back to "
           "the contraction result";
    return failure();
  }

  return mapResultPositionsToLoopDims(contraction, cast<OpResult>(resultValue),
                                      resultPositions);
}

void ContractionClosureAnalysis::computeOperandDimBindings(
    ContractionTileSemantics &sem) {
  auto contraction = cast<linalg::LinalgOp>(sem.contractionOp);

  for (OpOperand *inputOperand : contraction.getDpsInputOperands()) {
    Value operand = inputOperand->get();
    auto operandType = dyn_cast<ShapedType>(operand.getType());
    if (!operandType || !operandType.hasRank())
      continue;

    AffineMap map =
        contraction.getIndexingMapsArray()[inputOperand->getOperandNumber()];
    SmallVector<int64_t> operandLoopDims;
    SmallVector<int64_t> relevantPositions;

    for (auto [resultIdx, expr] : llvm::enumerate(map.getResults())) {
      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr)
        continue;

      int64_t loopDim = dimExpr.getPosition();
      operandLoopDims.push_back(loopDim);
      if (!sem.isTiledLoopDim(loopDim))
        relevantPositions.push_back(static_cast<int64_t>(resultIdx));
    }

    sortAndUniquePositions(operandLoopDims);
    sortAndUniquePositions(relevantPositions);
    sem.operandToDims[operand] = std::move(operandLoopDims);
    sem.operandRelevantPositions[operand] = std::move(relevantPositions);
  }
}

//===----------------------------------------------------------------------===//
// Phase 3: Dimension-Aware Backward Slice
//===----------------------------------------------------------------------===//

FeederClosure ContractionClosureAnalysis::traceFeederClosure(
    Value operand, ArrayRef<int64_t> relevantPositions,
    ArrayRef<int64_t> relevantOperandDims, Operation *contractionOp) const {
  FeederClosure closure;
  closure.contractionOp = contractionOp;
  closure.operand = operand;
  closure.relevantOperandDims.assign(relevantOperandDims.begin(),
                                     relevantOperandDims.end());

  SmallVector<ValueDimsWorkItem> worklist;
  DenseMap<Value, SmallVector<int64_t, 4>> knownPositions;
  SmallVector<int64_t, 4> initialPositions(relevantPositions.begin(),
                                           relevantPositions.end());
  sortAndUniquePositions(initialPositions);
  mergeRelevantPositions(knownPositions, operand, initialPositions, worklist);

  while (!worklist.empty()) {
    ValueDimsWorkItem item = worklist.pop_back_val();
    Operation *defOp = item.value.getDefiningOp();
    if (!defOp) {
      closure.isSupported = false;
      closure.nonExclusiveReason =
          "Operand feeder reaches a block argument or external SSA value";
      return closure;
    }
    if (!isInScope(defOp)) {
      closure.isSupported = false;
      closure.nonExclusiveReason =
          "Operand feeder reaches an operation outside the current substage";
      return closure;
    }

    FeederRole role = classifyFeederRole(defOp);
    addUniqueOp(closure.chainOps, defOp, role);

    if (role == FeederRole::Root) {
      addUniqueRoot(closure.storageRoots, defOp);
      continue;
    }
    if (role == FeederRole::Terminator) {
      closure.isSupported = false;
      closure.nonExclusiveReason =
          "Operand feeder crosses a control-flow or call boundary";
      return closure;
    }

    SmallVector<std::pair<Value, SmallVector<int64_t, 4>>, 2> predecessors;
    bool supported =
        TypeSwitch<Operation *, bool>(defOp)
            .Case<bufferization::ToTensorOp>(
                [&](bufferization::ToTensorOp toTensorOp) {
                  predecessors.push_back(
                      {toTensorOp.getOperand(), item.relevantPositions});
                  return true;
                })
            .Case<tensor::CastOp>([&](tensor::CastOp castOp) {
              if (!sameRankShaped(castOp.getSource(), castOp.getDest()))
                return false;
              predecessors.push_back(
                  {castOp.getSource(), item.relevantPositions});
              return true;
            })
            .Case<memref::CastOp>([&](memref::CastOp castOp) {
              if (!sameRankShaped(castOp.getSource(), castOp.getResult()))
                return false;
              predecessors.push_back(
                  {castOp.getSource(), item.relevantPositions});
              return true;
            })
            .Case<memref::ReinterpretCastOp>(
                [&](memref::ReinterpretCastOp castOp) {
                  if (!sameRankShaped(castOp.getSource(), castOp.getResult()))
                    return false;
                  predecessors.push_back(
                      {castOp.getSource(), item.relevantPositions});
                  return true;
                })
            .Case<linalg::TransposeOp>([&](linalg::TransposeOp transposeOp) {
              SmallVector<int64_t, 4> inputPositions =
                  mapTransposeResultPositionsToInput(
                      item.relevantPositions, transposeOp.getPermutation());
              predecessors.push_back(
                  {transposeOp.getInput(), std::move(inputPositions)});
              return true;
            })
            .Case<ViewLikeOpInterface>([&](ViewLikeOpInterface viewOp) {
              Value source = viewOp.getViewSource();
              if (!sameRankShaped(source, item.value))
                return false;
              predecessors.push_back({source, item.relevantPositions});
              return true;
            })
            .Default([](Operation *) { return false; });

    if (!supported) {
      closure.isSupported = false;
      closure.nonExclusiveReason =
          llvm::formatv("Unsupported feeder op while tracing "
                        "tile-invariant closure: {0}",
                        defOp->getName().getStringRef())
              .str();
      return closure;
    }

    for (auto &[predValue, predPositions] : predecessors)
      mergeRelevantPositions(knownPositions, predValue, predPositions,
                             worklist);
  }

  for (Operation *root : closure.storageRoots) {
    if (root->getNumResults() != 1)
      continue;

    Value rootValue = root->getResult(0);
    for (Operation *user : rootValue.getUsers()) {
      if (!isInScope(user) || !isWriteIntoValue(user, rootValue))
        continue;
      addUniqueOp(closure.chainOps, user, classifyFeederRole(user));
    }
  }

  return closure;
}

FeederRole ContractionClosureAnalysis::classifyFeederRole(Operation *op) const {
  return TypeSwitch<Operation *, FeederRole>(op)
      .Case<memref::AllocOp, tensor::EmptyOp, bufferization::AllocTensorOp>(
          [](auto) { return FeederRole::Root; })
      .Case<bufferization::ToTensorOp,
            bufferization::MaterializeInDestinationOp>(
          [](auto) { return FeederRole::Bridge; })
      .Case<linalg::TransposeOp, memref::CastOp, memref::ReinterpretCastOp,
            tensor::CastOp>([](auto) { return FeederRole::Transform; })
      .Case<memref::CopyOp, linalg::CopyOp, tensor::InsertSliceOp>(
          [](auto) { return FeederRole::Copy; })
      .Case<scf::ForOp, scf::WhileOp, func::CallOp>(
          [](auto) { return FeederRole::Terminator; })
      .Default([](Operation *) { return FeederRole::Other; });
}

//===----------------------------------------------------------------------===//
// Phase 4: Root Equivalence & Exclusivity
//===----------------------------------------------------------------------===//

void ContractionClosureAnalysis::buildRootEquivalenceClasses() {
  for (FeederClosure &closure : closures) {
    for (Operation *root : closure.storageRoots) {
      rootClasses[root].root = root;
      rootClasses[root].closures.push_back(&closure);
    }
  }

  for (auto &[root, cls] : rootClasses) {
    if (!cls.isShared())
      continue;

    LDBG("[RootEq] Root " << *root << " shared by " << cls.closures.size()
                          << " closure(s)");
    for (FeederClosure *closure : cls.closures) {
      closure->isExclusive = false;
      closure->nonExclusiveReason =
          "Storage root is shared by multiple contraction feeder closures";
    }
  }
}

bool ContractionClosureAnalysis::checkExclusivity(FeederClosure &closure) {
  if (!closure.isSupported)
    return false;
  if (!closure.nonExclusiveReason.empty())
    return false;
  if (!analyzeClosureExits(closure)) {
    closure.isExclusive = false;
    return false;
  }
  closure.isExclusive = true;
  return true;
}

bool ContractionClosureAnalysis::analyzeClosureExits(FeederClosure &closure) {
  DenseSet<Operation *> chainSet;
  for (const ChainOpInfo &info : closure.chainOps)
    chainSet.insert(info.op);

  for (const ChainOpInfo &info : closure.chainOps) {
    for (Value result : info.op->getResults()) {
      if (!isa<ShapedType>(result.getType()))
        continue;

      for (Operation *user : result.getUsers()) {
        if (chainSet.contains(user))
          continue;
        if (user == closure.contractionOp)
          continue;

        ClosureExit exit{result, user, /*isExternal=*/!isInScope(user)};
        closure.exits.push_back(exit);
        closure.nonExclusiveReason =
            exit.isExternal ? "Value escapes the current substage"
                            : "Value has an internal user outside the closure";
        LDBG("  [Exit] contraction=" << *closure.contractionOp
                                     << " exit-user=" << *user << " reason="
                                     << closure.nonExclusiveReason);
        return false;
      }
    }
  }

  return true;
}

bool ContractionClosureAnalysis::isInScope(Operation *op) const {
  return scopeOpSet.contains(op);
}

//===----------------------------------------------------------------------===//
// Phase 5: Decision Generation
//===----------------------------------------------------------------------===//

void ContractionClosureAnalysis::generateDecisions() {
  for (FeederClosure &closure : closures) {
    if (!closure.isExclusive)
      continue;

    for (const TilingUnit &unit : tilingUnits) {
      if (!isUnitAnchorInClosure(unit, closure))
        continue;

      auto [it, inserted] =
          decisionIndexByAnchor.try_emplace(unit.anchorOp, decisions.size());
      if (inserted) {
        SuppressionDecision decision;
        decision.action = SuppressionDecision::MarkNoTile;
        decision.anchorOp = unit.anchorOp;
        decision.reason =
            llvm::formatv("Anchor participates in a contraction feeder "
                          "closure that is invariant to the selected tiled "
                          "semantic dims of {0}",
                          closure.contractionOp->getName().getStringRef())
                .str();
        for (const ChainOpInfo &chainOp : closure.chainOps)
          decision.suppressedOps.push_back(chainOp.op);
        decision.suppressedRoots.append(closure.storageRoots.begin(),
                                        closure.storageRoots.end());
        decision.evidence.push_back(&closure);
        decisions.push_back(std::move(decision));
        LDBG("[Decision] Suppressing tiling anchor "
             << *unit.anchorOp << " because of " << *closure.contractionOp);
        continue;
      }

      decisions[it->second].evidence.push_back(&closure);
      for (const ChainOpInfo &chainOp : closure.chainOps) {
        if (!llvm::is_contained(decisions[it->second].suppressedOps,
                                chainOp.op))
          decisions[it->second].suppressedOps.push_back(chainOp.op);
      }
      for (Operation *root : closure.storageRoots) {
        if (!llvm::is_contained(decisions[it->second].suppressedRoots, root))
          decisions[it->second].suppressedRoots.push_back(root);
      }
    }
  }
}

ArrayRef<SuppressionDecision> ContractionClosureAnalysis::getDecisions() const {
  return decisions;
}

bool ContractionClosureAnalysis::isUnitAnchorInClosure(
    const TilingUnit &unit, const FeederClosure &closure) const {
  if (!unit.anchorOp)
    return false;

  if (llvm::any_of(closure.chainOps, [&](const ChainOpInfo &info) {
        return info.op == unit.anchorOp;
      })) {
    return true;
  }
  return llvm::is_contained(closure.storageRoots, unit.anchorOp);
}

} // namespace dicp
} // namespace mlir
