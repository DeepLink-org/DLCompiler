//===- DuplicateStageProducers.cpp - Duplicate cross-unit producers -------===//
//
// This pass performs use-site node duplication for SIMD-like pure compute ops.
// When a producer in one tiling unit is directly consumed by an op in another
// unit of the same stage/substage, the pass clones a private producer closure
// next to the consumer. The goal is to trade redundant computation for extra
// fusion freedom by breaking shared SSA edges across units.
//
// The pass is intentionally conservative:
//   * only `isSIMDLikeOp` producers are considered;
//   * only memory-effect-free SSA producers are duplicated;
//   * only same-block, same-stage, same-substage, different-unit uses match;
//   * tile anchor metadata is never propagated to duplicated ops.
//
//===----------------------------------------------------------------------===//

#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "duplicate-stage-producers"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace mlir::dicp;
using namespace mlir::dicp::LinalgExt;
using namespace mlir::dicp::stage_attrs;

namespace mlir::dicp::LinalgExt {
#define GEN_PASS_DEF_DUPLICATESTAGEPRODUCERS
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace mlir::dicp::LinalgExt

namespace {

static bool isSameStageAndSubDifferentUnit(const StageSubUnitTag &lhs,
                                           const StageSubUnitTag &rhs) {
  return isSameStageAndSubstage(lhs, rhs) && lhs.unit != rhs.unit;
}

/// Returns true if the operation is a pure SSA producer that is safe to
/// duplicate via redundant computation.
static bool isClonableSIMDLikeProducer(Operation *op) {
  if (!isSIMDLikeOp(op)) {
    LDBG("[Match] reject non-SIMD-like op: " << op->getName());
    return false;
  }
  if (op->getNumResults() == 0) {
    LDBG("[Match] reject zero-result op: " << *op);
    return false;
  }
  if (llvm::any_of(op->getResultTypes(), llvm::IsaPred<MemRefType>)) {
    LDBG("[Match] reject memref-producing op: " << *op);
    return false;
  }
  if (!isMemoryEffectFree(op)) {
    LDBG("[Match] reject side-effecting op: " << *op);
    return false;
  }
  return true;
}

/// Remove metadata that must never be copied to a duplicated private producer
/// and retag the clone for the target unit.
static void retagClonedOpForUnit(Operation *op, const StageSubUnitTag &tag) {
  clearTilingUnitAttrs(op);
  op->setAttr(getStageProducerToFuse(tag.stage, tag.sub, tag.unit),
              UnitAttr::get(op->getContext()));

  LDBG("[Clone] retag clone for stage=" << tag.stage << ", sub=" << tag.sub
                                        << ", unit=" << tag.unit << ": "
                                        << *op);
}

static void setCrossUnitAttr(Operation *op, StringRef reason) {
  op->setAttr(kCrossTillUnitAttr, UnitAttr::get(op->getContext()));
  LDBG("[CrossUnit] mark " << reason << ": " << *op);
}

static void clearCrossUnitAttr(Operation *op, StringRef reason) {
  clearCrossUnitUserAttrs(op);
  LDBG("[CrossUnit] clear " << reason << ": " << *op);
}

static bool hasCrossUnitUser(Operation *op,
                             const StageSubUnitTag &producerTag) {
  return llvm::any_of(op->getResults(), [&](Value result) {
    return llvm::any_of(result.getUses(), [&](OpOperand &use) {
      auto userTag = getProducerFuseTag(use.getOwner());
      return userTag && isSameStageAndSubDifferentUnit(producerTag, *userTag);
    });
  });
}

static void logClosureStop(StringRef reason, Operation *op) {
  LDBG("[Closure] stop at " << reason << ": " << *op);
}

static void logClosureReuse(Operation *op) {
  LDBG("[Closure] reuse existing def: " << *op);
}

static void refreshCrossUnitAttr(Operation *op) {
  auto producerTag = getProducerFuseTag(op);
  if (!producerTag) {
    clearCrossUnitAttr(op, "untagged producer");
    return;
  }

  if (hasCrossUnitUser(op, *producerTag)) {
    setCrossUnitAttr(op, "producer still has cross-unit users");
    return;
  }

  clearCrossUnitAttr(op, "localized producer");
}

/// Marks operand producers that still cross tiling units. This keeps the IR
/// self-describing for later debugging and subsequent greedy rounds.
static void markResidualCrossUnitOperands(Operation *op) {
  auto consumerTag = getProducerFuseTag(op);
  if (!consumerTag)
    return;

  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp)
      continue;

    auto defTag = getProducerFuseTag(defOp);
    if (!defTag || !isSameStageAndSubDifferentUnit(*defTag, *consumerTag))
      continue;

    setCrossUnitAttr(defOp, "residual cross-unit operand producer");
    LDBG("[CrossUnit] user requiring residual marker: " << *op);
  }
}

/// Collect a minimal clone closure by recursively following same-stage/substage
/// cross-unit SIMD-like producer operands. The closure is recorded in
/// topological order (defs before users) via post-order insertion.
static LogicalResult collectCloneClosure(Operation *op,
                                         const StageSubUnitTag &targetTag,
                                         DenseSet<Operation *> &visited,
                                         SetVector<Operation *> &closure) {
  if (!visited.insert(op).second)
    return success();

  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp)
      continue;

    if (defOp->getBlock() != op->getBlock()) {
      logClosureStop("cross-block def", defOp);
      continue;
    }

    auto defTag = getProducerFuseTag(defOp);
    if (!defTag) {
      logClosureStop("untagged def", defOp);
      continue;
    }

    if (!isSameStageAndSubDifferentUnit(*defTag, targetTag)) {
      logClosureReuse(defOp);
      continue;
    }

    if (!isClonableSIMDLikeProducer(defOp)) {
      logClosureStop("non-clonable cross-unit def", defOp);
      continue;
    }

    if (failed(collectCloneClosure(defOp, targetTag, visited, closure)))
      return failure();
  }

  closure.insert(op);
  return success();
}

static FailureOr<Operation *>
cloneClosureNextToUser(PatternRewriter &rewriter, Operation *root,
                       Operation *user, const StageSubUnitTag &targetTag) {
  DenseSet<Operation *> visited;
  SetVector<Operation *> closure;
  if (failed(collectCloneClosure(root, targetTag, visited, closure)))
    return failure();

  if (closure.empty())
    return failure();

  LDBG("[Clone] build private closure for user: " << *user);
  for (Operation *candidate : closure)
    LDBG("[Clone] closure op: " << *candidate);

  IRMapping mapping;
  Operation *clonedRoot = nullptr;
  Operation *lastClone = nullptr;
  for (Operation *op : closure) {
    if (lastClone)
      rewriter.setInsertionPointAfter(lastClone);
    else
      rewriter.setInsertionPoint(user);

    Operation *clone = rewriter.clone(*op, mapping);
    retagClonedOpForUnit(clone, targetTag);
    markResidualCrossUnitOperands(clone);
    lastClone = clone;

    if (op == root)
      clonedRoot = clone;
  }

  if (!clonedRoot)
    return failure();
  return clonedRoot;
}

struct QualifiedUserSite {
  Operation *user = nullptr;
  StageSubUnitTag userTag;
  SmallVector<OpOperand *, 2> operands;
};

/// Use-site duplication pattern for cross-unit shared SIMD-like producers.
struct DuplicateCrossUnitSIMDProducerPattern : public RewritePattern {
  DuplicateCrossUnitSIMDProducerPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isClonableSIMDLikeProducer(op))
      return failure();

    auto producerTag = getProducerFuseTag(op);
    if (!producerTag) {
      LDBG("[Match] reject untagged producer: " << *op);
      return failure();
    }

    DenseMap<Operation *, unsigned> userToIndex;
    SmallVector<QualifiedUserSite, 4> qualifiedUsers;
    for (Value result : op->getResults()) {
      for (OpOperand &use : llvm::make_early_inc_range(result.getUses())) {
        Operation *user = use.getOwner();
        if (user == op)
          continue;
        if (user->getBlock() != op->getBlock()) {
          LDBG("[Match] skip cross-block user: " << *user);
          continue;
        }

        auto userTag = getProducerFuseTag(user);
        if (!userTag) {
          LDBG("[Match] skip untagged user: " << *user);
          continue;
        }
        if (!isSameStageAndSubDifferentUnit(*producerTag, *userTag)) {
          LDBG("[Match] skip same-unit/non-peer use. producer(stage="
               << producerTag->stage << ", sub=" << producerTag->sub
               << ", unit=" << producerTag->unit
               << ") user(stage=" << userTag->stage << ", sub=" << userTag->sub
               << ", unit=" << userTag->unit << ")");
          continue;
        }

        unsigned idx = 0;
        auto [it, inserted] =
            userToIndex.try_emplace(user, qualifiedUsers.size());
        if (inserted) {
          QualifiedUserSite site;
          site.user = user;
          site.userTag = *userTag;
          qualifiedUsers.push_back(site);
        }
        idx = it->second;
        qualifiedUsers[idx].operands.push_back(&use);
      }
    }

    if (qualifiedUsers.empty())
      return failure();

    LDBG("[Match] Duplicating producer " << *op << " for "
                                         << qualifiedUsers.size()
                                         << " cross-unit use-site(s).");

    bool changed = false;
    for (QualifiedUserSite &site : qualifiedUsers) {
      FailureOr<Operation *> clonedRoot =
          cloneClosureNextToUser(rewriter, op, site.user, site.userTag);
      if (failed(clonedRoot)) {
        LDBG(
            "[Clone] failed to build private closure for user: " << *site.user);
        continue;
      }

      rewriter.modifyOpInPlace(site.user, [&]() {
        for (OpOperand *operand : site.operands) {
          auto oldResult = dyn_cast<OpResult>(operand->get());
          if (!oldResult || oldResult.getOwner() != op)
            continue;

          operand->set((*clonedRoot)->getResult(oldResult.getResultNumber()));
          LDBG("[Rewrite] replace use in " << *site.user
                                           << " with cloned result "
                                           << oldResult.getResultNumber());
        }
      });
      changed = true;
    }

    if (!changed)
      return failure();

    refreshCrossUnitAttr(op);
    markResidualCrossUnitOperands(op);
    return success();
  }
};

struct DuplicateStageProducersPass
    : public mlir::dicp::LinalgExt::impl::DuplicateStageProducersBase<
          DuplicateStageProducersPass> {
  using DuplicateStageProducersBase::DuplicateStageProducersBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    LDBG("[Pass] start duplicate-stage-producers on function "
         << funcOp.getSymName() << " ===");

    RewritePatternSet patterns(ctx);
    patterns.add<DuplicateCrossUnitSIMDProducerPattern>(ctx);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      LDBG("[Pass] duplicate-stage-producers failed on function "
           << funcOp.getSymName());
      signalPassFailure();
      return;
    }

    LDBG("[Pass] duplicate-stage-producers completed on function "
         << funcOp.getSymName() << " ===");
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::dicp::LinalgExt::createDuplicateStageProducersPass() {
  return std::make_unique<DuplicateStageProducersPass>();
}
