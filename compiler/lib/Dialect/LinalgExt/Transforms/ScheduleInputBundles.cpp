#include "dicp/Dialect/LinalgExt/Analysis/SliceParallelAnalysis.h"
#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"
#include "dicp/Utils/Utils.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "schedule-input-bundles"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace mlir::dicp;
using namespace mlir::dicp::LinalgExt;
using namespace mlir::dicp::stage_attrs;

namespace mlir::dicp::LinalgExt {
#define GEN_PASS_DEF_SCHEDULEINPUTBUNDLES
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace mlir::dicp::LinalgExt

namespace {

struct MemoryAccessSummary {
  SmallVector<MemoryRegion, 4> reads;
  SmallVector<MemoryRegion, 4> writes;
  bool hasUnknownEffects = false;
};

struct InputBundle {
  /// Storage root of the local buffer bundle.
  memref::AllocOp alloc;
  /// Tensor adapter consumed by the first compute op in the same substage.
  bufferization::ToTensorOp toTensor;
  /// Operation that makes the local buffer data-ready.
  memref::CopyOp readyCopy;
  SmallVector<Operation *, 8> members;
  SmallVector<Value, 8> producedValues;
  SmallVector<MemoryRegion, 4> sourceRegions;
  DenseSet<Operation *> memberSet;
  StageSubUnitTag tag;
};

static bool isSyncBlockOp(Operation *op) {
  return op && op->getName().getStringRef().starts_with("hivm.hir.sync_block_");
}

static bool isSchedulingBarrier(Operation *op) {
  return op && (isSyncBlockOp(op) || isStructuredControlFlowOp(op) ||
                op->hasTrait<OpTrait::IsTerminator>());
}

static bool isInSameLinearSubStage(Operation *op, const StageSubUnitTag &tag) {
  if (!op || !tag.isValid() || isSyncBlockOp(op))
    return false;

  if (std::optional<int64_t> stageId = getStageId(op))
    if (*stageId != tag.stage)
      return false;

  if (std::optional<StageSubUnitTag> opTag = getStageSubUnitTag(op))
    return isSameStageAndSubstage(*opTag, tag);
  return true;
}

static LogicalResult rejectBundle(memref::AllocOp alloc, Twine reason) {
  LDBG("[Bundle] reject " << alloc << ": " << reason);
  return failure();
}

static void addBundleMember(InputBundle &bundle, Operation *op) {
  if (bundle.memberSet.insert(op).second)
    bundle.members.push_back(op);
}

static bool isTrackedAliasOp(Operation *op) {
  return isa<ViewLikeOpInterface, memref::CastOp>(op);
}

static bool isSingleResultSingleUseOp(Operation *op) {
  return op && op->getNumResults() == 1 && op->getResult(0).hasOneUse();
}

static std::optional<Value> getAliasSource(Value value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  return TypeSwitch<Operation *, std::optional<Value>>(defOp)
      .Case<ViewLikeOpInterface>([](ViewLikeOpInterface viewLike) {
        return std::optional<Value>(viewLike.getViewSource());
      })
      .Case<memref::CastOp>([](memref::CastOp castOp) {
        return std::optional<Value>(castOp.getSource());
      })
      .Default([](Operation *) { return std::nullopt; });
}

static std::optional<MemoryRegion> getRegionOrRoot(Value value) {
  if (!value || !isa<BaseMemRefType>(value.getType()))
    return std::nullopt;
  if (std::optional<MemoryRegion> region = getMemoryRegion(value))
    return region;
  return MemoryRegion{traceToSourceRoot(value)};
}

static bool mayConflict(const MemoryRegion &lhs, const MemoryRegion &rhs,
                        AliasAnalysis &aliasAnalysis) {
  if (!lhs.isValid() || !rhs.isValid())
    return true;

  if (aliasAnalysis.alias(lhs.baseRoot, rhs.baseRoot) == AliasResult::NoAlias)
    return false;

  if (lhs.baseRoot != rhs.baseRoot)
    return true;

  FailureOr<bool> disjoint = proveMemoryRegionsDisjoint(lhs, rhs);
  return failed(disjoint) || !*disjoint;
}

static MemoryAccessSummary summarizeMemoryAccess(Operation *op) {
  MemoryAccessSummary summary;

  auto addRegion = [&](SmallVectorImpl<MemoryRegion> &regions,
                       Value value) -> LogicalResult {
    std::optional<MemoryRegion> region = getRegionOrRoot(value);
    if (!region)
      return failure();
    regions.push_back(*region);
    return success();
  };

  if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
    if (failed(addRegion(summary.reads, copyOp.getSource())) ||
        failed(addRegion(summary.writes, copyOp.getTarget()))) {
      summary.hasUnknownEffects = true;
    }
    return summary;
  }

  if (auto materialize =
          dyn_cast<bufferization::MaterializeInDestinationOp>(op)) {
    if (!isa<BaseMemRefType>(materialize.getDest().getType()) ||
        failed(addRegion(summary.writes, materialize.getDest()))) {
      summary.hasUnknownEffects = true;
    }
    return summary;
  }

  if (auto memEffects = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance, 4> effects;
    memEffects.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      Value effectedValue = effect.getValue();
      if (!effectedValue) {
        summary.hasUnknownEffects = true;
        return summary;
      }
      if (!isa<BaseMemRefType>(effectedValue.getType()))
        continue;

      if (isa<MemoryEffects::Read>(effect.getEffect())) {
        if (failed(addRegion(summary.reads, effectedValue))) {
          summary.hasUnknownEffects = true;
          return summary;
        }
        continue;
      }

      if (isa<MemoryEffects::Write, MemoryEffects::Allocate,
              MemoryEffects::Free>(effect.getEffect())) {
        if (failed(addRegion(summary.writes, effectedValue))) {
          summary.hasUnknownEffects = true;
          return summary;
        }
      }
    }
    return summary;
  }

  if (!isMemoryEffectFree(op))
    summary.hasUnknownEffects = true;
  return summary;
}

static DenseMap<Operation *, unsigned> buildOrderIndex(Block *block) {
  DenseMap<Operation *, unsigned> orderIndex;
  unsigned order = 0;
  for (Operation &op : *block)
    orderIndex[&op] = order++;
  return orderIndex;
}

static void sortAndFinalizeBundle(InputBundle &bundle) {
  DenseMap<Operation *, unsigned> orderIndex =
      buildOrderIndex(bundle.alloc->getBlock());
  llvm::sort(bundle.members, [&](Operation *lhs, Operation *rhs) {
    return orderIndex.lookup(lhs) < orderIndex.lookup(rhs);
  });

  bundle.members.erase(
      std::unique(bundle.members.begin(), bundle.members.end()),
      bundle.members.end());
  bundle.producedValues.clear();
  for (Operation *member : bundle.members)
    llvm::append_range(bundle.producedValues, member->getResults());
}

static bool usesBundleValue(Operation *op, const InputBundle &bundle) {
  return llvm::any_of(op->getOperands(), [&](Value operand) {
    return llvm::is_contained(bundle.producedValues, operand);
  });
}

static bool hasIllegalExternalUsers(const InputBundle &bundle) {
  Operation *readyOp = bundle.readyCopy;
  for (Value producedValue : bundle.producedValues) {
    for (Operation *user : producedValue.getUsers()) {
      if (bundle.memberSet.contains(user))
        continue;
      if (user->getBlock() != readyOp->getBlock()) {
        LDBG("[Bundle] reject " << bundle.alloc
                                << ": produced value escapes current block via "
                                << *user);
        return true;
      }
      if (!isInSameLinearSubStage(user, bundle.tag)) {
        LDBG("[Bundle] reject "
             << bundle.alloc << ": produced value escapes current substage via "
             << *user);
        return true;
      }
      if (user->isBeforeInBlock(readyOp)) {
        LDBG("[Bundle] reject " << bundle.alloc
                                << ": produced value is consumed before the "
                                   "bundle becomes data-ready: "
                                << *user);
        return true;
      }
    }
  }
  return false;
}

static Operation *findFirstRealConsumer(const InputBundle &bundle) {
  for (Operation *op = bundle.readyCopy->getNextNode(); op;
       op = op->getNextNode()) {
    if (!isInSameLinearSubStage(op, bundle.tag) || isSchedulingBarrier(op))
      return nullptr;
    if (usesBundleValue(op, bundle))
      return op;
  }
  return nullptr;
}

static void collectSingleUseSourceChain(Value source, InputBundle &bundle) {
  while (Operation *defOp = source.getDefiningOp()) {
    if (!isTrackedAliasOp(defOp) ||
        defOp->getBlock() != bundle.alloc->getBlock() ||
        !isInSameLinearSubStage(defOp, bundle.tag) ||
        !isSingleResultSingleUseOp(defOp)) {
      return;
    }

    addBundleMember(bundle, defOp);
    std::optional<Value> next = getAliasSource(defOp->getResult(0));
    if (!next)
      return;
    source = *next;
  }
}

static LogicalResult collectInputBundle(memref::AllocOp alloc,
                                        AliasAnalysis &aliasAnalysis,
                                        InputBundle &bundle) {
  if (!alloc->hasAttr(kStageProducerAllocToFuseAttr))
    return failure();

  bundle.alloc = alloc;
  addBundleMember(bundle, alloc.getOperation());

  DenseSet<Value> visitedValues;
  SmallVector<Value, 4> worklist{alloc.getResult()};

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visitedValues.insert(current).second)
      continue;

    for (Operation *user : current.getUsers()) {
      if (user->getBlock() != alloc->getBlock()) {
        return rejectBundle(alloc, Twine("alloc user escapes block: ") +
                                       user->getName().getStringRef());
      }

      LogicalResult result =
          TypeSwitch<Operation *, LogicalResult>(user)
              .Case<bufferization::ToTensorOp>(
                  [&](bufferization::ToTensorOp toTensor) {
                    if (toTensor.getBuffer() != current)
                      return success();
                    if (bundle.toTensor && bundle.toTensor != toTensor) {
                      return rejectBundle(alloc,
                                          "multiple to_tensor ops found");
                    }
                    bundle.toTensor = toTensor;
                    addBundleMember(bundle, toTensor.getOperation());
                    return success();
                  })
              .Case<memref::CopyOp>([&](memref::CopyOp copyOp) {
                if (copyOp.getTarget() != current)
                  return success();
                if (bundle.readyCopy && bundle.readyCopy != copyOp) {
                  return rejectBundle(alloc,
                                      "multiple ready-copy writers found");
                }
                std::optional<StageSubUnitTag> tag = getStageSubUnitTag(copyOp);
                if (!tag || !tag->isValid()) {
                  return rejectBundle(
                      alloc, Twine("ready copy has no valid substage tag: ") +
                                 copyOp->getName().getStringRef());
                }
                bundle.readyCopy = copyOp;
                bundle.tag = *tag;
                addBundleMember(bundle, copyOp.getOperation());
                return success();
              })
              .Default([&](Operation *op) {
                if (!isTrackedAliasOp(op))
                  return failure();
                std::optional<Value> aliasSource =
                    getAliasSource(op->getResult(0));
                if (!aliasSource || *aliasSource != current)
                  return success();
                addBundleMember(bundle, op);
                for (Value result : op->getResults())
                  if (isa<BaseMemRefType>(result.getType()))
                    worklist.push_back(result);
                return success();
              });
      if (failed(result)) {
        return rejectBundle(alloc, Twine("unsupported alloc-derived user: ") +
                                       user->getName().getStringRef());
      }
    }
  }

  if (!bundle.readyCopy || !bundle.toTensor || !bundle.tag.isValid()) {
    return rejectBundle(alloc, "expected alloc + to_tensor + ready copy");
  }

  if (std::optional<int64_t> stageId = getStageId(alloc.getOperation()))
    if (*stageId != bundle.tag.stage) {
      return rejectBundle(alloc, "alloc stage id does not match ready copy");
    }

  if (llvm::any_of(bundle.members, [&](Operation *member) {
        return member != alloc.getOperation() &&
               !isInSameLinearSubStage(member, bundle.tag);
      })) {
    return rejectBundle(alloc, "bundle members do not stay in one substage");
  }

  collectSingleUseSourceChain(bundle.readyCopy.getSource(), bundle);
  if (std::optional<MemoryRegion> sourceRegion =
          getRegionOrRoot(bundle.readyCopy.getSource())) {
    bundle.sourceRegions.push_back(*sourceRegion);
  } else {
    return rejectBundle(alloc, "failed to classify ready-copy source region");
  }

  sortAndFinalizeBundle(bundle);
  if (hasIllegalExternalUsers(bundle))
    return failure();

  LDBG("[Bundle] accept " << alloc << " with " << bundle.members.size()
                          << " member ops in stage " << bundle.tag.stage << "."
                          << bundle.tag.sub);
  return success();
}

static bool canCrossDown(Operation *candidate, const InputBundle &bundle,
                         AliasAnalysis &aliasAnalysis) {
  if (!isInSameLinearSubStage(candidate, bundle.tag) ||
      isSchedulingBarrier(candidate) || usesBundleValue(candidate, bundle))
    return false;

  MemoryAccessSummary access = summarizeMemoryAccess(candidate);
  if (access.hasUnknownEffects) {
    LDBG("[Sink] block on unknown memory effects: " << *candidate);
    return false;
  }

  std::optional<MemoryRegion> localRegion =
      getRegionOrRoot(bundle.alloc->getResult(0));
  if (!localRegion) {
    LDBG("[Sink] block because the bundle root region is not analyzable: "
         << bundle.alloc);
    return false;
  }

  auto touchesLocalRegion = [&](const SmallVectorImpl<MemoryRegion> &regions) {
    return llvm::any_of(regions, [&](const MemoryRegion &region) {
      return mayConflict(region, *localRegion, aliasAnalysis);
    });
  };
  if (touchesLocalRegion(access.reads) || touchesLocalRegion(access.writes)) {
    LDBG("[Sink] block on local-buffer interference: " << *candidate);
    return false;
  }

  for (const MemoryRegion &writeRegion : access.writes) {
    if (llvm::any_of(
            bundle.sourceRegions, [&](const MemoryRegion &sourceRegion) {
              return mayConflict(writeRegion, sourceRegion, aliasAnalysis);
            })) {
      LDBG("[Sink] block on source-buffer write interference: " << *candidate);
      return false;
    }
  }

  return true;
}

static bool sinkInputBundle(InputBundle &bundle, RewriterBase &rewriter,
                            AliasAnalysis &aliasAnalysis) {
  Operation *firstConsumer = findFirstRealConsumer(bundle);
  if (!firstConsumer) {
    LDBG("[Sink] skip " << bundle.alloc
                        << ": no real consumer found in the same substage");
    return false;
  }
  LDBG("[Sink] first real consumer for " << bundle.alloc << " is "
                                         << *firstConsumer);

  Operation *bundleTail = bundle.members.back();
  Operation *insertionPoint = firstConsumer;
  for (Operation *probe = bundleTail->getNextNode();
       probe && probe != firstConsumer; probe = probe->getNextNode()) {
    if (!canCrossDown(probe, bundle, aliasAnalysis)) {
      insertionPoint = probe;
      break;
    }
  }

  LDBG("[Sink] sink input bundle rooted at " << bundle.alloc
                                             << " to just "
                                                "before "
                                             << *insertionPoint);
  bool moved = false;
  for (Operation *member : bundle.members) {
    if (member == insertionPoint)
      continue;
    if (member->getNextNode() != insertionPoint)
      moved = true;
    rewriter.moveOpBefore(member, insertionPoint);
  }
  if (!moved) {
    LDBG("[Sink] skip " << bundle.alloc
                        << ": bundle is already materialized at the target");
    return false;
  }
  return true;
}

struct ScheduleInputBundlesPass
    : public mlir::dicp::LinalgExt::impl::ScheduleInputBundlesBase<
          ScheduleInputBundlesPass> {
  using mlir::dicp::LinalgExt::impl::ScheduleInputBundlesBase<
      ScheduleInputBundlesPass>::ScheduleInputBundlesBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    AliasAnalysis aliasAnalysis(func);
    IRRewriter rewriter(func.getContext());
    bool changed = false;

    SmallVector<memref::AllocOp, 16> allocs;
    func.walk([&](memref::AllocOp alloc) { allocs.push_back(alloc); });

    LDBG("[Pass] run ScheduleInputBundles on @" << func.getName());
    for (memref::AllocOp alloc : allocs) {
      if (!alloc->getBlock())
        continue;

      InputBundle bundle;
      if (failed(collectInputBundle(alloc, aliasAnalysis, bundle)))
        continue;

      if (sinkInputBundle(bundle, rewriter, aliasAnalysis))
        changed = true;
    }

    LDBG("[Pass] ScheduleInputBundles completed; changed="
         << (changed ? "true" : "false"));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::dicp::LinalgExt::createScheduleInputBundlesPass() {
  return std::make_unique<ScheduleInputBundlesPass>();
}
