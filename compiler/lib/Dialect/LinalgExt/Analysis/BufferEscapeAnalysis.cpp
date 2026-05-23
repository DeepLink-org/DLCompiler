#include "dicp/Dialect/LinalgExt/Analysis/BufferEscapeAnalysis.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dicp-buffer-escape-analysis"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace mlir::dicp;

namespace {

static Operation *getAliasAnalysisRoot(Operation *op) {
  if (auto funcOp = op->getParentOfType<func::FuncOp>())
    return funcOp.getOperation();
  return op->getParentOp();
}

/// Returns true if `value` is derived from `rootValue` through view-like ops
/// or buffer/tensor adapter wrappers.
static bool isValueDerivedFromRoot(Value value, Value rootValue) {
  SmallVector<Value> worklist{value};
  DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    if (current == rootValue)
      return true;

    Operation *defOp = current.getDefiningOp();
    if (!defOp)
      continue;

    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(defOp)) {
      worklist.push_back(viewLike.getViewSource());
      continue;
    }
    if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(defOp)) {
      worklist.push_back(toTensor.getBuffer());
      continue;
    }
    if (auto toBuffer = dyn_cast<bufferization::ToBufferOp>(defOp)) {
      worklist.push_back(toBuffer.getTensor());
      continue;
    }
  }

  return false;
}

static bool mayModifyRootBuffer(Operation *op, Value rootBuffer,
                                AliasAnalysis &aliasAnalysis) {
  if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
    return isValueDerivedFromRoot(copyOp.getTarget(), rootBuffer);
  }
  if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
    return isValueDerivedFromRoot(storeOp.getMemRef(), rootBuffer);
  }
  if (auto materializeOp =
          dyn_cast<bufferization::MaterializeInDestinationOp>(op)) {
    return isValueDerivedFromRoot(materializeOp.getDest(), rootBuffer);
  }

  auto memEffects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffects)
    return false;

  SmallVector<MemoryEffects::EffectInstance> effects;
  memEffects.getEffects(effects);
  for (const MemoryEffects::EffectInstance &effect : effects) {
    if (!isa<MemoryEffects::Write, MemoryEffects::Allocate,
             MemoryEffects::Free>(effect.getEffect())) {
      continue;
    }

    Value effectedValue = effect.getValue();
    if (!effectedValue) {
      LDBG("    -> Ignore root-agnostic effect on " << op->getName());
      continue;
    }

    if (!isa<BaseMemRefType>(effectedValue.getType())) {
      LDBG("    -> Ignore non-buffer effect value: " << effectedValue);
      continue;
    }

    if (isValueDerivedFromRoot(effectedValue, rootBuffer)) {
      LDBG("    -> Effect writes through a root-derived alias: "
           << effectedValue);
      return true;
    }

    auto aliasResult = aliasAnalysis.alias(effectedValue, rootBuffer);
    if (aliasResult != AliasResult::NoAlias)
      return true;
  }

  return false;
}

} // namespace

BufferEscapeSummary
mlir::dicp::analyzeBufferEscape(memref::AllocOp allocOp,
                                function_ref<bool(Operation *)> isInScope,
                                AliasAnalysis *aliasAnalysis) {
  BufferEscapeSummary summary;
  ForwardSliceOptions options;
  options.filter = [&](Operation *op) {
    return op->getBlock() == allocOp->getBlock();
  };
  getForwardSlice(allocOp.getOperation(), &summary.forwardSlice, options);

  AliasAnalysis localAlias(getAliasAnalysisRoot(allocOp));
  AliasAnalysis &aa = aliasAnalysis ? *aliasAnalysis : localAlias;

  LDBG("[BufferEscape] Analyzing alloc: " << *allocOp);
  LDBG("[BufferEscape] Forward slice size: " << summary.forwardSlice.size());

  for (Operation *op : summary.forwardSlice) {
    if (isInScope(op))
      continue;

    summary.externalUsers.push_back(op);

    if (isa<CallOpInterface>(op)) {
      summary.callEscapes.push_back(op);
      LDBG("  [ExternalUser] call escape: " << *op);
      continue;
    }

    if (mayModifyRootBuffer(op, allocOp.getResult(), aa)) {
      summary.modifyingEscapes.push_back(op);
      LDBG("  [ExternalUser] modifying escape: " << *op);
    } else {
      LDBG("  [ExternalUser] read-only escape: " << *op);
    }
  }

  return summary;
}
