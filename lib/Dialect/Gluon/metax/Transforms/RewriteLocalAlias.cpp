#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gluon-rewrite-local-alias"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[alias] " << X << "\n")

using namespace mlir;
namespace ttg = ::mlir::triton::gpu;

namespace mlir::triton::gluon {

#define GEN_PASS_DEF_GLUONREWRITELOCALALIASPASS
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h.inc"

namespace {

using AliasClass = SmallVector<LocalAliasOp, 4>;

uint64_t getStorageBits(ttg::MemDescType type) {
  return static_cast<uint64_t>(type.getNumElements()) *
         type.getElementTypeBitWidth();
}

LogicalResult rewriteLocalAliases(ModuleOp module) {
  DenseMap<Operation *, AliasClass> aliasClasses;
  DenseMap<LocalAliasOp, Operation *> aliasToAllocation;

  WalkResult collection = module.walk([&](Operation *op) -> WalkResult {
    if (auto allocation = dyn_cast<ttg::LocalAllocOp>(op)) {
      aliasClasses.try_emplace(allocation);
      return WalkResult::advance();
    }

    auto alias = dyn_cast<LocalAliasOp>(op);
    if (!alias)
      return WalkResult::advance();

    Operation *sourceOp = alias.getSrc().getDefiningOp();
    Operation *allocation = nullptr;
    if (isa_and_nonnull<ttg::LocalAllocOp>(sourceOp)) {
      allocation = sourceOp;
    } else if (auto sourceAlias = dyn_cast_or_null<LocalAliasOp>(sourceOp)) {
      allocation = aliasToAllocation.lookup(sourceAlias);
    }

    if (!allocation || !aliasClasses.contains(allocation)) {
      alias.emitOpError(
          "must refer directly to ttg.local_alloc or a preceding "
          "gluon.local_alias in the same storage-alias class");
      return WalkResult::interrupt();
    }

    aliasClasses[allocation].push_back(alias);
    aliasToAllocation[alias] = allocation;
    return WalkResult::advance();
  });
  if (collection.wasInterrupted())
    return failure();
  if (aliasToAllocation.empty())
    return success();

  DenseMap<Operation *, ttg::MemDescType> largestStorageType;
  for (auto &[allocation, aliases] : aliasClasses) {
    auto largestType = cast<ttg::MemDescType>(allocation->getResult(0).getType());
    uint64_t largestBits = getStorageBits(largestType);
    for (LocalAliasOp alias : aliases) {
      auto aliasType = alias.getResult().getType();
      uint64_t aliasBits = getStorageBits(aliasType);
      if (aliasBits <= largestBits)
        continue;
      largestType = aliasType;
      largestBits = aliasBits;
    }
    largestStorageType[allocation] = largestType;
  }

  OpBuilder builder(module.getContext());
  DenseMap<Operation *, ttg::LocalAllocOp> replacementAllocations;
  for (auto &[allocation, aliases] : aliasClasses) {
    auto baseType = cast<ttg::MemDescType>(allocation->getResult(0).getType());
    ttg::MemDescType largestType = largestStorageType.lookup(allocation);
    if (largestType == baseType)
      continue;

    builder.setInsertionPoint(allocation);
    auto replacement = ttg::LocalAllocOp::create(
        builder, allocation->getLoc(), largestType);
    replacementAllocations[allocation] = replacement;
    LDBG("backing=" << allocation->getResult(0)
                     << " old-bits=" << getStorageBits(baseType)
                     << " new-bits=" << getStorageBits(largestType));
  }

  for (auto &[originalAllocation, aliases] : aliasClasses) {
    Operation *backing = originalAllocation;
    if (auto replacement = replacementAllocations.lookup(originalAllocation)) {
      builder.setInsertionPoint(originalAllocation);
      auto originalType =
          cast<ttg::MemDescType>(originalAllocation->getResult(0).getType());
      auto originalView = ttg::MemDescReinterpretOp::create(
          builder, originalAllocation->getLoc(), originalType,
          replacement.getResult());
      originalAllocation->getResult(0).replaceAllUsesWith(
          originalView.getResult());
      originalAllocation->erase();
      backing = replacement;
    }

    for (LocalAliasOp alias : aliases) {
      builder.setInsertionPoint(alias);
      auto aliasView = ttg::MemDescReinterpretOp::create(
          builder, backing->getLoc(), alias.getResult().getType(),
          backing->getResult(0));
      LDBG("backing=" << backing->getResult(0)
                       << " alias=" << alias.getResult());
      alias.getResult().replaceAllUsesWith(aliasView.getResult());
      alias.erase();
    }
  }

  return success();
}

class GluonRewriteLocalAliasPass
    : public impl::GluonRewriteLocalAliasPassBase<
          GluonRewriteLocalAliasPass> {
public:
  using BaseT =
      impl::GluonRewriteLocalAliasPassBase<GluonRewriteLocalAliasPass>;
  using BaseT::BaseT;

  void runOnOperation() override {
    if (failed(rewriteLocalAliases(getOperation())))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton::gluon

#undef LDBG
#undef DEBUG_TYPE
