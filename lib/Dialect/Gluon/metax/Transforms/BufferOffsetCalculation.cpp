#include "triton/Dialect/Gluon/metax/IR/StorageAlias.h"

#include <functional>
#include <tuple>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gluon-buffer-offset-calculation"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[alias] " << X << "\n")

using namespace mlir;

namespace mlir::triton::gluon {

using StorageAliasOffsetMap =
    DenseMap<Value, std::tuple<int64_t, int64_t, int64_t>>;

static LogicalResult collectOffsets(Value element, int64_t currentOffset,
                                    int64_t bytesBetweenBufferGroups,
                                    int64_t alignment,
                                    int64_t currentGroupSize,
                                    StorageAliasOffsetMap &offsetMap) {
  if (element.getDefiningOp<StorageAliasLocalAllocOp>()) {
    offsetMap[element] = std::make_tuple(
        currentOffset, bytesBetweenBufferGroups, currentGroupSize);
    LDBG("allocation=" << element << " offset=" << currentOffset
                        << " group-stride=" << bytesBetweenBufferGroups
                        << " group-size=" << currentGroupSize);
    return success();
  }

  auto group = element.getDefiningOp<ReuseGroupOp>();
  if (!group) {
    emitError(element.getLoc(),
              "unexpected value in storage-alias overlap tree");
    return failure();
  }

  int64_t groupSize = group.getGroupSize();
  if (group.getGroupKind() == ReuseGroupKind::shared) {
    int64_t childStride = bytesBetweenBufferGroups / groupSize;
    for (Value child : group.getElements()) {
      if (failed(collectOffsets(child, currentOffset, childStride, alignment,
                                currentGroupSize * groupSize, offsetMap)))
        return failure();
    }
    return success();
  }

  int64_t runningOffset = currentOffset;
  for (Value child : group.getElements()) {
    runningOffset = alignStorageAliasSize(runningOffset, alignment);
    if (failed(collectOffsets(child, runningOffset, bytesBetweenBufferGroups,
                              alignment, currentGroupSize, offsetMap)))
      return failure();
    runningOffset += getStorageAliasElementSize(child, alignment);
  }
  int64_t requiredBytes = runningOffset - currentOffset;
  if (requiredBytes > bytesBetweenBufferGroups)
    return group.emitError("not enough space for distinct allocations: need ")
           << requiredBytes << " bytes, have " << bytesBetweenBufferGroups
           << " bytes";
  return success();
}

static void cleanupReuseGroups(ModuleOp module) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<ReuseGroupOp> deadGroups;
    module.walk([&](ReuseGroupOp group) {
      if (group.getResult().use_empty()) {
        deadGroups.push_back(group);
        changed = true;
      }
    });
    for (ReuseGroupOp group : deadGroups)
      group.erase();
  }
}

LogicalResult processBufferOverlapOps(ModuleOp module,
                                      StorageAliasOffsetMap &offsetMap) {
  SmallVector<SetBufferOverlapOp> overlapOps;
  module.walk([&](SetBufferOverlapOp overlap) {
    overlapOps.push_back(overlap);
  });

  DenseSet<Value> processedSpecs;
  for (SetBufferOverlapOp overlap : overlapOps) {
    Value spec = overlap.getStorageAliasSpec();
    if (!processedSpecs.insert(spec).second)
      return overlap.emitError(
          "storage_alias_spec already has a set_buffer_overlap; each spec "
          "may have only one overlap definition");

    Value root = overlap.getOverlapDef();
    int64_t numBuffers = 1;
    std::function<bool(Value)> findNumBuffers = [&](Value element) {
      if (auto alloc = element.getDefiningOp<StorageAliasLocalAllocOp>()) {
        numBuffers = alloc.getResult().getType().getShape()[0];
        return true;
      }
      if (auto group = element.getDefiningOp<ReuseGroupOp>())
        return llvm::any_of(group.getElements(), findNumBuffers);
      return false;
    };
    if (!findNumBuffers(root))
      return overlap.emitError(
          "could not find storage_alias_local_alloc in overlap definition");

    int64_t alignment = getStorageAliasElementAlignment(root);
    int64_t bytesPerBufferGroup = alignStorageAliasSize(
        getStorageAliasElementSize(root, alignment), alignment);
    if (failed(collectOffsets(root, /*currentOffset=*/0,
                              bytesPerBufferGroup, alignment,
                              /*currentGroupSize=*/1, offsetMap)))
      return failure();

    LDBG("overlap=" << overlap << " bytes-per-buffer-group="
                     << bytesPerBufferGroup << " alignment=" << alignment
                     << " num-buffers=" << numBuffers);
    overlap.erase();
  }

  cleanupReuseGroups(module);
  return success();
}

} // namespace mlir::triton::gluon

#undef LDBG
#undef DEBUG_TYPE
