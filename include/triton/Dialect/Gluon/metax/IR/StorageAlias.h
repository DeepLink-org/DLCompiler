#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "triton/Dialect/Gluon/IR/Dialect.h"

namespace mlir::triton::gluon {

inline int64_t getElementBytes(Type elementType) {
  int64_t elementBits = isa<triton::PointerType>(elementType)
                            ? 64
                            : elementType.getIntOrFloatBitWidth();
  return (elementBits + 7) / 8;
}

inline int64_t getAllocationSizePerBuffer(gpu::MemDescType memDescType) {
  return memDescType.getNumElements() *
         getElementBytes(memDescType.getElementType()) /
         memDescType.getShape().front();
}

// Keep TLX's conservative 128-byte SMEM alignment. This is an allocation
// contract, not a claim about bank-conflict freedom or access width.
constexpr int64_t kStorageAliasSmemAlignment = 128;

inline int64_t alignStorageAliasSize(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

inline int64_t getStorageAliasAllocAlignment(gpu::MemDescType memDescType) {
  return std::max(kStorageAliasSmemAlignment,
                  getElementBytes(memDescType.getElementType()));
}

inline int64_t getStorageAliasElementAlignment(Value element) {
  if (auto alloc = element.getDefiningOp<StorageAliasLocalAllocOp>())
    return getStorageAliasAllocAlignment(alloc.getResult().getType());

  auto group = element.getDefiningOp<ReuseGroupOp>();
  assert(group && "unexpected storage-alias reuse-group element");
  int64_t alignment = 1;
  for (Value child : group.getElements())
    alignment = std::max(alignment,
                         getStorageAliasElementAlignment(child));
  return alignment;
}

inline int64_t getStorageAliasElementSize(Value element, int64_t alignment) {
  if (auto alloc = element.getDefiningOp<StorageAliasLocalAllocOp>())
    return getAllocationSizePerBuffer(alloc.getResult().getType());

  auto group = element.getDefiningOp<ReuseGroupOp>();
  assert(group && "unexpected storage-alias reuse-group element");
  if (group.getGroupKind() == ReuseGroupKind::shared) {
    int64_t maxChildSize = 0;
    for (Value child : group.getElements())
      maxChildSize = std::max(
          maxChildSize, getStorageAliasElementSize(child, alignment));
    return maxChildSize * group.getGroupSize();
  }

  int64_t totalSize = 0;
  for (Value child : group.getElements()) {
    totalSize = alignStorageAliasSize(totalSize, alignment);
    totalSize += getStorageAliasElementSize(child, alignment);
  }
  return totalSize;
}

} // namespace mlir::triton::gluon
