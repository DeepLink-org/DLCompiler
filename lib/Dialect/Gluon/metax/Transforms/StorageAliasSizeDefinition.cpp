#include "triton/Dialect/Gluon/metax/IR/StorageAlias.h"

#include <algorithm>

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gluon-storage-alias-size-definition"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[alias] " << X << "\n")

using namespace mlir;
namespace ttg = ::mlir::triton::gpu;

namespace mlir::triton::gluon {

LogicalResult computeOrValidateStorageAliasSizes(ModuleOp module) {
  DenseMap<Value, SmallVector<StorageAliasLocalAllocOp, 4>> specUsers;
  module.walk([&](StorageAliasLocalAllocOp alloc) {
    specUsers[alloc.getStorageAlias()].push_back(alloc);
  });

  bool hasError = false;
  module.walk([&](StorageAliasSpecOp spec) {
    ArrayRef<StorageAliasLocalAllocOp> users = specUsers[spec.getResult()];
    if (users.empty()) {
      spec.emitWarning(
          "storage_alias_spec has no referencing "
          "gluon.storage_alias_local_alloc operations");
      return;
    }

    SetBufferOverlapOp overlap;
    for (Operation *user : spec.getResult().getUsers()) {
      auto candidate = dyn_cast<SetBufferOverlapOp>(user);
      if (!candidate)
        continue;
      overlap = candidate;
      break;
    }

    int64_t requiredBytes = 0;
    if (overlap) {
      Value root = overlap.getOverlapDef();
      int64_t alignment = getStorageAliasElementAlignment(root);
      int64_t bytesPerBuffer = alignStorageAliasSize(
          getStorageAliasElementSize(root, alignment), alignment);
      StorageAliasLocalAllocOp firstUser = users.front();
      int64_t numBuffers = firstUser.getResult().getType().getShape()[0];
      requiredBytes = bytesPerBuffer * numBuffers;
      LDBG("overlap-root=" << root << " bytes-per-buffer=" << bytesPerBuffer
                            << " num-buffers=" << numBuffers
                            << " required-bytes=" << requiredBytes);
    } else {
      for (StorageAliasLocalAllocOp alloc : users) {
        ttg::MemDescType type = alloc.getResult().getType();
        int64_t bytes =
            type.getNumElements() * getElementBytes(type.getElementType());
        requiredBytes = std::max(requiredBytes, bytes);
        LDBG("allocation=" << alloc.getResult() << " bytes=" << bytes);
      }
    }

    int64_t backingBytes = requiredBytes;
    if (IntegerAttr explicitSize = spec.getBufferSizeBytesAttr()) {
      backingBytes = explicitSize.getInt();
      if (backingBytes < requiredBytes) {
        spec.emitError("storage_alias_spec buffer_size_bytes ")
            << backingBytes << " is too small; requires at least "
            << requiredBytes << " bytes";
        hasError = true;
        return;
      }
    } else {
      spec.setBufferSizeBytesAttr(
          IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                           backingBytes));
    }

    spec.setBufferShapeAttr(
        DenseI64ArrayAttr::get(module.getContext(), {backingBytes}));
    LDBG("spec=" << spec.getResult() << " backing-bytes=" << backingBytes);
  });

  return hasError ? failure() : success();
}

} // namespace mlir::triton::gluon

#undef LDBG
#undef DEBUG_TYPE
