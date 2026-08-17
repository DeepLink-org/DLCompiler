#include "triton/Dialect/Gluon/IR/Dialect.h"

#include <tuple>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gluon-storage-alias-allocation"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[alias] " << X << "\n")

using namespace mlir;
namespace ttg = ::mlir::triton::gpu;

namespace mlir::triton::gluon {

using StorageAliasOffsetMap =
    DenseMap<Value, std::tuple<int64_t, int64_t, int64_t>>;

static void updateCapturedBlockArgumentTypes(Value value) {
  Type type = value.getType();
  for (OpOperand &use : value.getUses()) {
    auto partitions =
        dyn_cast<ttg::WarpSpecializePartitionsOp>(use.getOwner());
    if (!partitions)
      continue;
    unsigned index = use.getOperandNumber();
    for (Region &partition : partitions.getPartitionRegions()) {
      if (index < partition.getNumArguments())
        partition.getArgument(index).setType(type);
    }
  }
}

static void collectMemDescIndexOps(
    Value memDesc, SmallVectorImpl<ttg::MemDescIndexOp> &indexOps) {
  for (OpOperand &use : memDesc.getUses()) {
    Operation *user = use.getOwner();
    TypeSwitch<Operation *>(user)
        .Case<ttg::MemDescIndexOp>(
            [&](ttg::MemDescIndexOp index) { indexOps.push_back(index); })
        .Case<ttg::MemDescReinterpretOp>([&](auto reinterpret) {
          collectMemDescIndexOps(reinterpret.getResult(), indexOps);
        })
        .Case<LocalAliasOp>([&](auto alias) {
          collectMemDescIndexOps(alias.getResult(), indexOps);
        })
        .Case<ttg::WarpSpecializePartitionsOp>([&](auto partitions) {
          unsigned index = use.getOperandNumber();
          for (Region &partition : partitions.getPartitionRegions()) {
            if (index < partition.getNumArguments())
              collectMemDescIndexOps(partition.getArgument(index), indexOps);
          }
        });
  }
}

static FailureOr<Value> createSharedBacking(StorageAliasSpecOp spec,
                                            OpBuilder &builder) {
  DenseI64ArrayAttr shapeAttr = spec.getBufferShapeAttr();
  if (!shapeAttr) {
    spec.emitError(
        "storage_alias_spec has no shape; storage-alias size definition must "
        "run before allocation materialization");
    return failure();
  }

  ArrayRef<int64_t> shape = shapeAttr.asArrayRef();
  if (shape.size() != 1) {
    spec.emitError(
        "smem storage_alias_spec must have one-dimensional byte backing");
    return failure();
  }

  MLIRContext *context = builder.getContext();
  auto ctaLayout =
      ttg::CTAEncodingAttr::fromSplitParams(context, {1}, {1}, {0});
  auto sharedLayout = ttg::SwizzledSharedEncodingAttr::get(
      context, /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1, /*order=*/{0},
      ctaLayout);
  auto backingType = ttg::MemDescType::get(
      shape, builder.getI8Type(), sharedLayout,
      ttg::SharedMemorySpaceAttr::get(context), /*mutableMemory=*/true);
  return ttg::LocalAllocOp::create(builder, spec.getLoc(), backingType)
      .getResult();
}

static int64_t getBufferUnitSize(ttg::MemDescType memDescType) {
  int64_t bufferElements = 1;
  ArrayRef<int64_t> shape = memDescType.getShape();
  for (int64_t extent : shape.drop_front())
    bufferElements *= extent;
  return bufferElements * memDescType.getElementTypeBitWidth() / 8;
}

static LogicalResult rewriteIndexUsers(
    LocalAliasOp alias, ttg::MemDescType originalType,
    const std::tuple<int64_t, int64_t, int64_t> &offsetInfo,
    OpBuilder &builder) {
  auto [bufferOffset, bytesBetweenBufferGroups, groupSize] = offsetInfo;
  int64_t originalBufferBytes = getBufferUnitSize(originalType);
  int64_t scaleFactor = bytesBetweenBufferGroups / originalBufferBytes;
  int64_t offsetSlots = bufferOffset / originalBufferBytes;
  if (scaleFactor <= 1 && offsetSlots == 0)
    return success();

  SmallVector<ttg::MemDescIndexOp> indexOps;
  collectMemDescIndexOps(alias.getResult(), indexOps);
  for (ttg::MemDescIndexOp indexOp : indexOps) {
    builder.setInsertionPoint(indexOp);
    Location loc = indexOp.getLoc();
    Value originalIndex = indexOp.getIndex();
    auto constant = [&](int64_t value) {
      return arith::ConstantOp::create(
          builder, loc, builder.getI32IntegerAttr(value));
    };
    Value newIndex = originalIndex;
    if (scaleFactor > 1)
      newIndex = arith::MulIOp::create(builder, loc, originalIndex,
                                      constant(scaleFactor));
    if (offsetSlots > 0)
      newIndex = arith::AddIOp::create(builder, loc, newIndex,
                                      constant(offsetSlots));
    if (groupSize > 1) {
      Value groupIndex = arith::RemSIOp::create(
          builder, loc, originalIndex, constant(groupSize));
      newIndex = arith::AddIOp::create(builder, loc, newIndex, groupIndex);
    }
    indexOp.getIndexMutable().assign(newIndex);
    LDBG("rewrote memdesc_index=" << indexOp
                                   << " scale=" << scaleFactor
                                   << " offset-slots=" << offsetSlots
                                   << " group-size=" << groupSize);
  }
  return success();
}

LogicalResult materializeStorageAliasAllocations(
    ModuleOp module, const StorageAliasOffsetMap &offsetMap) {
  SmallVector<StorageAliasSpecOp> specs;
  SmallVector<StorageAliasLocalAllocOp> allocations;
  module.walk([&](StorageAliasSpecOp spec) { specs.push_back(spec); });
  module.walk(
      [&](StorageAliasLocalAllocOp alloc) { allocations.push_back(alloc); });

  OpBuilder builder(module.getContext());
  DenseMap<Value, Value> backingForSpec;
  for (StorageAliasSpecOp spec : specs) {
    builder.setInsertionPoint(spec);
    FailureOr<Value> backing = createSharedBacking(spec, builder);
    if (failed(backing))
      return failure();
    backingForSpec[spec.getResult()] = *backing;
  }

  for (StorageAliasLocalAllocOp alloc : allocations) {
    Value backing = backingForSpec.lookup(alloc.getStorageAlias());
    if (!backing)
      return alloc.emitError(
          "storage_alias_spec has no materialized Shared backing");

    ttg::MemDescType originalType = alloc.getResult().getType();
    ttg::MemDescType aliasType = originalType;
    auto offset = offsetMap.find(alloc.getResult());
    if (offset != offsetMap.end()) {
      auto [bufferOffset, bytesBetweenBufferGroups, groupSize] = offset->second;
      int64_t originalBufferBytes = getBufferUnitSize(originalType);
      if (bytesBetweenBufferGroups % originalBufferBytes != 0)
        return alloc.emitError("bytes_between_buffer_groups (")
               << bytesBetweenBufferGroups
               << ") must be a multiple of the original buffer size ("
               << originalBufferBytes << ")";
      if (bufferOffset % originalBufferBytes != 0)
        return alloc.emitError("buffer_offset (")
               << bufferOffset
               << ") must be a multiple of the original buffer size ("
               << originalBufferBytes << ")";

      int64_t scaleFactor = bytesBetweenBufferGroups / originalBufferBytes;
      int64_t offsetSlots = bufferOffset / originalBufferBytes;
      if (scaleFactor > 1 || offsetSlots > 0) {
        ArrayRef<int64_t> shape = originalType.getShape();
        int64_t lastIndex = shape.front() - 1;
        int64_t expandedBufferCount =
            scaleFactor * lastIndex + offsetSlots +
            (lastIndex % groupSize) + 1;
        SmallVector<int64_t> expandedShape(shape.begin(), shape.end());
        expandedShape.front() = expandedBufferCount;
        aliasType = ttg::MemDescType::get(
            expandedShape, originalType.getElementType(),
            originalType.getEncoding(), originalType.getMemorySpace(),
            originalType.getMutableMemory());
      }
    }

    builder.setInsertionPoint(alloc);
    auto alias =
        LocalAliasOp::create(builder, alloc.getLoc(), aliasType, backing);
    alloc.getResult().replaceAllUsesWith(alias.getResult());
    alloc.erase();
    if (aliasType != originalType)
      updateCapturedBlockArgumentTypes(alias.getResult());
    if (offset != offsetMap.end() &&
        failed(rewriteIndexUsers(alias, originalType, offset->second, builder)))
      return failure();
    LDBG("backing=" << backing << " alias=" << alias.getResult()
                     << " original-type=" << originalType
                     << " materialized-type=" << aliasType);
  }

  for (StorageAliasSpecOp spec : specs) {
    if (!spec.getResult().use_empty())
      return spec.emitError(
          "storage_alias_spec still has uses after allocation "
          "materialization");
    spec.erase();
  }
  return success();
}

} // namespace mlir::triton::gluon

#undef LDBG
#undef DEBUG_TYPE
