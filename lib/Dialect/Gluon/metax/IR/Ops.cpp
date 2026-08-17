#include "triton/Dialect/Gluon/IR/Dialect.h"

#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace mlir::triton::gluon {

OpFoldResult RequireLayoutOp::fold(FoldAdaptor) {
  if (getSrc().getType() == getType())
    return getSrc();
  return {};
}

LogicalResult StorageAliasSpecOp::verify() {
  if (getStorage() != StorageKind::smem)
    return emitOpError("only smem storage aliases are supported");
  if (auto size = getBufferSizeBytesAttr(); size && size.getInt() <= 0)
    return emitOpError("buffer_size_bytes must be positive, got ")
           << size.getInt();
  return success();
}

LogicalResult StorageAliasLocalAllocOp::verify() {
  auto aliasType = getStorageAlias().getType();
  if (aliasType.getStorage() != StorageKind::smem)
    return emitOpError("only smem storage aliases are supported");
  if (!isa<triton::gpu::SharedMemorySpaceAttr>(
          getResult().getType().getMemorySpace()))
    return emitOpError(
        "storage_alias_spec has smem storage but result is not in shared "
        "memory");
  return success();
}

LogicalResult ReuseGroupOp::verify() {
  if (getElements().empty())
    return emitOpError("reuse_group requires at least one element");
  if (getGroupSize() < 1)
    return emitOpError("group_size must be a positive integer, got ")
           << getGroupSize();
  if (getType().getGroupKind() != getGroupKind())
    return emitOpError("group_kind attribute must match the result type");
  return success();
}

static LogicalResult collectReuseGroupLeaves(
    Value value, SmallVectorImpl<Value> &leaves, Operation *context) {
  if (auto group = value.getDefiningOp<ReuseGroupOp>()) {
    for (Value element : group.getElements())
      if (failed(collectReuseGroupLeaves(element, leaves, context)))
        return failure();
    return success();
  }
  if (!isa<triton::gpu::MemDescType>(value.getType()))
    return context->emitOpError(
        "reuse_group values must be defined by gluon.reuse_group");
  leaves.push_back(value);
  return success();
}

LogicalResult SetBufferOverlapOp::verify() {
  if (!getOverlapDef().getDefiningOp<ReuseGroupOp>())
    return emitOpError(
        "overlap_def must be defined by a gluon.reuse_group op");

  SmallVector<Value> leaves;
  if (failed(collectReuseGroupLeaves(getOverlapDef(), leaves,
                                     getOperation())))
    return failure();
  if (leaves.empty())
    return emitOpError("reuse_group tree must contain at least one allocation");

  DenseSet<Value> seen;
  for (Value leaf : leaves) {
    if (!seen.insert(leaf).second)
      return emitOpError("reuse_group tree contains duplicate allocations");
    auto allocation = leaf.getDefiningOp<StorageAliasLocalAllocOp>();
    if (!allocation)
      return emitOpError(
          "reuse_group leaves must be produced by "
          "gluon.storage_alias_local_alloc");
    if (allocation.getStorageAlias() != getStorageAliasSpec())
      return emitOpError(
          "all reuse_group leaves must reference this storage_alias_spec");
  }
  return success();
}

LogicalResult LocalAliasOp::verify() {
  // MemDescType verifies the encoding/memory-space pair, and the ODS trait
  // verifies that source and result use the same memory space. Phase one only
  // supports Shared storage aliases; ordinary reinterpretation remains a
  // native ttg.memdesc_reinterpret operation.
  if (!isa<triton::gpu::SharedMemorySpaceAttr>(
          getSrc().getType().getMemorySpace()))
    return emitOpError("only Shared-memory aliases are supported");
  return success();
}

static LogicalResult verifySliceBounds(Operation *op,
                                       ArrayRef<int64_t> sourceShape,
                                       ArrayRef<int64_t> sliceShape,
                                       ArrayRef<int64_t> offsets,
                                       StringRef sliceName) {
  if (sourceShape.size() != sliceShape.size())
    return op->emitOpError()
           << sliceName << " rank " << sliceShape.size()
           << " must match source rank " << sourceShape.size();
  if (offsets.size() != sourceShape.size())
    return op->emitOpError()
           << "offset count " << offsets.size() << " must match source rank "
           << sourceShape.size();

  for (size_t dim = 0; dim < sourceShape.size(); ++dim) {
    int64_t sourceExtent = sourceShape[dim];
    int64_t sliceExtent = sliceShape[dim];
    int64_t offset = offsets[dim];
    if (ShapedType::isDynamic(sourceExtent) || sourceExtent <= 0)
      return op->emitOpError()
             << "source extent at dimension " << dim
             << " must be a positive static value, got " << sourceExtent;
    if (ShapedType::isDynamic(sliceExtent) || sliceExtent <= 0)
      return op->emitOpError()
             << sliceName << " extent at dimension " << dim
             << " must be a positive static value, got " << sliceExtent;
    if (offset < 0)
      return op->emitOpError()
             << "offset at dimension " << dim
             << " must be non-negative, got " << offset;
    if (sliceExtent > sourceExtent || offset > sourceExtent - sliceExtent)
      return op->emitOpError()
             << sliceName << " at dimension " << dim << " is out of bounds: "
             << "offset " << offset << " + extent " << sliceExtent
             << " exceeds source extent " << sourceExtent;
  }
  return success();
}

LogicalResult ExtractSliceOp::verify() {
  auto sourceType = cast<RankedTensorType>(getSource().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());
  if (sourceType.getElementType() != resultType.getElementType())
    return emitOpError()
           << "result element type " << resultType.getElementType()
           << " must match source element type "
           << sourceType.getElementType();
  return verifySliceBounds(getOperation(), sourceType.getShape(),
                           resultType.getShape(), getOffsets(), "result");
}

LogicalResult InsertSliceOp::verify() {
  auto baseType = cast<RankedTensorType>(getBase().getType());
  auto updateType = cast<RankedTensorType>(getUpdate().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());
  if (baseType.getShape() != resultType.getShape())
    return emitOpError()
           << "result shape " << resultType.getShape()
           << " must match base shape " << baseType.getShape();
  if (baseType.getElementType() != resultType.getElementType())
    return emitOpError()
           << "result element type " << resultType.getElementType()
           << " must match base element type " << baseType.getElementType();
  if (baseType.getElementType() != updateType.getElementType())
    return emitOpError()
           << "update element type " << updateType.getElementType()
           << " must match base element type " << baseType.getElementType();
  return verifySliceBounds(getOperation(), baseType.getShape(),
                           updateType.getShape(), getOffsets(), "update");
}

} // namespace mlir::triton::gluon
