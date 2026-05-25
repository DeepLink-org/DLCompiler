

#include "dicp/TritonToLinalg/DescriptorConverter.h"
#include "dicp/TritonToLinalg/BlockPtrAnalysis.h"
#include "dicp/TritonToLinalg/MaskAnalysis.h"
#include "dicp/TritonToLinalg/TritonOpConverter.h"
#include "dicp/TritonToLinalg/TritonToLinalgPass.h"
#include "dicp/Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"

namespace DescriptorConverter {
using namespace mlir;
using namespace triton;

bool hasATensorDescriptorType(mlir::TypeRange types) {
  return llvm::any_of(types, [](mlir::Type t) {
    return llvm::isa<mlir::triton::TensorDescType>(t);
  });
}

Descriptor unpackDescriptor(TensorDescType type, Value desc,
                            ConversionPatternRewriter &rewriter) {
  auto makeDescOp = desc.getDefiningOp<triton::MakeTensorDescOp>();
  assert(makeDescOp && "Descriptor must be defined by MakeTensorDescOp");

  Descriptor res;

  res.base = makeDescOp.getBase();
  for (auto s : makeDescOp.getShape()) {
    res.shape.push_back(rewriter.createOrFold<arith::ExtSIOp>(
        makeDescOp.getLoc(), rewriter.getI64Type(), s));
  }
  for (auto st : makeDescOp.getStrides()) {
    res.strides.push_back(rewriter.createOrFold<arith::ExtSIOp>(
        makeDescOp.getLoc(), rewriter.getI64Type(), st));
  }
  res.padding = makeDescOp.getPaddingAttr();

  return res;
}

SmallVector<int32_t> computeOrder(ArrayRef<int64_t> shape) {
  SmallVector<int32_t> order;
  int rank = shape.size();
  order.reserve(rank);
  // default by [dims - 1, ..., 0]
  for (int i = rank - 1; i >= 0; --i) {
    order.push_back(i);
  }
  return order;
}

DenseI32ArrayAttr getFullBoundaryCheckAttr(ConversionPatternRewriter &rewriter,
                                           ArrayRef<int64_t> shape) {
  SmallVector<int32_t> boundaryCheck;
  boundaryCheck.reserve(shape.size());
  for (int32_t dim = 0; dim < static_cast<int32_t>(shape.size()); ++dim) {
    boundaryCheck.push_back(dim);
  }
  return rewriter.getDenseI32ArrayAttr(boundaryCheck);
}

LogicalResult DescriptorLoadConverter::matchAndRewrite(
    triton::DescriptorLoadOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  const auto blockShape = op.getDesc().getType().getBlockType().getShape();
  auto descTy = op.getDesc().getType();
  auto indices = op.getIndices();

  // 1. unpack descriptor
  auto desc = unpackDescriptor(descTy, adaptor.getDesc(), rewriter);

  // 2. create make_tensor_ptr
  SmallVector<int32_t> tensorShapeValues;
  for (auto dim : blockShape) {
    tensorShapeValues.push_back(static_cast<int32_t>(dim));
  }
  Value tensorPtr =
      rewriter.create<triton::MakeTensorPtrOp>(loc,
                                               desc.base,         // base
                                               desc.shape,        // shape
                                               desc.strides,      // strides
                                               indices,           // offset
                                               tensorShapeValues, // tensorShape
                                               computeOrder(blockShape) // order
      );
  // 3. replace tt.load
  auto boundaryCheck = getFullBoundaryCheckAttr(rewriter, blockShape);
  triton::PaddingOptionAttr padding = desc.padding;
  auto cache = triton::CacheModifierAttr::get(rewriter.getContext(),
                                              triton::CacheModifier::NONE);
  auto evict = triton::EvictionPolicyAttr::get(rewriter.getContext(),
                                               triton::EvictionPolicy::NORMAL);
  auto isVolatile = rewriter.getBoolAttr(false);

  if (auto a = op->getAttrOfType<triton::CacheModifierAttr>("cache"))
    cache = a;
  if (auto a = op->getAttrOfType<triton::EvictionPolicyAttr>("evict"))
    evict = a;
  if (auto a = op->getAttrOfType<BoolAttr>("isVolatile"))
    isVolatile = a;

  auto newLoad = rewriter.create<triton::LoadOp>(
      loc, descTy.getSignlessBlockType(), tensorPtr,
      Value(), // mask
      Value(), // other
      boundaryCheck, padding, cache, evict, isVolatile);

  rewriter.replaceOp(op, newLoad.getResult());

  return success();
}

LogicalResult DescriptorStoreConverter::matchAndRewrite(
    triton::DescriptorStoreOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  const auto blockShape = op.getDesc().getType().getBlockType().getShape();
  auto descTy = op.getDesc().getType();
  auto indices = op.getIndices();

  // 1. unpack descriptor
  auto desc = unpackDescriptor(descTy, adaptor.getDesc(), rewriter);

  // 2. create make_tensor_ptr
  SmallVector<int32_t> tensorShapeValues;
  for (auto dim : blockShape) {
    tensorShapeValues.push_back(static_cast<int32_t>(dim));
  }
  Value tensorPtr =
      rewriter.create<triton::MakeTensorPtrOp>(loc,
                                               desc.base,         // base
                                               desc.shape,        // shape
                                               desc.strides,      // strides
                                               indices,           // offset
                                               tensorShapeValues, // tensorShape
                                               computeOrder(blockShape) // order
      );

  // 3. replace tt.store
  Value valueToStore = adaptor.getSrc();

  auto maskType = RankedTensorType::get(blockShape, rewriter.getI1Type());
  rewriter.create<arith::ConstantOp>(loc,
                                     DenseElementsAttr::get(maskType, true));
  auto boundaryCheck = getFullBoundaryCheckAttr(rewriter, blockShape);
  auto cacheModifier = triton::CacheModifierAttr::get(
      rewriter.getContext(), triton::CacheModifier::NONE);
  auto evictionPolicy = triton::EvictionPolicyAttr::get(
      rewriter.getContext(), triton::EvictionPolicy::NORMAL);

  auto newStore = rewriter.create<triton::StoreOp>(loc, tensorPtr, valueToStore,
                                                   Value(), // mask
                                                   boundaryCheck, cacheModifier,
                                                   evictionPolicy);

  rewriter.eraseOp(op);
  return success();
}

} // namespace DescriptorConverter
