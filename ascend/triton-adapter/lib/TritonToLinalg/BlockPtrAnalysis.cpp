//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//
#include "TritonToLinalg/BlockPtrAnalysis.h"
#include "Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <set>

#define DEBUG_TYPE "triton-block-ptr-analysis"
namespace mlir {
namespace triton {

// MemAccType selectMaxMemAccTy(const MemAccType &v1, const MemAccType &v2) {
//   return (v1 > v2) ? v1 : v2;
// }

SmallVector<OpFoldResult> &BlockData::getOffsetsRef() { return this->offsets; }

SmallVector<OpFoldResult> &BlockData::getSizesRef() { return this->sizes; }

SmallVector<OpFoldResult> &BlockData::getStridesRef() { return this->strides; }

Value &BlockData::getSourceRef() { return this->source; }

OpFoldResult &BlockData::getScalarRef() { return this->scalar; }

SmallVector<OpFoldResult> BlockData::getOffsets() const {
  return this->offsets;
}

SmallVector<OpFoldResult> BlockData::getSizes() const { return this->sizes; }

SmallVector<OpFoldResult> BlockData::getStrides() const {
  return this->strides;
}

OpFoldResult BlockData::getOffset(int index) const {
  return this->offsets[index];
}

OpFoldResult BlockData::getSize(int index) const { return this->sizes[index]; }

OpFoldResult BlockData::getStride(int index) const {
  return this->strides[index];
}

OpFoldResult BlockData::getScalar() const { return this->scalar; }

Value BlockData::getSource() const { return this->source; }

MemAccType BlockData::getMemAccType() const { return this->memAccTy; };

MemAccType &BlockData::getMemAccTypeRef() { return this->memAccTy; };

bool BlockData::isScalar() const { return !(this->scalar).isNull(); }

bool BlockData::isEmpty() const {
  return !(this->getRank() || this->source || !(this->scalar).isNull());
}

bool BlockData::hasSource() const { return this->source != nullptr; }

void BlockData::removeSource() { this->source = nullptr; };

bool BlockData::hasResElemTy() const { return this->resElemTy != nullptr; }

Type &BlockData::getResElemTyRef() { return this->resElemTy; }

Type BlockData::getResElemTy() const { return this->resElemTy; }

int64_t BlockData::getRank() const {
  assert(offsets.size() == sizes.size() && offsets.size() == strides.size());
  return this->offsets.size();
}

void BlockData::setResElemTy(const Type &Ty) { this->resElemTy = Ty; }

void BlockData::setScalar(const OpFoldResult &scalar) { this->scalar = scalar; }

void BlockData::setSource(const Value &src) { this->source = src; }

void BlockData::setOffsets(const SmallVector<OpFoldResult> &offsets) {
  this->offsets = offsets;
}

void BlockData::setStrides(const SmallVector<OpFoldResult> &strides) {
  this->strides = strides;
}

void BlockData::setSizes(const SmallVector<OpFoldResult> &szs) {
  this->sizes = szs;
}

void BlockData::setMemAccTy(const MemAccType &v) { this->memAccTy = v; }

void BlockData::setMemAccVal(const MemAccVal v) { this->memAccTy.value = v; }

OpFoldResult BlockData::inferBlockOffset(const Location &loc,
                                         OpBuilder &builder) const {
  OpFoldResult retOffset = builder.getIndexAttr(0);
  for (auto ofr : offsets) {
    retOffset = addOpFoldResult(retOffset, ofr, loc, builder);
  }
  return retOffset;
}

MemRefType BlockData::getResultMemrefType(int64_t offset,
                                          ArrayRef<int64_t> resultShape) const {
  SmallVector<int64_t> staticStrides;
  SmallVector<Value> dynamicStrides;
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);

  auto elementType =
      dyn_cast<BaseMemRefType>(this->source.getType()).getElementType();
  auto layout =
      StridedLayoutAttr::get(this->source.getContext(), offset, staticStrides);
  return MemRefType::get(resultShape, elementType, layout);
}

void BlockData::addBlock(BlockData &lBlock, BlockData &rBlock, Location loc,
                         ConversionPatternRewriter &rewriter) {
  assert(this->isEmpty() && lBlock.getRank() == rBlock.getRank());
  // When both left block and right block have source, it is indirect load.
  assert(!(lBlock.hasSource() && rBlock.hasSource()) &&
         "Don't support each BlockData has own base source pointer");
  this->source =
      lBlock.hasSource() ? lBlock.getSourceRef() : rBlock.getSourceRef();

  assert(!(lBlock.hasResElemTy() && rBlock.hasResElemTy()));
  if (lBlock.hasResElemTy()) {
    assert(lBlock.hasSource());
    this->resElemTy = lBlock.getResElemTyRef();
  } else if (rBlock.hasResElemTy()) {
    assert(rBlock.hasSource());
    this->resElemTy = rBlock.getResElemTyRef();
  }

  // Acctually `scalar` should be accumulated into `offset` and `stride` finally
  // In addBlock, just pass `scalar` when:
  // 1. both lhs and rhs have `scalar`
  // 2. otherwise, both lhs and rhs are scalar type with rank 0
  // Except above, original `scalar` has been fused into `offset` under add.
  if (lBlock.isScalar() && rBlock.isScalar()) {
    auto addScalar = addOpFoldResult(lBlock.getScalarRef(),
                                     rBlock.getScalarRef(), loc, rewriter);
    this->scalar = addScalar;
  } else if (lBlock.getRank() == 0) {
    // When both lhs and rhs are scalar type with rank 0, just try passing
    // potential `scalar`
    this->scalar =
        lBlock.isScalar() ? lBlock.getScalarRef() : rBlock.getScalarRef();
  }

  for (const auto &[lOffset, rOffset] :
       llvm::zip(lBlock.getOffsetsRef(), rBlock.getOffsetsRef())) {
    this->offsets.push_back(addOpFoldResult(lOffset, rOffset, loc, rewriter));
  }

  for (const auto &[lStride, rStride] :
       llvm::zip(lBlock.getStridesRef(), rBlock.getStridesRef())) {
    this->strides.push_back(addOpFoldResult(lStride, rStride, loc, rewriter));
  }

  // Both sizes are same implicitly under `add`
  this->sizes = lBlock.getSizesRef();

  this->getMemAccTypeRef().merge(lBlock.getMemAccTypeRef());
  this->getMemAccTypeRef().merge(rBlock.getMemAccTypeRef());
  // this->setMemAccTy(selectMaxMemAccTy(lBlock.getMemAccType(),
  // rBlock.getMemAccType()));
}

void BlockData::mulBlock(BlockData &lBlock, BlockData &rBlock, Location loc,
                         ConversionPatternRewriter &rewriter) {
  assert(this->isEmpty() && lBlock.getRank() == rBlock.getRank());

  assert(!(lBlock.hasSource() && rBlock.hasSource()));

  assert(
      (lBlock.isScalar() ^ rBlock.isScalar()) &&
      "Currently only support one and only one scalar in function mulBlock()");

  BlockData *lb = &lBlock;
  BlockData *rb = &rBlock;
  if (lb->isScalar()) {
    std::swap(lb, rb);
  }

  // In mulBlock, `scalar` will be accumulated into `offset` and `stride`
  OpFoldResult rScalar = rb->getScalarRef();
  for (const auto &lOffset : lb->getOffsetsRef()) {
    this->offsets.push_back(mulOpFoldResult(lOffset, rScalar, loc, rewriter));
  }

  for (const auto &lStride : lb->getStridesRef()) {
    this->strides.push_back(mulOpFoldResult(lStride, rScalar, loc, rewriter));
  }

  this->sizes = lb->getSizesRef();

  this->getMemAccTypeRef().merge(lBlock.getMemAccTypeRef());
  this->getMemAccTypeRef().merge(rBlock.getMemAccTypeRef());
  // this->setMemAccTy(selectMaxMemAccTy(lBlock.getMemAccType(),
  // rBlock.getMemAccType()));
}

void BlockData::divBlock(BlockData &lBlock, BlockData &rBlock, Location loc,
                         ConversionPatternRewriter &rewriter) {
  assert(this->isEmpty() && lBlock.getRank() == rBlock.getRank());

  assert(!(lBlock.hasSource() && rBlock.hasSource()));

  for (const auto &[lOffset, rOffset] :
       llvm::zip(lBlock.getOffsetsRef(), rBlock.getOffsetsRef())) {
    this->offsets.push_back(divOpFoldResult(lOffset, rOffset, loc, rewriter));
  }

  for (const auto &[lStride, rStride] :
       llvm::zip(lBlock.getStridesRef(), rBlock.getStridesRef())) {
    this->strides.push_back(divOpFoldResult(lStride, rStride, loc, rewriter));
  }

  this->sizes = lBlock.getSizesRef();

  this->getMemAccTypeRef().merge(lBlock.getMemAccTypeRef());
  this->getMemAccTypeRef().merge(rBlock.getMemAccTypeRef());
  // this->setMemAccTy(selectMaxMemAccTy(lBlock.getMemAccType(),
  // rBlock.getMemAccType()));
}

memref::ReinterpretCastOp BlockData::createCastOp(ArrayRef<int64_t> resultShape,
                                                  const Location &loc,
                                                  OpBuilder &builder) const {
  OpFoldResult resOffset = this->inferBlockOffset(loc, builder);
  auto resultType = this->getResultMemrefType(
      isa<Attribute>(resOffset) ? getConstantIntValue(resOffset).value()
                                : ShapedType::kDynamic,
      resultShape);

  return builder.create<memref::ReinterpretCastOp>(
      loc, resultType, this->source, resOffset, this->sizes, this->strides);
}

void BlockData::dump() const {
  llvm::outs() << "[INFO][BEG] BlockData info\n";
  llvm::outs() << "offsets has " << offsets.size() << " items\n";
  int cnt = 0;
  for (auto it = offsets.begin(); it != offsets.end(); it++) {
    llvm::outs() << "offsets[" << cnt++ << "] = " << *it << "\n";
  }
  llvm::outs() << "sizes has " << sizes.size() << " items\n";
  cnt = 0;
  for (auto it = sizes.begin(); it != sizes.end(); it++) {
    llvm::outs() << "sizes[" << cnt++ << "] = " << *it << "\n";
  }
  llvm::outs() << "strides has " << strides.size() << " items\n";
  cnt = 0;
  for (auto it = strides.begin(); it != strides.end(); it++) {
    llvm::outs() << "strides[" << cnt++ << "] = " << *it << "\n";
  }
  llvm::outs() << "source = " << source << "\n";
  llvm::outs() << "scalar = " << scalar << "\n";
  llvm::outs() << "resElemTy = " << resElemTy << "\n";
  llvm::outs() << "memAccTy = " << memAccTy.toString() << "\n";
  llvm::outs() << "[INFO][END] BlockData info\n";
}

Value BlockDataParser::getScalarMemRef(Value ptr, Value memref,
                                       const Location &loc,
                                       ConversionPatternRewriter &rewriter) {
  assert(isa<triton::PointerType>(ptr.getType()) && "expect a scalar pointer");
  if (ptr.getDefiningOp<triton::AddPtrOp>()) {
    if (auto castOp = memref.getDefiningOp<memref::ReinterpretCastOp>()) {
      return castOp.getResult();
    } else {
      llvm_unreachable("pointer value is defined by an unexpected op");
    }
  }

  assert(isa<BlockArgument>(ptr) &&
         "pointer should be produced by addptr or block argument");
  BlockData data;
  data.setSource(memref);
  data.getOffsetsRef().push_back(rewriter.getIndexAttr(0));
  data.getSizesRef().push_back(rewriter.getIndexAttr(1));
  data.getStridesRef().push_back(rewriter.getIndexAttr(1));
  auto castOp = data.createCastOp(SmallVector<int64_t>(1, 1), loc, rewriter);
  return castOp.getResult();
}

void BlockDataParser::parse(
    Value operand, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  if (known.find(operand) != known.end()) {
    return data = known.lookup(operand), void();
  }

  if (isa<IntegerType>(operand.getType())) {
    data.setScalar(getOpFoldResultOfLayoutInfo(operand, rewriter));
    return;
  }

  //
  if (isa<triton::PointerType>(operand.getType())) {
    // Just consider two state: ptr<scalar> and ptr<tensor<scalar>>
    auto remappedPtr = rewriter.getRemappedValue(operand);
    assert(remappedPtr);
    if (auto op = operand.getDefiningOp()) {
      if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
        parseAddPtr(addPtrOp, data, loc, rewriter, known);
      } else if (auto bitcastOp = dyn_cast<triton::BitcastOp>(op)) {
        parseBitcast(bitcastOp, data, loc, rewriter, known);
      } else if (auto makeTensorPtrOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
        parseTensorPtr(makeTensorPtrOp, data, loc, rewriter, known);
      } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(op)) {
        // To support
        // ptr_0 = tl.advance(ptr)
        // ptr_1 = tl.advance(ptr_0)
        parseTensorPtr(advanceOp, data, loc, rewriter, known);
      } else {
        llvm_unreachable(
            "Unexpected operand defining operation, a scalar "
            "pointer can only be produced by AddPtrOp or direct block ptr");
      }
    } else {
      data.setSource(remappedPtr);
    }
    return;
  }

  // not a scalar pointer
  if (auto addOp = operand.getDefiningOp<arith::AddIOp>()) {
    parseAdd(addOp, data, loc, rewriter, known);
  } else if (auto mulOp = operand.getDefiningOp<arith::MulIOp>()) {
    parseMul(mulOp, data, loc, rewriter, known);
  } else if (auto addPtrOp = operand.getDefiningOp<triton::AddPtrOp>()) {
    parseAddPtr(addPtrOp, data, loc, rewriter, known);
  } else if (auto constOp = operand.getDefiningOp<arith::ConstantOp>()) {
    parseConstSplat(constOp, data, loc, rewriter, known);
  } else if (auto broadcastOp = operand.getDefiningOp<triton::BroadcastOp>()) {
    parseBroadcast(broadcastOp, data, loc, rewriter, known);
  } else if (auto splatOp = operand.getDefiningOp<triton::SplatOp>()) {
    parseSplat(splatOp, data, loc, rewriter, known);
  } else if (auto expandDimsOp =
                 operand.getDefiningOp<triton::ExpandDimsOp>()) {
    parseExpandDims(expandDimsOp, data, loc, rewriter, known);
  } else if (auto remOp = operand.getDefiningOp<arith::RemSIOp>()) {
    parseRem(remOp, data, loc, rewriter, known);
  } else if (auto bitcastOp = operand.getDefiningOp<triton::BitcastOp>()) {
    parseBitcast(bitcastOp, data, loc, rewriter, known);
  } else if (auto extsiOp = operand.getDefiningOp<arith::ExtSIOp>()) {
    parseExtSI(extsiOp, data, loc, rewriter, known);
  } else if (auto divOp = operand.getDefiningOp<arith::DivSIOp>()) {
    parseDiv(divOp, data, loc, rewriter, known);
  } else if (auto makeRangeOp = operand.getDefiningOp<triton::MakeRangeOp>()) {
    parseMakeRange(makeRangeOp, data, loc, rewriter, known);
  } else if (auto reduceOp = operand.getDefiningOp<triton::ReduceOp>()) {
    parseReduce(reduceOp, data, loc, rewriter, known);
  } else if (auto loadOp = operand.getDefiningOp<triton::LoadOp>()) {
    parseIndirectLoad<triton::LoadOp>(loadOp, data, loc, rewriter, known);
  } else if (auto castOp = operand.getDefiningOp<arith::FPToSIOp>()) {
    parseIndirectLoad<arith::FPToSIOp>(castOp, data, loc, rewriter, known);
  } else {
    operand.dump();
    llvm_unreachable("encountered AddPtrOp produced by unsupported operation");
  }
}

void BlockDataParser::parseAdd(
    arith::AddIOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  BlockData lBlock, rBlock;
  parse(op.getLhs(), lBlock, loc, rewriter, known);
  parse(op.getRhs(), rBlock, loc, rewriter, known);
  data.addBlock(lBlock, rBlock, loc, rewriter);
}

void BlockDataParser::parseMul(
    arith::MulIOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  BlockData lBlock, rBlock;
  parse(op.getLhs(), lBlock, loc, rewriter, known);
  parse(op.getRhs(), rBlock, loc, rewriter, known);

  data.mulBlock(lBlock, rBlock, loc, rewriter);
}

void BlockDataParser::parseDiv(
    arith::DivSIOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  BlockData lBlock, rBlock;
  parse(op.getLhs(), lBlock, loc, rewriter, known);
  parse(op.getRhs(), rBlock, loc, rewriter, known);
  data.divBlock(lBlock, rBlock, loc, rewriter);
}

// TODO : support modulos
void BlockDataParser::parseRem(
    arith::RemSIOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(false && "Address expression with modulo is not supported yet, it "
                  "shall be analysis at linearize.");
}

void BlockDataParser::parseMakeRange(
    triton::MakeRangeOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());
  auto shape = dyn_cast<ShapedType>(op.getType()).getShape();

  auto start = op.getStart();
  auto end = op.getEnd();
  auto stride = (end >= start) && (end - start <= shape[0]);
  assert(stride == 1 &&
         "make_range op should always return a tensor of stride 1");

  data.getOffsetsRef().push_back(rewriter.getIndexAttr(start));
  data.getSizesRef().push_back(rewriter.getIndexAttr(shape[0]));
  data.getStridesRef().push_back(rewriter.getIndexAttr(stride));
}

void BlockDataParser::parseExpandDims(
    triton::ExpandDimsOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());

  parse(op.getSrcMutable().get(), data, loc, rewriter, known);
  auto resShape = dyn_cast<ShapedType>(op.getResult().getType()).getShape();
  auto axis = op.getAxis();

  assert(resShape[axis] == 1 &&
         "The destiny shape of changed dimension should be 1");

  data.getOffsetsRef().insert(data.getOffsetsRef().begin() + axis,
                              rewriter.getIndexAttr(0));
  data.getSizesRef().insert(data.getSizesRef().begin() + axis,
                            rewriter.getIndexAttr(1));
  data.getStridesRef().insert(data.getStridesRef().begin() + axis,
                              rewriter.getIndexAttr(0));
}

void BlockDataParser::parseBitcast(
    triton::BitcastOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());
  parse(op.getSrc(), data, loc, rewriter, known);

  auto resType = op.getResult().getType();
  mlir::Type resElemPointeeTy;
  if (auto resShapedTy = dyn_cast<ShapedType>(resType)) {
    auto resElemTy = resShapedTy.getElementType();
    resElemPointeeTy =
        dyn_cast<triton::PointerType>(resElemTy).getPointeeType();
  } else {
    resElemPointeeTy = dyn_cast<triton::PointerType>(resType).getPointeeType();
  }
  data.setResElemTy(resElemPointeeTy);
}

void BlockDataParser::parseExtSI(
    arith::ExtSIOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());
  parse(op.getIn(), data, loc, rewriter, known);
}

void BlockDataParser::parseBroadcast(
    triton::BroadcastOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());

  auto src = op.getSrcMutable().get();
  auto dst = op.getResult();
  assert(isa<ShapedType>(src.getType()) &&
         "tt.broadcast's input should be a tensor");

  auto srcShape = dyn_cast<ShapedType>(src.getType()).getShape();
  auto dstShape = dyn_cast<ShapedType>(dst.getType()).getShape();
  assert(srcShape.size() == dstShape.size() &&
         "rank of source shoule be equal to destnation");

  parse(src, data, loc, rewriter, known);

  for (const auto &[idx, src_dst] :
       llvm::enumerate(llvm::zip(srcShape, dstShape))) {
    const auto &[srcAxis, dstAxis] = src_dst;
    if (srcAxis == dstAxis) {
      continue;
    }
    assert(srcAxis < dstAxis &&
           "srcShape of broadcastOp must be less than dstShape.");
    data.getSizesRef()[idx] = rewriter.getIndexAttr(dstAxis);
  }
}

void BlockDataParser::parseSplat(
    triton::SplatOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());
  auto src = op.getSrc();
  auto dst = op.getResult();
  auto dstShape = dyn_cast<ShapedType>(dst.getType()).getShape();

  parse(src, data, loc, rewriter, known);

  if (isa<IntegerType>(src.getType()) ||
      isa<triton::PointerType>(src.getType())) {
    if (!data.isEmpty()) {
      data.getOffsetsRef().clear();
      data.getSizesRef().clear();
      data.getStridesRef().clear();
    }
    for (auto dstAxis : dstShape) {
      data.getOffsetsRef().push_back(rewriter.getIndexAttr(0));
      data.getSizesRef().push_back(rewriter.getIndexAttr(dstAxis));
      data.getStridesRef().push_back(rewriter.getIndexAttr(0));
    }
  } else {
    op->emitError("Block data Analysis: unsupported splat pattern");
    return;
  }
  if (data.isScalar()) {
    data.getOffsetsRef()[0] = data.getScalarRef();
  }
}

void BlockDataParser::parseConstSplat(
    arith::ConstantOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());

  DenseElementsAttr denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
  assert(denseAttr && denseAttr.isSplat() &&
         isa<IntegerType>(denseAttr.getElementType()));

  auto innerVal = denseAttr.getValues<IntegerAttr>()[0].getValue();
  auto innerValIndexAttr = rewriter.getIndexAttr(innerVal.getSExtValue());

  // for mul state
  data.setScalar(innerValIndexAttr);

  auto resType = dyn_cast<ShapedType>(op.getResult().getType());
  size_t loopLimit = resType.getShape().size();
  for (auto i = 0; i < loopLimit; i++) {
    // Add original dense val to first dim offset for add state
    if (i == 0) {
      data.getOffsetsRef().push_back(innerValIndexAttr);
    } else {
      data.getOffsetsRef().push_back(rewriter.getIndexAttr(0));
    }
    data.getSizesRef().push_back(rewriter.getIndexAttr(resType.getShape()[i]));
    data.getStridesRef().push_back(rewriter.getIndexAttr(0));
  }
}

template <typename T>
std::enable_if_t<std::is_same_v<T, triton::MakeTensorPtrOp> ||
                 std::is_same_v<T, triton::AdvanceOp>>
BlockDataParser::parseTensorPtr(
    T op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());

  Value remappedValue = rewriter.getRemappedValue(op);
  if (auto castOp = remappedValue.getDefiningOp<memref::ReinterpretCastOp>()) {
    parseReinterpretCast(castOp, data, loc, rewriter, known);
  } else {
    llvm_unreachable("the value should be mapped to memref.reinterpret_cast");
  }
}

void BlockDataParser::parseAddPtr(
    triton::AddPtrOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());

  BlockData ptrBlock, offsetBlock;
  parse(op.getPtr(), ptrBlock, op.getLoc(), rewriter, known);
  parse(op.getOffset(), offsetBlock, op.getLoc(), rewriter, known);

  assert(ptrBlock.hasSource() &&
         "Ptr field should provide source/base pointer");
  // offset has source means offset is from tl.load and other ops(TODO)
  if (offsetBlock.hasSource()) {
    ptrBlock.setMemAccTy(offsetBlock.getMemAccType());
    offsetBlock.removeSource();
  }

  // handle for loop & scalar
  if (ptrBlock.getRank() == 1 && offsetBlock.getRank() == 0) {
    offsetBlock.getSizesRef().push_back(rewriter.getIndexAttr(1));
    offsetBlock.getOffsetsRef().push_back(offsetBlock.getScalarRef());
    offsetBlock.getStridesRef().push_back(rewriter.getIndexAttr(0));
  }

  assert(ptrBlock.getRank() == offsetBlock.getRank() &&
         "ptr and offset should have same rank");
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[parseAddPtr][BEG] =========================\n";
    os << "[parseAddPtr] op is " << op << "\n";
    for (int i = 0; i < ptrBlock.getRank(); i++) {
      os << "ptrBlock.getOffsetsRef()[" << i
         << "] = " << ptrBlock.getOffsetsRef()[i] << "\n";
      os << "ptrBlock.getSizesRef()[" << i
         << "] = " << ptrBlock.getSizesRef()[i] << "\n";
      os << "ptrBlock.getStridesRef()[" << i
         << "] = " << ptrBlock.getStridesRef()[i] << "\n";
      os << "offsetBlock.getOffsetsRef()[" << i
         << "] = " << offsetBlock.getOffsetsRef()[i] << "\n";
      os << "offsetBlock.getSizesRef()[" << i
         << "] = " << offsetBlock.getSizesRef()[i] << "\n";
      os << "offsetBlock.getStridesRef()[" << i
         << "] = " << offsetBlock.getStridesRef()[i] << "\n";
    }
    os << "[parseAddPtr][END] -------------------------\n";
  });
  data.addBlock(ptrBlock, offsetBlock, op.getLoc(), rewriter);
}

void BlockDataParser::parseReinterpretCast(
    memref::ReinterpretCastOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());

  data.setOffsets(op.getMixedOffsets());
  data.setSizes(op.getMixedSizes());
  data.setStrides(op.getMixedStrides());
  data.setSource(op.getSource());

  // In memref::ReinterpretCastOp, offset means the total of collapsing multiple
  // dimensions, which corresponds to first dim offset in block data.
  // Here populate the rest of the dimensions with zeroes.
  assert(data.getOffsetsRef().size() == 1);
  size_t loopLimit = data.getSizesRef().size();
  for (size_t i = 1; i < loopLimit; i++) {
    data.getOffsetsRef().push_back(rewriter.getIndexAttr(0));
  }

  // Block data cannot accept patterns of size 1 and non-zero const stride from
  // memref type. Set stride back to zero if this scenario is detected.
  loopLimit = data.getStridesRef().size();
  for (size_t i = 0; i < loopLimit; i++) {
    auto strideConst = getConstantIntValue(data.getStridesRef()[i]);
    auto sizeConst = getConstantIntValue(data.getSizesRef()[i]);
    assert(sizeConst.has_value());
    if (sizeConst.value() == 1 && strideConst.has_value()) {
      data.getStridesRef()[i] = rewriter.getIndexAttr(0);
    }
  }
}

void BlockDataParser::parseReduce(
    triton::ReduceOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {

  const std::string scenarioMessages =
      "PtsAnalysis supports indirectly block load in the following scenario\n"
      "B = tl.load(Aptr + Aoffset) # B is 1D tensor\n"
      "s = tl.min(B) # s is a scalar\n"
      "D = tl.load(Cptr + s + Coffset) # s is used as the scalar offset\n";

  auto reduce_src = op->getOperand(0);
  BlockData srcBlock;
  parse(reduce_src, srcBlock, loc, rewriter, known);
  if (!srcBlock.hasSource()) {
    llvm_unreachable(scenarioMessages.c_str());
  }
  if (!isa<triton::LoadOp>(srcBlock.getSource().getDefiningOp())) {
    llvm_unreachable(scenarioMessages.c_str());
  }

  auto reduce_result = op->getResult(0);
  auto shaped_ty = dyn_cast<RankedTensorType>(reduce_result.getType());
  auto shape = shaped_ty.getShape();
  auto ops = llvm::map_to_vector(op.getBody()->without_terminator(),
                                 [](Operation &op) { return &op; });
  // Support only the case: scalar = tl.load(1D tensor)
  if (shape.size() != 1 || op.getAxis() != 0 || ops.size() != 1 ||
      !isa<arith::MinSIOp>(ops.front())) {
    llvm_unreachable(scenarioMessages.c_str());
  }

  auto castOp = rewriter.create<arith::IndexCastOp>(
      loc, RankedTensorType::get(shape, rewriter.getIndexType()),
      reduce_result);
  auto offset = castOp.getResult();
  if (data.isEmpty()) {
    data.getOffsetsRef().push_back(offset);
    data.getSizesRef().push_back(rewriter.getIndexAttr(shape[0]));
    data.getStridesRef().push_back(rewriter.getIndexAttr(1));
  } else {
    llvm_unreachable("parseReduce with offset already setup not yet supported");
  }
}

template <typename OpTy>
void parseIndirectLoad(OpTy op, BlockData &data, const Location &loc,
                       ConversionPatternRewriter &rewriter,
                       const llvm::SmallDenseMap<Value, BlockData> &known) {
  // FIXME: assume single result of operation
  auto opRes = op->getResult(0);
  auto opResTy = opRes.getType();
  std::vector<int64_t> resShape;
  if (auto shapedResTy = dyn_cast<ShapedType>(opResTy)) {
    // For now, we consider this is UnstrucMemAcc because we have no other info.
    // Visiting other ops may change the type due to more info.
    data.setMemAccVal(MemAccVal::UnstrucMemAcc);
    resShape = shapedResTy.getShape().vec();
  } else {
    // scalar load means this is used as offset. It is StrucMemAcc.
    data.setMemAccVal(MemAccVal::StrucMemAcc);
    resShape.push_back(1);
  }
  for (auto &s : resShape) {
    data.getOffsetsRef().push_back(rewriter.getIndexAttr(0));
    data.getSizesRef().push_back(rewriter.getIndexAttr(s));
    data.getStridesRef().push_back(rewriter.getIndexAttr(1));
  }
  // set the source in BlockData so that we know an indirect-load op exists in
  // the chain.
  data.setSource(opRes);
}

void BlockDataParser::rewriteAddPtr(
    triton::AddPtrOp op, triton::AddPtrOp::Adaptor &adaptor,
    ConversionPatternRewriter &rewriter,
    llvm::SmallDenseMap<Value, BlockData> &known) {
  auto insertPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(op);

  BlockData data;
  parseAddPtr(op, data, op.getLoc(), rewriter, known);

  if (data.getMemAccTypeRef().isUnstructured()) {
    // TODO: Based on more info, try to create a performant IR
    rewriteAddPtrToUnstrucMemAcc(op, adaptor, rewriter, data);
    LLVM_DEBUG({ llvm::dbgs() << *getModuleOpFromOperation(op) << "\n"; });
    return;
  }

  if (data.getSizesRef().size() == 0) {
    data.getSizesRef().push_back(rewriter.getIndexAttr(1));
    data.getStridesRef().push_back(rewriter.getIndexAttr(0));
    data.getOffsetsRef().push_back(data.getScalarRef());
  }

  ArrayRef<int64_t> resultShape;
  // shape {1,} is stub for single ptr
  SmallVector<int64_t> stubScalarTypeShape(1, 1);
  if (auto shapedType = dyn_cast<ShapedType>(op.getResult().getType())) {
    resultShape = shapedType.getShape();
  } else {
    assert(data.getRank() == 1);
    resultShape = stubScalarTypeShape;
  }

  known[op.getResult()] = data;

  // If there are dimensions with size 1 and stride 0, replace 0 stride with the
  // product of sizes of all lower dimensions. This avoids creating memref with
  // zero stride.
  // And here store the unmodified state into known ptrs, since any following
  // pointer arithmetic operations should still use the original 0 stride.
  auto inferedSize = 1;
  for (int i = data.getSizesRef().size() - 1; i >= 0; i--) {
    auto strideConst = getConstantIntValue(data.getStridesRef()[i]);
    auto sizeConst = getConstantIntValue(data.getSizesRef()[i]);
    assert(sizeConst.has_value());
    if (sizeConst.value() == 1 && strideConst && strideConst.value() == 0) {
      data.getStridesRef()[i] = rewriter.getIndexAttr(inferedSize);
    }
    inferedSize *= sizeConst.value();
  }

  if (data.hasResElemTy()) {
    // Handle bitcast scenario
    auto memrefType = dyn_cast<BaseMemRefType>(data.getSourceRef().getType())
                          .cloneWith(std::nullopt, data.getResElemTyRef());
    UnrealizedConversionCastOp castOp =
        rewriter.create<mlir::UnrealizedConversionCastOp>(
            op.getLoc(), memrefType, data.getSourceRef());
    data.setSource(castOp.getOutputs()[0]);
  }

  // ToDo: need to handle module scenario

  memref::ReinterpretCastOp castOp =
      data.createCastOp(resultShape, op.getLoc(), rewriter);
  Value src = castOp.getResult();
  LLVM_DEBUG({
    llvm::dbgs() << "cast MemRefType:\n";
    castOp.getOperation()->print(llvm::dbgs(),
                                 OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });

  rewriter.replaceOp(op, src);
  rewriter.restoreInsertionPoint(insertPoint);
}

void BlockDataParser::rewriteAdvanceOp(
    triton::AdvanceOp op, ConversionPatternRewriter &rewriter,
    llvm::SmallDenseMap<Value, BlockData> &known) {
  OpBuilder::InsertionGuard insertionGuard(rewriter);
  rewriter.setInsertionPoint(op);
  auto loc = op.getLoc();

  BlockData blockData;
  parse(op.getOperand(0), blockData, loc, rewriter, known);

  // region [BUGFIX] Add the code block below following the same logic as 'BlockDataParser::rewriteAddPtr' function.
  known[op.getResult()] = blockData;
  auto inferedSize = 1;
  for (int i = blockData.getSizesRef().size() - 1; i >= 0; i--) {
    auto strideConst = getConstantIntValue(blockData.getStridesRef()[i]);
    auto sizeConst = getConstantIntValue(blockData.getSizesRef()[i]);
    assert(sizeConst.has_value());
    if (sizeConst.value() == 1 && strideConst && strideConst.value() == 0) {
      blockData.getStridesRef()[i] = rewriter.getIndexAttr(inferedSize);
    }
    inferedSize *= sizeConst.value();
  }
  // endregion

  SmallVector<OpFoldResult> incrementOffsets =
      llvm::map_to_vector(op.getOffsets(), [&](Value offset) {
        return getOpFoldResultOfLayoutInfo(offset, rewriter);
      });

  SmallVector<OpFoldResult> newOffsets;
  for (const auto [increment, originalOffset, stride] :
       llvm::zip(incrementOffsets, blockData.getOffsetsRef(),
                 blockData.getStridesRef())) {
    auto curDimOffset =
        addOpFoldResult(mulOpFoldResult(increment, stride, loc, rewriter),
                        originalOffset, loc, rewriter);

    newOffsets.push_back(curDimOffset);
  }

  blockData.getOffsetsRef().clear();

  for (auto offset : newOffsets)
    blockData.getOffsetsRef().push_back(offset);

  SmallVector<int64_t> scalarShape(1, 1); // Stub shape
  ArrayRef<int64_t> resultShape;
  auto pointerType = cast<mlir::triton::PointerType>(op.getResult().getType());

  if (auto shapedType = dyn_cast<ShapedType>(pointerType.getPointeeType())) {
    resultShape = shapedType.getShape();
  } else {
    // scalar pointer, should produce a one dimensional memref
    resultShape = scalarShape;
    assert(blockData.getRank() == 1);
  }

  auto newOp = blockData.createCastOp(resultShape, loc, rewriter);
  rewriter.replaceOp(op, newOp.getResult());

  known[newOp.getResult()] = blockData;
}

void BlockDataParser::rewriteYieldOp(
    scf::YieldOp op, ConversionPatternRewriter &rewriter,
    const std::set<int> blockArgIdxSet,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  // Any inserted instruction should be before this yield
  OpBuilder::InsertionGuard insertionGuard{rewriter};
  rewriter.setInsertionPoint(op);

  auto adaptor = scf::YieldOp::Adaptor(op);

  SmallVector<BlockData, 5> initArgState;
  SmallVector<Value> operands(adaptor.getOperands());

  // For each of the init arg that we added additional Values in for loop, we
  // need to add corresponding Values as yield operands. The loop below gathers
  // BlockData for those values.
  for (auto [i, v] : llvm::enumerate(adaptor.getOperands())) {
    if (auto mappedV = rewriter.getRemappedValue(v)) {
      // If this value is a tensor of pointers produced by AddPtrOp,
      // we should have already converted to a ReinterpretCastOp without
      // layout information for the normal cases
      if (v.getDefiningOp<triton::AddPtrOp>() ||
          v.getDefiningOp<triton::AdvanceOp>() ||
          v.getDefiningOp<triton::MakeTensorPtrOp>()) {
        if (auto castOp = mappedV.getDefiningOp<memref::ReinterpretCastOp>()) {
          v = castOp;
        } else {
          llvm_unreachable("mapped value defined by an unexpected op");
        }
      } else {
        // If this value is not a tensor of pointers, we will use the
        // mapped value, and rely on the conversion will happen later
        // automatically when we legalize loop body.

        // TODO:
        // The scenario where a value is a tensor of pointers but not
        // produced by AddPtrOp is not supported
        if (isa<TensorType>(mappedV.getType()) &&
            isa<triton::PointerType>(
                dyn_cast<TensorType>(mappedV.getType()).getElementType()))
          llvm_unreachable("unsupported scenario where a value is a tensor of "
                           "pointers but not produced by AddPtrOp");
        v = mappedV;
      }
    }

    if (blockArgIdxSet.find(i) == blockArgIdxSet.end())
      continue;

    auto reintCastOp = v.getDefiningOp<memref::ReinterpretCastOp>();
    assert(
        reintCastOp ||
        (isa<TensorType>(v.getType()) &&
         isa<IndexType>(dyn_cast<TensorType>(v.getType()).getElementType())));

    BlockData state;
    if (reintCastOp) {
      parseReinterpretCast(reintCastOp, state, op.getLoc(), rewriter, known);
    } else {
      parse(v, state, op.getLoc(), rewriter, known);
    }
    initArgState.push_back(state);
  }

  // For each of the BlockData recorded in the last step, extract value
  // that correspond to offset and stride for each dimension and append
  // them to yield operands.
  for (auto state : initArgState) {
    for (auto offset : state.getOffsetsRef()) {
      // offsets can be IntAttr zeroes, since reinterpret_cast collapses
      // them for the input memref, and the for loop may not update
      // offsets other than offsets[0]. Create constants Values for those
      // zeroes.
      if (isa<Attribute>(offset)) {
        auto constOffset = offset.get<Attribute>();
        assert(isa<IntegerAttr>(constOffset) &&
               dyn_cast<IntegerAttr>(constOffset).getInt() == 0 &&
               "attribute offsets should be zeroes");
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(0));
        operands.push_back(constOp.getResult());
      } else {
        operands.push_back(offset.get<Value>());
      }
    }

    for (OpFoldResult stride : state.getStridesRef()) {
      assert(!isa<Attribute>(stride) && "BlockData strides for yield within for"
                                        " loop not expected to be attribute.");
      operands.push_back(stride.get<Value>());
    }
  }

  // Yield is a terminator op that must be at the end of the function
  rewriter.setInsertionPointAfter(op);
  auto newOp = rewriter.replaceOpWithNewOp<scf::YieldOp>(op, operands);
  assert(op->getNumResults() == 0);

  LLVM_DEBUG({
    llvm::dbgs() << "new yield:";
    newOp.getOperation()->print(llvm::dbgs(),
                                OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });
}

void BlockDataParser::rewriteForOp(
    scf::ForOp op, ConversionPatternRewriter &rewriter,
    llvm::SmallDenseMap<Value, BlockData> &known) {

  SmallVector<Value> newInitArgs;

  SmallVector<std::pair<int, BlockData>, 5> initArgIndexIfBlockData;
  SmallVector<std::pair<int, BlockData>, 5> knownPtrsTmp;
  std::set<int> blockArgIdxSet;

  // Create a new list of init args
  for (auto [i, arg] : llvm::enumerate(op.getInitArgs())) {
    auto mappedV = rewriter.getRemappedValue(arg);
    memref::ReinterpretCastOp reintCastOp;

    // If this init arg is supposed to be remapped, use the remapped
    // value instead.
    // In addition, if this init arg is a memref created by a reinterpret_cast
    // or a tensor of index, there is a chance that it will be used in addptr.
    // Create BlockData for each such init arg.
    if (mappedV) {
      // TODO:
      //  Passing a block argument pointer directly into a for loop not
      //  supported.
      assert(!(isa<BlockArgument>(mappedV) &&
               isa<UnrankedMemRefType>(mappedV.getType())) &&
             "cannot take pointer block argument as init arg for for loop");
      if (auto op = mappedV.getDefiningOp<memref::ReinterpretCastOp>()) {
        // Record memref::ReinterpretCastOp
        reintCastOp = op;
        newInitArgs.push_back(mappedV);
      } else {
        newInitArgs.push_back(mappedV);
      }
    } else {
      newInitArgs.push_back(arg);
    }

    auto indexTensor =
        isa<TensorType>(arg.getType()) &&
        isa<IndexType>(dyn_cast<TensorType>(arg.getType()).getElementType());

    // Handle memref::ReinterpretCastOp and tensor<Index> specially
    if (!reintCastOp && !indexTensor)
      continue;

    BlockData data;
    if (reintCastOp) {
      parseReinterpretCast(reintCastOp, data, op.getLoc(), rewriter,
                           llvm::SmallDenseMap<Value, BlockData>(0));
    } else {
      parse(arg, data, op.getLoc(), rewriter,
            llvm::SmallDenseMap<Value, BlockData>(0));
    }

    // Record the BlockData for later processing
    initArgIndexIfBlockData.push_back(std::make_pair(i, data));
  }

  // Set insertion point to be before the for loop for new variables passed
  // into the new loop.
  auto origIp = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(op);

  // For each of the BlockData recorded in the last step, insert new
  // instructions to describe offset and stride for each dimension and append
  // them to init args
  for (auto [i, data] : initArgIndexIfBlockData) {
    // For each dimension, if the corresponding offset and stride is an
    // integer attribute, create a constant value and append them at the
    // end of init arg list, which is prepared for calculate layout info with
    // loop interation index
    for (auto [j, dataOffset] : llvm::enumerate(data.getOffsetsRef())) {

      if (isa<Attribute>(dataOffset)) {
        auto constDataOffset = dataOffset.get<Attribute>();
        assert(isa<IntegerAttr>(constDataOffset));
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(
                             dyn_cast<IntegerAttr>(constDataOffset).getInt()));
        newInitArgs.push_back(constOp.getResult());
        data.getOffsetsRef()[j] = constOp.getResult();
      } else {
        assert(isa<IndexType>(dataOffset.get<Value>().getType()));
        newInitArgs.push_back(dataOffset.get<Value>());
      }
    }

    for (auto [j, dataStride] : llvm::enumerate(data.getStridesRef())) {

      if (isa<Attribute>(dataStride)) {
        auto constDataStride = dataStride.get<Attribute>();
        assert(isa<IntegerAttr>(constDataStride));
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(
                             dyn_cast<IntegerAttr>(constDataStride).getInt()));
        newInitArgs.push_back(constOp.getResult());
        data.getStridesRef()[j] = constOp.getResult();
      } else {
        assert(isa<IndexType>(dataStride.get<Value>().getType()));
        newInitArgs.push_back(dataStride.get<Value>());
      }
    }

    // Note that we want the knownPtrs to be indexed by block arg, but we
    // only have index for now. Also, the blockdata we record is the init
    // arg, but want to to use newly created block arg. These block args
    // are not created yet. We will translate this mapping later.
    knownPtrsTmp.push_back(std::make_pair(i, data));
    blockArgIdxSet.emplace(i);

    // If the original init arg is a memref produced by reinterpret_cast,
    // create a new memref using new strides and offsets created above.
    // This produces a canonicalized memref, which will match what the
    // for loop generates if it modifies the memref. E.g., original
    // reinterpret_cast can produce a memref with const stride:
    //  - memref<4x256xbf16, affine_map<(d0, d1)[s0, s1] -> (d0 * 256 +
    //  s0 + d1
    //  * s1)>>
    // The new reinterpret_cast will always have dynamic stride and
    // offset:
    //  - memref<4x256xbf16, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1
    //  + s0 + d1 * s2)>>
    if (newInitArgs[i].getDefiningOp<memref::ReinterpretCastOp>()) {
      SmallVector<int64_t> resultShape;
      for (auto size : data.getSizesRef()) {
        auto constSize = getConstantIntValue(size);
        assert(constSize && "expected constant size");
        resultShape.push_back(constSize.value());
      }

      // In current block data layout info, strides and offsets must be dynamic
      // value
      auto castOp = data.createCastOp(resultShape, op.getLoc(), rewriter);

      LLVM_DEBUG({
        llvm::dbgs() << "new reinterpret_cast with dynamic sizes "
                        "and offsets:";
        castOp->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
        llvm::dbgs() << "\n";
      });

      newInitArgs[i] = castOp.getResult();
    }
  }

  rewriter.restoreInsertionPoint(origIp);

  // Create a new scf::ForOp that uses updated init args and same loop body
  auto newOp = rewriter.create<scf::ForOp>(
      op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(),
      newInitArgs, [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
        IRMapping mapping;
        mapping.map(op.getInductionVar(), iv);
        mapping.map(op.getInitArgs(), newInitArgs);
        mapping.map(op.getRegionIterArgs(), args);

        for (auto &bodyOp : op.getRegion().getOps()) {
          b.clone(bodyOp, mapping);
        }
      });

  // Convert the book-keeping data structure to use the correct key and value.
  // Key is converted from init arg index to newly created block arg, and
  // Value's BlockData fields are converted from init arg to newly created block
  // arg
  int cnt = op.getRegionIterArgs().size();
  for (auto [i, data] : knownPtrsTmp) {
    for (auto it = data.getOffsetsRef().begin();
         it != data.getOffsetsRef().end(); it++) {
      *it = newOp.getRegionIterArgs()[cnt];
      cnt++;
    }

    for (auto it = data.getStridesRef().begin();
         it != data.getStridesRef().end(); it++) {
      *it = newOp.getRegionIterArgs()[cnt];
      cnt++;
    }

    auto key = newOp.getRegionIterArgs()[i];
    known.insert(std::make_pair(key, data));
  }
  assert(static_cast<size_t>(cnt) == newOp.getRegionIterArgs().size() &&
         "expect to remap all new block args");

  // Replace only the results that correspond to the original scf.for
  auto resultsToReplaceWith = ResultRange(
      newOp.result_begin(), newOp.result_begin() + op.getNumResults());
  rewriter.replaceOp(op, resultsToReplaceWith);

  // Update the loop body. Manually invoke the rewrite logic on addptr and yield
  // in the loop body, so we can take advantage of the states we built up
  for (auto &bodyOp : newOp.getRegion().getOps()) {
    if (auto addptrOp = dyn_cast<triton::AddPtrOp>(bodyOp)) {
      // FIXME: Constructed adaptor here does not hold the transformed op info.
      auto adaptor = triton::AddPtrOp::Adaptor(addptrOp);
      rewriteAddPtr(addptrOp, adaptor, rewriter, known);
    } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(bodyOp)) {
      rewriteAdvanceOp(advanceOp, rewriter, known);
    } else if (auto forOp = dyn_cast<scf::ForOp>(bodyOp)) {
      // TODO:
      //  Nested for loops are not supported at the moment
      assert(0 && "nested loops currently not supported");
    }
  }

  if (op.getNumRegionIterArgs()) {
    auto yieldOp = cast<scf::YieldOp>(newOp.getBody()->getTerminator());
    rewriteYieldOp(yieldOp, rewriter, blockArgIdxSet, known);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "new for\n";
    newOp.getOperation()->print(llvm::dbgs(),
                                OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });
}

/// @brief Rewrite the triton::AddPtrOp to handle unstructured memory access.
/// @param op The triton::AddPtrOp to be rewritten.
/// @param adaptor The adaptor of the triton::AddPtrOp, used to get operands.
/// @param rewriter The pattern rewriter used to modify the IR.
/// @param data The BlockData containing information about the memory access.
void BlockDataParser::rewriteAddPtrToUnstrucMemAcc(
    triton::AddPtrOp op, triton::AddPtrOp::Adaptor &adaptor,
    ConversionPatternRewriter &rewriter, BlockData &data) {
  auto loc = op.getLoc();
  auto &offsets = data.getOffsetsRef();
  auto &blockSizes = data.getSizesRef();
  auto &strides = data.getStridesRef();
  Value ptrOffset = adaptor.getOffset();
  Value zeroIdx =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
  Value oneIdx =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
  auto addptrRes = op.getResult();
  assert(addptrRes.hasOneUse() && "Invalid: tt.addptr has multiple users");
  auto loadOp = *(addptrRes.user_begin());

  // Prepare empty tensor for loop based scalar load
  // FIXME: We use cast here because addptr must return tensor<?x!tt.ptr<f32>>.
  // True?
  auto resTy = cast<ShapedType>(addptrRes.getType());
  auto resEPtrTy = resTy.getElementType();
  auto resETy = cast<triton::PointerType>(resEPtrTy).getPointeeType();
  Value loaded = rewriter.create<tensor::EmptyOp>(loc, blockSizes, resETy);
  SmallVector<Value> initArgs;
  initArgs.push_back(loaded);

  SmallVector<Value> forLBs;
  SmallVector<Value> forUBs;
  SmallVector<Value> forSteps;
  for (auto &s : offsets) {
    forLBs.push_back(zeroIdx);
  }
  for (auto &s : blockSizes) {
    forUBs.push_back(getValueOrCreateConstantIndexOp(rewriter, loc, s));
  }
  for (auto &s : strides) {
    forSteps.push_back(oneIdx);
  }
  SmallVector<Value> ivs;
  OpBuilder builder(op);
  auto loop = createNestedLoops(
      builder, loc, 0, blockSizes.size(), forLBs, forUBs, forSteps, ivs,
      initArgs,
      [&](OpBuilder &bB, Location bLoc, SmallVector<Value> &allIVs,
          ValueRange iterArgs) {
        OpBuilder::InsertionGuard g(bB);
        bB.setInsertionPointToStart(bB.getBlock());

        Value scalarOffsetRaw =
            bB.create<tensor::ExtractOp>(bLoc, ptrOffset, allIVs);
        Value scalarOffset = bB.create<arith::IndexCastOp>(
            bLoc, bB.getIndexType(), scalarOffsetRaw);
        // Replace offset & size. Only single element.
        data.getOffsetsRef().clear();
        data.getOffsetsRef().push_back(scalarOffset);
        data.getSizesRef().clear();
        data.getSizesRef().push_back(bB.getIndexAttr(1));
        data.getStridesRef().clear();
        data.getStridesRef().push_back(bB.getIndexAttr(1));
        memref::ReinterpretCastOp castOp = data.createCastOp({1}, bLoc, bB);
        rewriter.replaceOp(op, castOp);
        // Move tt.load using this tt.addptr into this block
        loadOp->moveAfter(castOp);
        loadOp->setAttr("IndirectLoad", UnitAttr::get(op.getContext()));
        bB.create<scf::YieldOp>(bLoc, iterArgs);
      });
}

} // namespace triton
} // namespace mlir
