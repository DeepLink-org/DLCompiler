#ifndef TRITON_CONVERSION_PATTERNS_FLAGTREE
#define TRITON_CONVERSION_PATTERNS_FLAGTREE

//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Some functions in this hpp file are partially borrowed from the
// open-source project "Triton-Linalg", with the address:
// https://github.com/Cambricon/triton-linalg.
// * The original author's Copyright (C) [2022-2025] by Cambricon.
// * Several modifications have been made based on this.
//
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

namespace {

// Copyright (C) [2022-2025] by Cambricon.
// flagtree: Extract the first slice along a specified dimension from
// input tensor. This function creates a tensor slice containing only
// the first element along the given dimension
Value sliceFirst(ConversionPatternRewriter &rewriter, Location loc, Value input,
                 int64_t dim, bool reverse = false) {
  ShapedType inputType = cast<ShapedType>(input.getType());
  auto sizes =
      llvm::to_vector(llvm::map_range(inputType.getShape(), [&](int64_t t) {
        return OpFoldResult(rewriter.getI64IntegerAttr(t));
      }));
  int64_t rank = inputType.getRank();
  // Retrieve slice offsets of input.
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  if (reverse)
    offsets[dim] = rewriter.getIndexAttr(inputType.getDimSize(dim) - 1);
  // Retrieve slice sizes of input.
  sizes[dim] = rewriter.getIndexAttr(1);
  // Retrieve slice strides of input.
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  // Create the slice of input.
  return rewriter.create<tensor::ExtractSliceOp>(loc, input, offsets, sizes,
                                                 strides);
}

// Copyright (C) [2022-2025] by Cambricon.
// flagtree: Extract the remaining slices (excluding first) along a specified
// dimension from input tensor. This function creates a tensor slice containing
// all elements except the first along the given dimension
Value sliceRemaining(ConversionPatternRewriter &rewriter, Location loc,
                     Value input, int64_t dim, bool reverse = false) {
  ShapedType inputType = cast<ShapedType>(input.getType());
  auto sizes =
      llvm::to_vector(llvm::map_range(inputType.getShape(), [&](int64_t t) {
        return OpFoldResult(rewriter.getI64IntegerAttr(t));
      }));
  int64_t rank = inputType.getRank();
  // Retrieve slice sizes of input.
  sizes[dim] = rewriter.getIndexAttr(inputType.getDimSize(dim) - 1);
  // Retrieve slice offsets of input.
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  if (!reverse)
    offsets[dim] = rewriter.getIndexAttr(1);
  // Retrieve slice strides of input.
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  // Create the slice of input.
  return rewriter.create<tensor::ExtractSliceOp>(loc, input, offsets, sizes,
                                                 strides);
}

// Copyright (C) [2022-2025] by Cambricon.
// flagtree: Create reassociation maps for tensor reshape operations
// between expanded and collapsed shapes. This function generates the
// mapping needed for tensor.collapse_shape operations
bool createReassociationMaps(
    OpBuilder &builder, llvm::ArrayRef<int64_t> expandedShape,
    llvm::ArrayRef<int64_t> collapsedShape,
    llvm::SmallVector<ReassociationExprs, 4> &reassociationMap) {
  if (collapsedShape.empty()) {
    reassociationMap = {};
    return true;
  }

  // As tensor.expand_shape/tensor.collapse_shape expected rank
  // expansion/reduction.
  if (expandedShape.size() == collapsedShape.size())
    return false;
  if (ShapedType::isDynamicShape(expandedShape) ||
      ShapedType::isDynamicShape(collapsedShape))
    return false;
  // flagtree: Initialize reassociation map with size equal to
  // collapsed dimensions
  reassociationMap.resize(collapsedShape.size());
  unsigned currExpandDim = 0, currCollapseDim = 0;
  // flagtree: Iterate through dimensions to create mapping between
  // expanded and collapsed shapes
  while (currExpandDim < expandedShape.size() &&
         currCollapseDim < collapsedShape.size()) {
    int64_t dstSize = collapsedShape[currCollapseDim];
    int64_t srcSize = expandedShape[currExpandDim];

    // flagtree: Accumulate dimensions until we match the target
    // collapsed dimension size
    while (srcSize < dstSize && currExpandDim < expandedShape.size()) {
      reassociationMap[currCollapseDim].push_back(
          builder.getAffineDimExpr(currExpandDim++));
      srcSize *= expandedShape[currExpandDim];
    }
    if (srcSize == dstSize) {
      reassociationMap[currCollapseDim].push_back(
          builder.getAffineDimExpr(currExpandDim++));
      // If the next dim in collapsedShape is not 1, treat subsequent dims in
      // expandedShape which are 1 to be collapsed.
      if (currCollapseDim == collapsedShape.size() - 1 ||
          collapsedShape[currCollapseDim + 1] != 1) {
        while (currExpandDim < expandedShape.size() &&
               expandedShape[currExpandDim] == 1) {
          reassociationMap[currCollapseDim].push_back(
              builder.getAffineDimExpr(currExpandDim++));
        }
      }
    }
    // If the reassociationMap for the currCollapseDim is empty, clear all
    // mappings and return false.
    if (reassociationMap[currCollapseDim].empty()) {
      reassociationMap.clear();
      return false;
    }
    currCollapseDim++;
  }
  // If both iterators didn't reach the end, we have leftover dimentions which
  // implies that we have a mismatch in shape.
  return currExpandDim == expandedShape.size() &&
         currCollapseDim == collapsedShape.size();
}

// flagtree: Lower tt.reduce to linalg.reduce, by initialization with the
// first element.
LogicalResult applyLinalgReduce(triton::ReduceOp op,
                                typename triton::ReduceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) {
  Location loc = op->getLoc();

  // Derive types. `tt.reduce` treats reducing a 1-D tensor with a special
  // case that returns a scalar, but we treat it as a 0-D tensor in these
  // types.
  auto convertedInputTensorTypes =
      llvm::map_range(adaptor.getOperands().getTypes(),
                      [](Type t) { return cast<TensorType>(t); });
  assert(llvm::all_equal(llvm::map_range(
      convertedInputTensorTypes, [](TensorType t) { return t.getShape(); })));
  static_cast<void>(convertedInputTensorTypes);

  auto originalResultTensorTypes =
      llvm::map_range(op.getResultTypes(), [](Type t) -> TensorType {
        if (auto tensorType = dyn_cast<TensorType>(t))
          return tensorType;
        return RankedTensorType::get({}, t);
      });
  assert(llvm::all_equal(llvm::map_range(
      originalResultTensorTypes, [](TensorType t) { return t.getShape(); })));
  ArrayRef<int64_t> resultShape =
      (*originalResultTensorTypes.begin()).getShape();
  auto convertedResultTensorTypes =
      llvm::map_range(originalResultTensorTypes, [&](TensorType t) {
        return RankedTensorType::get(resultShape, t.getElementType());
      });

  llvm::SmallVector<Value> initVals;
  llvm::SmallVector<Value> inputVals;
  // To lowering to linalg.reduce, we use the first slice of the reduction
  // axis of input operands as the init value of init operands. And then,
  // reduce the remaining elements of input operands.
  // We assume that the number of input operands is same as init operands and
  // corresponds one to one.
  // TODO: This restriction will need to be relaxed in the future.

  assert(adaptor.getOperands().size() == op.getNumResults() &&
         "tt.reduce requires the same input number and init number");
  for (auto [inputVal, initTy] :
       llvm::zip(adaptor.getOperands(), convertedResultTensorTypes)) {
    ShapedType inputTy = cast<ShapedType>(inputVal.getType());
    ArrayRef<int64_t> inputShape = inputTy.getShape();

    // If the size of reduce axis is 1, we will replace init operands by input
    // operands, so we should resize the input operands' shape by init
    // operands.
    if (inputShape[op.getAxis()] <= 1) {
      assert(inputVals.empty() &&
             "tt.reduce requires the same shape of all input operands");
      SmallVector<ReassociationExprs, 4> reassociationMap;
      [[maybe_unused]] bool res = createReassociationMaps(
          rewriter, inputShape, initTy.getShape(), reassociationMap);
      assert(res && "attempting to collapse into an incompatible shape");
      auto collapse = rewriter.create<tensor::CollapseShapeOp>(
          loc, inputVal, reassociationMap);
      initVals.push_back(collapse);
      continue;
    }

    // 1. Slice the first elements of input operands, and use them as init
    //    operands' init value.
    {
      Value slice = sliceFirst(rewriter, loc, inputVal, op.getAxis());
      auto sliceShape = cast<ShapedType>(slice.getType()).getShape();

      // Resize slice value's shape by init operand.
      SmallVector<ReassociationExprs, 4> reassociationMap;
      [[maybe_unused]] bool res = createReassociationMaps(
          rewriter, sliceShape, initTy.getShape(), reassociationMap);
      assert(res && "attempting to collapse into an incompatible shape");
      auto collapse = rewriter.create<tensor::CollapseShapeOp>(
          loc, slice, reassociationMap);
      initVals.push_back(collapse);
    }
    // 2. Slice the remaining elements of input operands, reduce them and
    //    init value.
    {
      Value slice = sliceRemaining(rewriter, loc, inputVal, op.getAxis());
      inputVals.push_back(slice);
    }
  }

  // If the results are scalar, we need to extract the scalar from the
  // 0-ranked result tensor.
  auto getFinalResults = [&](ValueRange results) -> SmallVector<Value> {
    if (!resultShape.empty())
      return results;
    SmallVector<Value> extractResults;
    for (auto [tensor, type] : llvm::zip(results, convertedResultTensorTypes)) {
      Value scalar = rewriter.create<tensor::ExtractOp>(
          loc, type.getElementType(), tensor, /*indices=*/ValueRange{});
      extractResults.push_back(scalar);
    }
    return extractResults;
  };

  // If the the size of reduce axis is 1, we just replace the init operands by
  // input operands.
  if (inputVals.empty()) {
    rewriter.replaceOp(op, getFinalResults(initVals));
    return success();
  }

  // Create a linalg.reduce on the same input and move the combine region
  // there. (ReduceReturnOpConversion will take care of the terminator.)
  auto reduceOp = rewriter.create<linalg::ReduceOp>(
      loc, /*resultTypes=*/SmallVector<Type>(convertedResultTensorTypes),
      /*inputs=*/inputVals, /*inits=*/initVals,
      /*dimensions=*/ArrayRef<int64_t>{op.getAxis()});
  rewriter.inlineRegionBefore(op.getCombineOp(), reduceOp.getCombiner(),
                              reduceOp.getCombiner().end());

  rewriter.replaceOp(op, getFinalResults(reduceOp.getResults()));
  return success();
}

} // namespace

#endif
