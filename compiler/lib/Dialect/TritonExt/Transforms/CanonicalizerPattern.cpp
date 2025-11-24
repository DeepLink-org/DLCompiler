
#include "dicp/Dialect/TritonExt/Transforms/CanonicalizerPattern.h"
#include "dicp/Dialect/TritonExt/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"

#include "triton-shared/Analysis/MaskAnalysis.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <cassert>
#include <numeric>
#include <type_traits>

#define DEBUG_TYPE "triton-load-store-converter"

using namespace mlir;
using namespace triton;

namespace mlir::dicp::trtion_ext {
const std::string GeneratedByMakeTensorPtrTAG = "GeneratedByMakeTensorPtr";
static SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n) {
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}

AtomicRMWConverter::AtomicRMWConverter(MLIRContext *context)
    : OpConversionPattern<triton::AtomicRMWOp>(context) {}

// lowering tt.atomicRMW to linalg.generic
// If atomic op's return value is used by other op as it's the old value stored
// at the ptrwe will use tt.load to get it
//
// example:
// input:
//  %return_value = tt.atomic_rmw fadd, acq_rel, gpu,
//     %output_memref, %input_tensor, %mask :
//             (tensor<256x!tt.ptr<f32>>, tensor<256xf32>, tensor<256xi1>)
//                       -> tensor<256xf32>
//
// output:
//  memref.copy %output_memref, %ub_buf : memref<?xf32> to memref<?xf32>
//  %17 = bufferization.to_tensor %alloc_3 restrict writable : memref<256xf32>
//  linalg.generic
//    {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
//    ins(%output_memref, %masked_input_memref : memref<?xf32>, memref<?xf32>)
//    outs(%subview_2 : memref<?xf32>)
//    attrs = {GenericAtomicRMW = "fadd", MemSemantic = "acq_rel",
//                                        MemSyncScope = "gpu"} {
//    ^bb0(%in: f32, %in_9: f32, %out: f32):
//      %25 = arith.addf %in, %in_9 : f32
//      linalg.yield %25 : f32
//    }
LogicalResult
AtomicRMWConverter::matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  // If the result of AtomicRMWOp is not used, we don't need to load the old
  // data stored at the ptr
  auto ptr = adaptor.getPtr();
  auto val = op.getVal();
  auto loc = op.getLoc();

  auto resType = dyn_cast<TensorType>(op.getResult().getType());
  if (!resType) {
    return rewriter.notifyMatchFailure(
        op, "atomicRMWConverter: scalar will be handled by "
            "ScalarAtomicRMWCanonicalizer");
  }

  auto rmwOp = op.getAtomicRmwOp();
  if (rmwOp == triton::RMWOp::UMAX || rmwOp == triton::RMWOp::UMIN) {
    return rewriter.notifyMatchFailure(
        op, "AtomicRMWConverter: unsupported atomic kind for now");
  }

  // 1. Simple case where no mask is used.
  auto type = dyn_cast<MemRefType>(ptr.getType());
  if (!type) {
    // Seen when implicit broadcasting is done late in a chain of
    // operations. The workaround is to broadcast the pointers early in the
    // address calculation. A proper fix is complicated, but at least we can
    // provide a better error message.
    return rewriter.notifyMatchFailure(
        op, "AtomicRMWOp expects a memref, not a memref of pointers");
  }

  auto dstMemref = ptr;
  // Well, linalg structure op wouldn't support mixed tensor/buffer semantics
  // any more in latest LLVM(triton LLVM dependency has involed this), so we
  // need to convert tensor to buffer early.
  auto dstOriType = cast<MemRefType>(dstMemref.getType());
  MemRefType dstType =
      MemRefType::get(dstOriType.getShape(), dstOriType.getElementType());
  Value inputMemref =
      rewriter.create<bufferization::ToBufferOp>(loc, dstType, val);

  // 2. handle the mask for the atomic op
  // When the dsl do not pass the mask to this op like
  // `tl.atomic_add(out_ptr0 + xindex, tmp2)`, it will create a constant mask
  // for this op by default, which is not supported by maskAnalysis, so we
  // need to handle this situation
  //
  // This logic come from semantic.py:
  //
  // if not mask:
  //     mask_ir = builder.get_int1(True)
  //     mask_ty = tl.int1
  //     if ptr.type.is_block():
  //         mask_ir = \
  //             builder.create_splat(mask_ir, ptr.type.get_block_shapes())
  //         mask_ty = tl.block_type(tl.int1, ptr.type.get_block_shapes())
  //     mask = tl.tensor(mask_ir, mask_ty)
  //
  // ...
  //
  // return ptr, val, mask
  //
  if (auto mask = op.getMask()) {
    triton::MaskState mstate;
    auto constantMask = mask.getDefiningOp<arith::ConstantOp>();
    if (!constantMask) {
      auto isContMask = mstate.parse(mask, loc, rewriter);

      if (isContMask.failed()) {
        return rewriter.notifyMatchFailure(
            op, "Cannot lower continuous masked loads");
      }
      dstMemref = mstate.getSubview(ptr, loc, rewriter);
      inputMemref = mstate.getSubview(inputMemref, loc, rewriter);
    } else {
      if (!isConstantMaskTrue(mask)) {
        rewriter.eraseOp(op);
        return success();
      }
    }
  }

  // create element-wise map
  int64_t rank = type.getRank();
  SmallVector<AffineExpr> inputDims;
  auto context = rewriter.getContext();

  for (int i = 0; i < rank; i++) {
    inputDims.push_back(getAffineDimExpr(i, context));
  }

  SmallVector<AffineMap> indexingMaps;
  // As mask has been erased for now
  // the number of input must be 2
  // the input memref is also the output memref
  // Thus, there are a total of three inputs and outputs.
  // so here we have 3 map to create
  for (int i = 0; i < 3; i++) {
    indexingMaps.push_back(AffineMap::get(rank, 0, inputDims, context));
  }

  Value tensorToReplace;
  if (!op.getResult().use_empty()) {
    auto tensorType =
        RankedTensorType::get(type.getShape(), type.getElementType());
    auto alloc = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(type.getShape(), type.getElementType()));
    // For the return value, don't need to care about mask for now
    // this op don't support other, so we best not fill it
    rewriter.create<memref::CopyOp>(loc, ptr, alloc);
    tensorToReplace = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, alloc, true /* restrict */, true /* writable */);
  }

  auto linalgOp = rewriter.create<linalg::GenericOp>(
      loc, /* operands */ ValueRange{dstMemref, inputMemref},
      ValueRange{dstMemref}, indexingMaps, getNParallelLoopsAttrs(rank),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        Value opResult = createAtomicBinaryOps(nestedBuilder, nestedLoc, op,
                                               type.getElementType(),
                                               blockArgs[0], blockArgs[1]);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, opResult);
      });

  // "library_call"
  // indicating the actual semantic of this op
  // TODO: If the hardware support the MemSemantic/MemSyncScope
  //       We pass them down
  //       otherwise they need to be deleted
  const StringRef genericAtomicRMW = "GenericAtomicRMW";
  const StringRef memSemantic = "MemSemantic";
  const StringRef memSyncScope = "MemSyncScope";
  linalgOp->setAttr(genericAtomicRMW,
                    rewriter.getStringAttr(stringifyEnum(op.getAtomicRmwOp())));
  linalgOp->setAttr(memSemantic,
                    rewriter.getStringAttr(stringifyEnum(op.getSem())));
  linalgOp->setAttr(memSyncScope,
                    rewriter.getStringAttr(stringifyEnum(op.getScope())));

  // Mark atomic_and/or/xor specially which need software simulation in terms
  // of backend restriction
  if (softwareAtomicKinds.contains(op.getAtomicRmwOp()))
    linalgOp->setAttr("Software", rewriter.getUnitAttr());

  // tt.atomicRMW op has two part of feature
  // 1. load the old data at the ptr
  // 2. atomically store the data on ub to the ptr
  //    at the same time it perform the action it has been assigned
  // So we lower this op to load + atomically store
  //
  // The first part is not necessary when the returned value of atomic op
  // is not used, it will be deleted cause it's meaningless
  // Here, we preemptively determine whether it will be used
  // and decide whether it is necessary to create the load process based on
  // this assessment.
  //
  // logic of handling is copied
  // TODO: decoupling the logic of load, put it in the Utils
  if (!op.getResult().use_empty()) {
    rewriter.replaceOp(op, tensorToReplace);
  } else {
    rewriter.eraseOp(op);
  }
  return success();
}

LogicalResult
AtomicCASConverter::matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  // If the result of AtomicCASOp is not used, we don't need to load the old
  // data stored at the ptr
  auto ptr = adaptor.getPtr();
  auto cmp = op.getCmp();
  auto val = op.getVal();
  auto loc = op.getLoc();

  auto resType = dyn_cast<TensorType>(op.getResult().getType());
  if (!resType) {
    return rewriter.notifyMatchFailure(
        op, "atomicCASConverter: scalar will be handled by "
            "ScalarAtomicCASCanonicalizer");
  }

  // 1. Simple case where no mask is used.
  auto type = dyn_cast<MemRefType>(ptr.getType());
  if (!type) {
    // Seen when implicit broadcasting is done late in a chain of
    // operations. The workaround is to broadcast the pointers early in the
    // address calculation. A proper fix is complicated, but at least we can
    // provide a better error message.
    return rewriter.notifyMatchFailure(
        op, "AtomicCASOp expects a memref, not a memref of pointers");
  }

  auto dstMemref = ptr;
  // Well, linalg structure op wouldn't support mixed tensor/buffer semantics
  // any more in latest LLVM(triton LLVM dependency has involed this), so we
  // need to convert tensor to buffer early.
  auto dstOriType = cast<MemRefType>(dstMemref.getType());
  MemRefType dstType =
      MemRefType::get(dstOriType.getShape(), dstOriType.getElementType());
  Value inputMemref =
      rewriter.create<bufferization::ToBufferOp>(loc, dstType, val);

  Value cmpMemref =
      rewriter.create<bufferization::ToBufferOp>(loc, dstType, cmp);

  // create element-wise map
  int64_t rank = type.getRank();
  SmallVector<AffineExpr> inputDims;
  auto context = rewriter.getContext();

  for (int i = 0; i < rank; i++) {
    inputDims.push_back(getAffineDimExpr(i, context));
  }

  SmallVector<AffineMap> indexingMaps;
  // As mask has been erased for now
  // the number of input must be 2
  // the input memref is also the output memref
  // Thus, there are a total of four inputs and outputs.
  // so here we have 4 map to create
  for (int i = 0; i < 4; i++) { // 4: 3 input and 1 output
    indexingMaps.push_back(AffineMap::get(rank, 0, inputDims, context));
  }

  auto linalgOp = rewriter.create<linalg::GenericOp>(
      loc, ValueRange{dstMemref, cmpMemref, inputMemref},
      mlir::ValueRange{dstMemref}, indexingMaps, getNParallelLoopsAttrs(rank),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        Value lhs = blockArgs[0];
        Value rhs = blockArgs[1];
        Value setValue = blockArgs[2];
        Value cond;
        if (mlir::isa<mlir::FloatType>(lhs.getType())) {
          cond = nestedBuilder.create<arith::CmpFOp>(
              nestedLoc, arith::CmpFPredicate::UEQ, lhs, rhs);
        } else {
          cond = nestedBuilder.create<arith::CmpIOp>(
              nestedLoc, arith::CmpIPredicate::eq, lhs, rhs);
        }
        auto ifOp = nestedBuilder.create<scf::IfOp>(
            nestedLoc, TypeRange{setValue.getType()}, cond, true);
        {
          OpBuilder::InsertionGuard guard(nestedBuilder);
          nestedBuilder.setInsertionPointToEnd(&ifOp.getThenRegion().front());
          nestedBuilder.create<scf::YieldOp>(nestedLoc, setValue);
        }
        {
          OpBuilder::InsertionGuard guard(nestedBuilder);
          nestedBuilder.setInsertionPointToEnd(&ifOp.getElseRegion().front());
          nestedBuilder.create<scf::YieldOp>(nestedLoc, lhs);
        }
        nestedBuilder.setInsertionPointToEnd(nestedBuilder.getBlock());
        nestedBuilder.create<mlir::linalg::YieldOp>(nestedLoc,
                                                    ifOp.getResult(0));
      });

  const StringRef genericAtomicRMW = "GenericAtomicRMW";
  const StringRef memSemantic = "MemSemantic";
  const StringRef memSyncScope = "MemSyncScope";
  auto attr = mlir::StringAttr::get(context, "cas");

  linalgOp->setAttr(genericAtomicRMW, attr);
  linalgOp->setAttr(memSemantic,
                    rewriter.getStringAttr(stringifyEnum(op.getSem())));
  linalgOp->setAttr(memSyncScope,
                    rewriter.getStringAttr(stringifyEnum(op.getScope())));

  linalgOp->setAttr("Software", rewriter.getUnitAttr());

  // tt.atomicRMW op has two part of feature
  // 1. load the old data at the ptr
  // 2. atomically store the data on ub to the ptr
  //    at the same time it perform the action it has been assigned
  // So we lower this op to load + atomically store
  //
  // The first part is not necessary when the returned value of atomic op
  // is not used, it will be deleted cause it's meaningless
  // Here, we preemptively determine whether it will be used
  // and decide whether it is necessary to create the load process based on
  // this assessment.
  //
  // logic of handling is copied
  if (!op.getResult().use_empty()) {
    auto tensorType =
        RankedTensorType::get(type.getShape(), type.getElementType());
    auto alloc = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(type.getShape(), type.getElementType()));

    // For the return value, don't need to care about mask for now
    // this op don't support other, so we best not fill it
    rewriter.create<memref::CopyOp>(loc, ptr, alloc);
    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, alloc, true /* restrict */, true /* writable */);
    rewriter.replaceOp(op, tensor);
  } else {
    rewriter.eraseOp(op);
  }
  return success();
}

LogicalResult
ScalarStoreCanonicalizer::matchAndRewrite(triton::StoreOp op,
                                          PatternRewriter &rewriter) const {
  if (!op.getValue().getType().isIntOrIndexOrFloat()) {
    return rewriter.notifyMatchFailure(
        op, "ScalarStoreCanonicalizer handles scalar store scene!");
  }
  auto ptr = op.getPtr();
  auto mask = op.getMask();
  auto value = op.getValue();
  if (mask) {
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, mask, [&](OpBuilder &b, Location loc) {
          b.create<triton::StoreOp>(loc, ptr, value, op.getCache(),
                                    op.getEvict());
          b.create<scf::YieldOp>(loc);
        });
    return success();
  }

  auto ptrTy = RankedTensorType::get({(int64_t)1}, ptr.getType());
  auto ptrSplat = rewriter.create<triton::SplatOp>(op.getLoc(), ptrTy, ptr);
  auto valTy = RankedTensorType::get({(int64_t)1}, value.getType());
  auto valSplat = rewriter.create<triton::SplatOp>(op.getLoc(), valTy, value);
  auto newStoreOp = rewriter.create<triton::StoreOp>(
      op.getLoc(), ptrSplat, valSplat, op.getCache(), op.getEvict());
  rewriter.replaceOp(op, newStoreOp);
  return success();
}

LogicalResult
ScalarAtomicRMWCanonicalizer::matchAndRewrite(triton::AtomicRMWOp op,
                                              PatternRewriter &rewriter) const {
  if (!op.getVal().getType().isIntOrIndexOrFloat()) {
    return rewriter.notifyMatchFailure(
        op, "ScalarAtomicRMWCanonicalizer handles scalar atomic rmw op scene!");
  }

  auto ptr = op.getPtr();
  auto ptrTy = RankedTensorType::get({(int64_t)1}, ptr.getType());
  auto ptrSplat = rewriter.create<triton::SplatOp>(op.getLoc(), ptrTy, ptr);
  auto valTy = RankedTensorType::get({(int64_t)1}, op.getVal().getType());
  auto valSplat =
      rewriter.create<triton::SplatOp>(op.getLoc(), valTy, op.getVal());
  auto maskTy = RankedTensorType::get({(int64_t)1}, op.getMask().getType());
  auto maskSplat =
      rewriter.create<triton::SplatOp>(op.getLoc(), maskTy, op.getMask());

  auto newAtomicOp = rewriter.create<triton::AtomicRMWOp>(
      op.getLoc(), valTy, op.getAtomicRmwOp(), ptrSplat, valSplat, maskSplat,
      op.getSem(), op.getScope());
  auto idxZero =
      rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0));
  rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, newAtomicOp,
                                                 ValueRange({idxZero}));
  return success();
}

LogicalResult
ScalarAtomicCASCanonicalizer::matchAndRewrite(triton::AtomicCASOp op,
                                              PatternRewriter &rewriter) const {
  if (!op.getVal().getType().isIntOrIndexOrFloat() &&
      !op.getCmp().getType().isIntOrIndexOrFloat()) {
    return rewriter.notifyMatchFailure(
        op, "ScalarAtomicCASCanonicalizer handles scalar atomic cas op scene!");
  }

  auto ptr = op.getPtr();
  auto ptrTy = RankedTensorType::get({(int64_t)1}, ptr.getType());
  auto ptrSplat = rewriter.create<triton::SplatOp>(op.getLoc(), ptrTy, ptr);
  auto cmpTy = RankedTensorType::get({(int64_t)1}, op.getCmp().getType());
  auto cmpSplat =
      rewriter.create<triton::SplatOp>(op.getLoc(), cmpTy, op.getCmp());
  auto valTy = RankedTensorType::get({(int64_t)1}, op.getVal().getType());
  auto valSplat =
      rewriter.create<triton::SplatOp>(op.getLoc(), valTy, op.getVal());

  auto newAtomicOp = rewriter.create<triton::AtomicCASOp>(
      op.getLoc(), valTy, ptrSplat, cmpSplat, valSplat, op.getSem(),
      op.getScope());
  auto idxZero =
      rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0));
  rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, newAtomicOp,
                                                 ValueRange({idxZero}));
  return success();
}

// The atomic max op with float input will be devided into
// two atomic max ops with integer input
// One handles the part of the tensor greater than zero
// the other deals with the part less than zero
// It will lead to maskAnalysis failure
// So here we need to revert the procedures in semantics.py
// The triton IR is like
//
// %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x256xf32>
// %1 = tt.bitcast %value : tensor<1x256xf32> -> tensor<1x256xi32>
// %2 = tt.bitcast %ptr : tensor<1x256x!tt.ptr<f32>> ->
// tensor<1x256x!tt.ptr<i32>> %3 = arith.cmpf oge, %1, %cst_0 %4 = arith.cmpf
// olt, %1, %cst_0 %5 = arith.andi %8, %3 %6 = tt.atomic_rmw max, acq_rel, gpu,
// %2, %1, %5 :
//    (tensor<1x256x!tt.ptr<i32>>, tensor<1x256xi32>, tensor<1x256xi1>) ->
//    tensor<1x256xi32>
// %7 = arith.andi %8, %4
// %8 = tt.atomic_rmw umin, acq_rel, gpu, %2, %1, %7 :
//    (tensor<1x256x!tt.ptr<i32>>, tensor<1x256xi32>, tensor<1x256xi1>) ->
//    tensor<1x256xi32>
//
// it's hard to handle and meaningless complicated for our device
// so we revert it to
// %0 = tt.atomic_rmw max, acq_rel, gpu, %23, %21, %8 :
//    (tensor<1x256x!tt.ptr<f32>>, tensor<1x256xf32>, tensor<1x256xi1>) ->
//    tensor<1x256xf32>
LogicalResult
AtomicMaxMinCanonicalizer::matchAndRewrite(triton::AtomicRMWOp op,
                                           PatternRewriter &rewriter) const {
  // Revert the op to its original form
  auto ptrBitcastOp = op.getPtr().getDefiningOp<triton::BitcastOp>();
  auto valueBitcastOp = op.getVal().getDefiningOp<triton::BitcastOp>();
  if (!ptrBitcastOp || !valueBitcastOp) {
    return failure();
  }

  // We only need to handle the op when the element type is float
  auto elementType =
      dyn_cast<TensorType>(valueBitcastOp.getSrc().getType()).getElementType();
  if (!isa<FloatType>(elementType)) {
    return failure();
  }

  auto rmwOp = op.getAtomicRmwOp();
  // here we know that atomic UMAX/UMIN
  // is created by special logic of triton right now
  // so we can simply delete it
  if (rmwOp == triton::RMWOp::UMAX || rmwOp == triton::RMWOp::UMIN) {
    // if the return value of op is used, we can't simply erase it
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }

  if (rmwOp != triton::RMWOp::MAX && rmwOp != triton::RMWOp::MIN) {
    return failure();
  }

  // 1. Though semantic interpreter will generate full true tensor as original
  // mask if atomicrmwOp don't have it, above float devision process will also
  // generate positive and negative comparison mask, which will cause to fold
  // true mask.
  // 2. While if atomicrmwOp has original mask, there exists andiop between
  // original mask and positive/negative comparison mask
  //
  // Here wanna extract original mask
  Value originalMask = op.getMask();
  if (auto andOp = originalMask.getDefiningOp<arith::AndIOp>())
    // LHS is convention in semantic interpreter
    originalMask = andOp.getLhs();
  else if (auto cmpOp = originalMask.getDefiningOp<arith::CmpFOp>()) {
    if (cmpOp.getPredicate() != mlir::arith::CmpFPredicate::OGE ||
        !matchPattern(cmpOp.getRhs(),
                      /*positive float zero matcher*/ m_PosZeroFloat()))
      // Here recheck frontend interpreter generation in no manual mask state
      return op->emitError("Illegal mask for atomicrmwOp of float type");
    // Restore original true mask
    originalMask = rewriter.create<arith::ConstantOp>(
        op->getLoc(),
        /*typed attr*/ DenseElementsAttr::get(
            cast<ShapedType>(originalMask.getType()), true));
  } else
    return op->emitError("Illegal mask for atomicrmwOp of float type");

  auto originAtomicOp = rewriter.create<triton::AtomicRMWOp>(
      op.getLoc(), valueBitcastOp.getSrc().getType(), op.getAtomicRmwOp(),
      ptrBitcastOp.getSrc(), valueBitcastOp.getSrc(), originalMask, op.getSem(),
      op.getScope());

  // if the return value of op is used
  // we need to handle its usage
  // In semantic.py, if the atomic Max/Min with float input is used
  // It will use select + bitcast to get float value
  // so here we need to revert it too
  //
  // For example:
  // %0 = tt.atomic_rmw max, acq_rel, gpu, %gm, %input, %mask1 :
  // (tensor<32x!tt.ptr<i32>>... %1 = tt.atomic_rmw umin, acq_rel, gpu, %gm,
  // %input, %mask2 : (tensor<32x!tt.ptr<i32>>... %2 = arith.select
  // %devidedMask, %0, %1 : tensor<32xi1>, tensor<32xi32> %3 = tt.bitcast %2 :
  // tensor<32xi32> -> tensor<32xf32> tt.store %outputMemref, %3 :
  // tensor<32x!tt.ptr<f32>>
  //
  // will be revert to:
  // %0 = tt.atomic_rmw max, acq_rel, gpu, %gm, %input, %mask :
  // (tensor<32x!tt.ptr<f32>>... tt.store %outputMemref, %0 :
  // tensor<32x!tt.ptr<f32>>
  //
  if (!op.getResult().use_empty()) {
    for (OpOperand &use : op->getUses()) {
      auto selectOp = dyn_cast<arith::SelectOp>(use.getOwner());
      if (!selectOp)
        continue;

      for (OpOperand &selectUse : selectOp->getUses()) {
        if (auto bitcastOp =
                dyn_cast<triton::BitcastOp>(selectUse.getOwner())) {
          bitcastOp.getResult().replaceAllUsesWith(originAtomicOp);
        }
      }
    }
    rewriter.replaceOp(op, originAtomicOp);
  } else {
    rewriter.eraseOp(op);
  }

  return success();
}

/*
 * Move tt.bitcast to a previous location if tt.bitcast is not directly applied
 * on function arguments
 */
LogicalResult
BitcastCanonicalizer::matchAndRewrite(triton::BitcastOp bitcastOp,
                                      PatternRewriter &rewriter) const {
  Value castSrc = bitcastOp.getSrc();
  Value castRes = bitcastOp.getResult();
  Type castSrcTy = castSrc.getType();
  Type castSrcPtrTy = isa<ShapedType>(castSrcTy)
                          ? cast<ShapedType>(castSrcTy).getElementType()
                          : castSrcTy;
  if (!isa<triton::PointerType>(castSrcPtrTy))
    return failure();

  auto origBitwidth = getPointeeBitWidth(castSrc.getType());
  auto castBitwidth = getPointeeBitWidth(castRes.getType());

  if (origBitwidth == 1)
    origBitwidth = 8;
  if (castBitwidth == 1)
    castBitwidth = 8;
  if (origBitwidth != castBitwidth) {
    bitcastOp.emitError() << "Casting pointers with unmatched bitwidth!\n";
    return failure();
  }

  Operation *beforeCastOp = castSrc.getDefiningOp();
  if (beforeCastOp == nullptr) {
    return failure();
  }

  auto newRes =
      TypeSwitch<Operation *, FailureOr<Operation *>>(beforeCastOp)
          // before: addptr - bitcast - load/store
          // after: bitcast - addptr - load/store
          .Case<triton::AddPtrOp>([&](triton::AddPtrOp addptrOp) {
            auto newCastOp = rewriter.create<triton::BitcastOp>(
                bitcastOp.getLoc(), castRes.getType(), addptrOp.getPtr());
            return rewriter.create<triton::AddPtrOp>(
                bitcastOp.getLoc(), castRes.getType(), newCastOp.getResult(),
                addptrOp.getOffset());
          })
          .Case<triton::SplatOp>([&](triton::SplatOp splatOp) {
            Type newCastSrcTy =
                cast<RankedTensorType>(castRes.getType()).getElementType();

            Value splatSrc = splatOp.getSrc();
            Type splatSrcTy = splatSrc.getType();
            if (auto splatSrcTensorTy = dyn_cast<RankedTensorType>(splatSrcTy))
              newCastSrcTy =
                  splatSrcTensorTy.cloneWith(std::nullopt, newCastSrcTy);
            auto newCastOp = rewriter.create<triton::BitcastOp>(
                bitcastOp.getLoc(), newCastSrcTy, splatSrc);
            return rewriter.create<triton::SplatOp>(
                bitcastOp.getLoc(), castRes.getType(), newCastOp);
          })
          // before: bitcast - bitcast
          // after(fusion optimization): bitcast
          .Case<triton::BitcastOp>([&](triton::BitcastOp prevCastOp) {
            return rewriter.create<triton::BitcastOp>(
                bitcastOp.getLoc(), castRes.getType(), prevCastOp.getSrc());
          })
          .Default([&](Operation *op) {
            return rewriter.notifyMatchFailure(bitcastOp,
                                               "Unknown bitcast pattern");
          });
  if (succeeded(newRes)) {
    rewriter.replaceOp(bitcastOp, newRes.value());
    if (beforeCastOp->use_empty()) {
      rewriter.eraseOp(beforeCastOp);
    }
    return success();
  }
  return failure();
}

void rewriteUserWithNewOrder(
    mlir::OpOperand *use, PatternRewriter &rewriter,
    llvm::SmallVector<int64_t, 8> &blkShapeI64, // 8: container size
    mlir::Location &loc, llvm::ArrayRef<int32_t> &order, size_t &orderSize) {
  Operation *user = use->getOwner();
  rewriter.setInsertionPointAfter(user);
  if (auto loadOp = dyn_cast<triton::LoadOp>(user)) {
    auto loadResTy = loadOp.getResult().getType();
    auto loadResShapedTy = cast<ShapedType>(loadResTy);
    auto newLoadTy = loadResShapedTy.cloneWith(
        blkShapeI64, loadResShapedTy.getElementType());
    auto newLoadOp = rewriter.create<triton::LoadOp>(
        loc, newLoadTy, loadOp->getOperands(), loadOp->getAttrs());
    newLoadOp->setAttr(GeneratedByMakeTensorPtrTAG,
                       UnitAttr::get(rewriter.getContext()));
    rewriter.replaceOp(loadOp, newLoadOp);
    // load contiguous data then permute. thus the permute order is as
    // follows.
    SmallVector<int32_t, 8> permuteOrder; // 8: container size
    for (auto [i, v] : llvm::enumerate(order)) {
      permuteOrder.push_back(orderSize - 1 - order[i]);
    }
    auto permuteOp = rewriter.create<triton::TransOp>(
        loc, newLoadOp.getResult(),
        DenseI32ArrayAttr::get(loadOp.getContext(), permuteOrder));
    newLoadOp.getResult().replaceAllUsesExcept(permuteOp.getResult(),
                                               permuteOp);
  } else if (auto storeOp = dyn_cast<triton::StoreOp>(user)) {
    // permute to contiguous then store. thus the permute order is as follows.
    SmallVector<int32_t, 8> permuteOrder; // 8: container size
    for (auto [i, v] : llvm::enumerate(order)) {
      permuteOrder.push_back(order[orderSize - 1 - i]);
    }
    auto permuteOp = rewriter.create<triton::TransOp>(
        loc, storeOp.getValue(),
        DenseI32ArrayAttr::get(storeOp.getContext(), permuteOrder));
    storeOp.getValue().replaceAllUsesExcept(permuteOp.getResult(), permuteOp);
    auto newStoreOp = rewriter.create<triton::StoreOp>(
        loc, storeOp.getPtr(), storeOp.getValue(), storeOp.getMask(),
        storeOp.getBoundaryCheck(), storeOp.getCache(), storeOp.getEvict());
    rewriter.replaceOp(storeOp, newStoreOp);
  } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(user)) {
    auto advanceResPtrTy =
        cast<triton::PointerType>(advanceOp.getResult().getType());
    auto advanceResShapedTy =
        cast<ShapedType>(advanceResPtrTy.getPointeeType());
    auto newAdvanceResShapedTy = advanceResShapedTy.cloneWith(
        blkShapeI64, advanceResShapedTy.getElementType());
    auto newAdvanceResPtrTy = triton::PointerType::get(
        newAdvanceResShapedTy, advanceResPtrTy.getAddressSpace());
    auto advanceOffsets = advanceOp.getOffsets();
    llvm::SmallVector<Value, 8> newAdvanceOffsets; // 8: container size
    for (int i = orderSize - 1; i >= 0; i--) {
      newAdvanceOffsets.push_back(advanceOffsets[order[i]]);
    }
    SmallVector<OpOperand *> resUses;
    for (auto &use : advanceOp->getUses())
      resUses.push_back(&use);
    auto newAdvanceOp = rewriter.create<triton::AdvanceOp>(
        loc, newAdvanceResPtrTy, advanceOp.getPtr(), newAdvanceOffsets);
    rewriter.replaceOp(advanceOp, newAdvanceOp);
    for (auto resUse : resUses)
      rewriteUserWithNewOrder(resUse, rewriter, blkShapeI64, loc, order,
                              orderSize);
  } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(user)) {
    auto initArg = use->get();
    auto iterArg = loopOp.getTiedLoopRegionIterArg(use);
    auto resultValue = loopOp.getTiedLoopResult(use);
    iterArg.setType(initArg.getType());
    resultValue.setType(initArg.getType());
    for (auto &argUse : iterArg.getUses())
      rewriteUserWithNewOrder(&argUse, rewriter, blkShapeI64, loc, order,
                              orderSize);
    for (auto &resUse : resultValue.getUses())
      rewriteUserWithNewOrder(&resUse, rewriter, blkShapeI64, loc, order,
                              orderSize);
  } else if (isa<scf::YieldOp>(user)) {
    return;
  } else {
    llvm_unreachable(
        "[MakeTensorPtrCanonicalizer] tt.make_tensor_ptr's result is "
        "not used by load/store/advance op");
  }
}

void markLoadUsers(mlir::OpOperand *use, PatternRewriter &rewriter) {
  Operation *user = use->getOwner();
  if (auto loadOp = dyn_cast<triton::LoadOp>(user)) {
    loadOp->setAttr(GeneratedByMakeTensorPtrTAG,
                    UnitAttr::get(rewriter.getContext()));
  } else if (auto storeOp = dyn_cast<triton::StoreOp>(user)) {
    return;
  } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(user)) {
    SmallVector<OpOperand *> resUses;
    for (auto &use : advanceOp->getUses())
      resUses.push_back(&use);
    for (auto resUse : resUses)
      markLoadUsers(resUse, rewriter);
  } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(user)) {
    auto initArg = use->get();
    auto iterArg = loopOp.getTiedLoopRegionIterArg(use);
    auto resultValue = loopOp.getTiedLoopResult(use);
    iterArg.setType(initArg.getType());
    resultValue.setType(initArg.getType());
    for (auto &argUse : iterArg.getUses())
      markLoadUsers(&argUse, rewriter);
    for (auto &resUse : resultValue.getUses())
      markLoadUsers(&resUse, rewriter);
  } else if (isa<scf::YieldOp>(user)) {
    return;
  } else {
    llvm_unreachable(
        "[MakeTensorPtrCanonicalizer] tt.make_tensor_ptr's result is "
        "not used by load/store/advance op");
  }
}

LogicalResult
MakeTensorPtrCanonicalizer::matchAndRewrite(triton::MakeTensorPtrOp op,
                                            PatternRewriter &rewriter) const {
  auto order = op.getOrder();
  auto orderSize = order.size();
  if (orderSize == 1) {
    return rewriter.notifyMatchFailure(
        op, "make_tensor_ptr's order has single value.");
  }

  bool isPermuted = false;
  for (auto [first, second] : llvm::zip(order.slice(0, orderSize - 1),
                                        order.slice(1, orderSize - 1))) {
    if (first != second + 1) {
      isPermuted = true;
      break;
    }
  }

  auto loc = op.getLoc();
  auto base = op.getBase();
  auto shape = op.getShape();
  auto strides = op.getStrides();
  auto offsets = op.getOffsets();
  auto result = op.getResult();
  SmallVector<OpOperand *> opUses;

  for (auto &use : result.getUses())
    opUses.push_back(&use);
  for (auto use : opUses)
    markLoadUsers(use, rewriter);

  if (!isPermuted) {
    return rewriter.notifyMatchFailure(
        op, "make_tensor_ptr's order is contiguous.");
  }

  llvm::SmallVector<int32_t, 8> blkShapeI32;
  llvm::SmallVector<int64_t, 8> blkShapeI64;
  auto resPtrType = cast<triton::PointerType>(result.getType());
  if (auto resShapedTy = dyn_cast<ShapedType>(resPtrType.getPointeeType())) {
    auto resBlkShape = resShapedTy.getShape();
    for (auto [i, v] : llvm::enumerate(resBlkShape)) {
      auto reverseI = orderSize - 1 - i;
      blkShapeI32.push_back(resBlkShape[order[reverseI]]);
      blkShapeI64.push_back(resBlkShape[order[reverseI]]);
    }
  }

  llvm::SmallVector<Value, 8> newShape;
  llvm::SmallVector<Value, 8> newStrides;
  llvm::SmallVector<Value, 8> newOffsets;
  for (int i = orderSize - 1; i >= 0; i--) {
    newShape.push_back(shape[order[i]]);
    newStrides.push_back(strides[order[i]]);
    newOffsets.push_back(offsets[order[i]]);
  }

  llvm::SmallVector<int, 8> contiguousOrder;
  for (int i = orderSize - 1; i >= 0; i--)
    contiguousOrder.push_back(i);

  rewriter.setInsertionPoint(op);
  auto newMakeTensorPtrOp = rewriter.create<triton::MakeTensorPtrOp>(
      loc, base, ValueRange(newShape), ValueRange(newStrides),
      ValueRange(newOffsets), blkShapeI32, contiguousOrder);
  rewriter.replaceOp(op, newMakeTensorPtrOp);
  for (auto use : opUses)
    rewriteUserWithNewOrder(use, rewriter, blkShapeI64, loc, order, orderSize);
  return success();
}

LogicalResult
ReduceSingleCanonicalizer::matchAndRewrite(triton::ReduceOp reduceOp,
                                           PatternRewriter &rewriter) const {
  auto srcs = reduceOp.getSrcs();
  bool allSrcSingleElem = true;
  for (auto src : srcs) {
    auto srcType = cast<RankedTensorType>(src.getType());
    auto srcShape = srcType.getShape();
    int64_t numel = 1;
    for (auto s : srcShape) {
      numel *= s;
    }
    if (numel != 1) {
      allSrcSingleElem = false;
      break;
    }
  }

  if (!allSrcSingleElem) {
    return rewriter.notifyMatchFailure(
        reduceOp, "reduce's srcs are not all with single element");
  }

  auto results = reduceOp.getResult();
  auto loc = reduceOp->getLoc();
  auto zero = rewriter
                  .create<arith::ConstantOp>(
                      loc, rewriter.getIndexType(),
                      rewriter.getIntegerAttr(rewriter.getIndexType(), 0))
                  .getResult();
  for (int i = 0; i < srcs.size(); i++) {
    auto src = srcs[i];
    auto srcType = cast<RankedTensorType>(src.getType());
    auto srcRank = srcType.getRank();
    auto res = results[i];
    Value extracted;
    if (srcRank == 1) {
      // vector reduce generates a scalar result
      extracted =
          rewriter.create<tensor::ExtractOp>(loc, src, zero).getResult();
    } else {
      auto srcShape = srcType.getShape();
      auto resType = cast<RankedTensorType>(res.getType());
      auto resShape = resType.getShape();
      auto collapseReassociationIndicesOptional =
          getReassociationIndicesForCollapse(srcShape, resShape);
      if (!collapseReassociationIndicesOptional.has_value()) {
        return rewriter.notifyMatchFailure(
            reduceOp, "Failure with getReassociationIndicesForCollapse call");
      }
      auto collapseReassociationIndices =
          collapseReassociationIndicesOptional.value();
      extracted = rewriter
                      .create<tensor::CollapseShapeOp>(
                          loc, src, collapseReassociationIndices)
                      .getResult();
    }
    res.replaceAllUsesWith(extracted);
  }

  return success();
}

/**
 * @brief Wrapper to check if any operand of the given operation traces back to
 * triton::LoadOp.
 */
static bool anyOperandFromTritonLoad(Operation *op) {
  const int kMaxTraceDepth = 3;

  std::function<bool(Value, int)> trace = [&](Value value, int depth) -> bool {
    // Base case 1: Reached maximum depth.
    if (depth > kMaxTraceDepth)
      return false;

    Operation *defOp = value.getDefiningOp();
    // Base case 2: Value is a block argument (no defining op), stop tracing.
    if (!defOp)
      return false;

    // Base case 3: Found the target operation!
    if (dyn_cast<triton::LoadOp>(defOp))
      return true;

    // Recursive step: Check all operands of the current defining operation.
    // We increment the depth for the next level.
    for (Value operand : defOp->getOperands()) {
      if (trace(operand, depth + 1))
        return true;
    }
    return false;
  };

  // Iterate over the operands of the top-level operation (op)
  for (Value operand : op->getOperands()) {
    // Start tracing from depth 1 (first hop from the current op)
    if (trace(operand, 1))
      return true;
  }
  return false;
}

LogicalResult
RemfToBasicArithmetic::matchAndRewrite(arith::RemFOp op,
                                       PatternRewriter &rewriter) const {
  if (!anyOperandFromTritonLoad(op)) {
    return rewriter.notifyMatchFailure(
        op, "None of the operands are defined by triton::LoadOp.");
  }
  Value lhs = op.getLhs(); // %a
  Value rhs = op.getRhs(); // %b
  Location loc = op.getLoc();
  // Implementation: a - b * floor(a / b)
  // %div = arith.divf %a, %b
  Value divOp = rewriter.create<arith::DivFOp>(loc, lhs, rhs);
  // %flr = math.floor %div
  Value floorOp = rewriter.create<math::FloorOp>(loc, divOp);
  // %mul = arith.mulf %b, %flr
  Value mulOp = rewriter.create<arith::MulFOp>(loc, rhs, floorOp);
  // %result = arith.subf %a, %mul
  rewriter.replaceOpWithNewOp<arith::SubFOp>(op, lhs, mulOp);
  return success();
}

LogicalResult
RemSIToBasicArithmetic::matchAndRewrite(arith::RemSIOp op,
                                        PatternRewriter &rewriter) const {
  if (!anyOperandFromTritonLoad(op)) {
    return rewriter.notifyMatchFailure(
        op, "None of the operands are defined by triton::LoadOp.");
  }
  Value lhs = op.getLhs(); // %a
  Value rhs = op.getRhs(); // %b
  Location loc = op.getLoc();
  // Implementation: a - b * (a / b)
  // %div = arith.divsi %a, %b (Signed truncating integer division)
  Value divOp = rewriter.create<arith::DivSIOp>(loc, lhs, rhs);
  // %mul = arith.muli %b, %div
  Value mulOp = rewriter.create<arith::MulIOp>(loc, rhs, divOp);
  // %result = arith.subi %a, %mul
  rewriter.replaceOpWithNewOp<arith::SubIOp>(op, lhs, mulOp);
  return success();
}

} // namespace mlir::dicp::trtion_ext
