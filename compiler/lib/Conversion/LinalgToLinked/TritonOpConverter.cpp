#include "dicp/Conversion/LinalgToLinked/TritonOpConverter.h"
#include <unordered_set>

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"

#include "triton-shared/Analysis/MaskAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <iostream>
#define DEBUG_TYPE "triton-to-linked-npu"

using namespace mlir;
using namespace mlir::dicp::linked;

/// This function generates a series of `scf.for` loops for the given dimensions
/// in `loopDims`. Although the loops are created sequentially, nesting is
/// simulated by adjusting the insertion point to the body of the last created
/// loop. This allows the `bodyFunc` to be inserted into the innermost scope.
///
/// \param rewriter The MLIR OpBuilder used to create operations.
/// \param loc The source location information for debuggability.
/// \param target The memref value whose dimensions are being looped over.
/// \param loopDims An array of dimension indices to create loops for.
/// \param bodyFunc A callable that defines the operations to insert in the
/// innermost loop. It takes a SmallVector of induction variables (one per
/// loop).
///
template <typename Func>
static void createSimpleNestedLoops(OpBuilder &rewriter, Location loc,
                                    Value target, ArrayRef<int> loopDims,
                                    Func bodyFunc) {
  MemRefType type = cast<MemRefType>(target.getType());
  int rank = type.getRank();

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  llvm::SmallVector<scf::ForOp, 4> loops;
  llvm::SmallVector<Value, 4> ivs;

  for (int dim : loopDims) {
    Value ub;
    if (type.isDynamicDim(dim)) {
      ub = rewriter.create<memref::DimOp>(loc, target, dim).getResult();
    } else {
      ub = rewriter.create<arith::ConstantIndexOp>(loc, type.getDimSize(dim));
    }

    auto forOp = rewriter.create<scf::ForOp>(loc, zero, ub, one);
    rewriter.setInsertionPointToStart(forOp.getBody());
    loops.push_back(forOp);
    ivs.push_back(forOp.getInductionVar());
  }

  bodyFunc(ivs);

  if (!loops.empty()) {
    rewriter.setInsertionPointAfter(loops.front());
  }
}

bool ReduceConverter::isReductionOpSupported(Operation *redOp) const {
  return isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MaximumFOp,
             arith::MaxNumFOp, arith::MinimumFOp, arith::MinNumFOp,
             arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp,
             arith::AndIOp, arith::OrIOp, arith::XOrIOp>(redOp);
}

LogicalResult
ReduceConverter::convertToTargetOp(triton::ReduceOp op,
                                   typename triton::ReduceOp::Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  auto source = adaptor.getOperands().front();
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto elemType = sourceType.getElementType();
  auto resType = op.getResult().front().getType();
  auto loc = op.getLoc();
  auto reductionOps = this->getRedOps(op);

  // Reduction of arbitrary operations isn't supported because using the first
  // element across the reduction dimension requires us to iterate over a
  // subview that skips over each first element.
  if (!this->isReductionOpSupported(reductionOps.front())) {
    return rewriter.notifyMatchFailure(
        op, "Only support lowering reduction with single op and limited types "
            "of reducetion");
  }

  auto rop = reductionOps.front();
  auto axis = op.getAxis();
  auto isVectorReduce = sourceType.getRank() == 1;

  auto constantType = elemType;

  auto accBaseConstOp = this->getRedBaseConstOp(rewriter, rop, constantType);
  Value initTensor;

  if (isVectorReduce) {
    auto holder = rewriter.create<bufferization::AllocTensorOp>(
        loc, RankedTensorType::get({}, constantType), ValueRange{});
    initTensor = rewriter
                     .create<linalg::FillOp>(loc, accBaseConstOp.getResult(),
                                             holder.getResult())
                     .getResult(0);
  } else {
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, cast<RankedTensorType>(resType).getShape(), constantType);
    initTensor =
        rewriter.create<linalg::FillOp>(loc, accBaseConstOp.getResult(), init)
            .getResult(0);
  }

  Value finalResult =
      rewriter
          .create<linalg::ReduceOp>(
              loc, ValueRange{source}, ValueRange{initTensor},
              SmallVector<int64_t>{axis},
              [&](OpBuilder &opBuilder, Location loc, ValueRange inputs) {
                assert(inputs.size() == 2);
                Value result = this->getRedElement(inputs[0], inputs[1], loc,
                                                   rop, opBuilder, false);
                opBuilder.create<linalg::YieldOp>(loc, result);
              })
          .getResult(0);

  if (sourceType.getRank() == 1) {
    finalResult =
        rewriter.create<tensor::ExtractOp>(loc, constantType, finalResult);
  }

  rewriter.replaceOp(op, finalResult);
  return success();
}

static LogicalResult
addReduceWithIndexAttrIfNeeded(ConversionPatternRewriter &rewriter,
                               linalg::ReduceOp reduceOp) {
  // To verify whether the operation of the reduceOp is ReduceWithIndex
  // TODO: maybe a better way of judging?
  Block &body = reduceOp.getCombiner().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());

  auto yieldValue = yieldOp.getValues();
  if (yieldValue.size() == 0) {
    return failure();
  }

  const StringRef reduceRef = "reduce_mode";
  const StringRef tieBreakLeftRef = "tie_break_left";
  // INT
  // Composite predicate to pick index of min (or max) element have to be
  // written in following form: value1 < value2 or (value1 == value2 and index1
  // < index2) - for leftmost element (value1 == value2 and index1 < index2) or
  // value1 < value2 - for leftmost element value1 < value2 or (value1 == value2
  // and index1 > index2) - for rightmost element (value1 == value2 and index1 >
  // index2) or value1 < value2 - for rightmost element table below encodes all
  // possible cases of sequences of predicates for min/max and
  // leftmost/rightmost elements
  std::map<std::vector<arith::CmpIPredicate>,
           std::pair<std::string, std::string>>
      m{
          {{arith::CmpIPredicate::eq, arith::CmpIPredicate::sgt,
            arith::CmpIPredicate::sgt},
           {"max_with_index", "false"}},
          {{arith::CmpIPredicate::eq, arith::CmpIPredicate::sgt,
            arith::CmpIPredicate::slt},
           {"min_with_index", "false"}},
          {{arith::CmpIPredicate::eq, arith::CmpIPredicate::slt,
            arith::CmpIPredicate::sgt},
           {"max_with_index", "true"}},
          {{arith::CmpIPredicate::eq, arith::CmpIPredicate::slt,
            arith::CmpIPredicate::slt},
           {"min_with_index", "true"}},
          {{arith::CmpIPredicate::sgt, arith::CmpIPredicate::eq,
            arith::CmpIPredicate::sgt},
           {"max_with_index", "false"}},
          {{arith::CmpIPredicate::slt, arith::CmpIPredicate::eq,
            arith::CmpIPredicate::sgt},
           {"min_with_index", "false"}},
          {{arith::CmpIPredicate::sgt, arith::CmpIPredicate::eq,
            arith::CmpIPredicate::slt},
           {"max_with_index", "true"}},
          {{arith::CmpIPredicate::slt, arith::CmpIPredicate::eq,
            arith::CmpIPredicate::slt},
           {"min_with_index", "true"}},

          {{arith::CmpIPredicate::eq, arith::CmpIPredicate::ugt,
            arith::CmpIPredicate::ugt},
           {"max_with_index", "false"}},
          {{arith::CmpIPredicate::eq, arith::CmpIPredicate::ugt,
            arith::CmpIPredicate::ult},
           {"min_with_index", "false"}},
          {{arith::CmpIPredicate::eq, arith::CmpIPredicate::ult,
            arith::CmpIPredicate::ugt},
           {"max_with_index", "true"}},
          {{arith::CmpIPredicate::eq, arith::CmpIPredicate::ult,
            arith::CmpIPredicate::ult},
           {"min_with_index", "true"}},
          {{arith::CmpIPredicate::ugt, arith::CmpIPredicate::eq,
            arith::CmpIPredicate::ugt},
           {"max_with_index", "false"}},
          {{arith::CmpIPredicate::ult, arith::CmpIPredicate::eq,
            arith::CmpIPredicate::ugt},
           {"min_with_index", "false"}},
          {{arith::CmpIPredicate::ugt, arith::CmpIPredicate::eq,
            arith::CmpIPredicate::ult},
           {"max_with_index", "true"}},
          {{arith::CmpIPredicate::ult, arith::CmpIPredicate::eq,
            arith::CmpIPredicate::ult},
           {"min_with_index", "true"}},
      };

  std::vector<arith::CmpIPredicate> preds;
  using arith::CmpIPredicate;
  std::unordered_set<arith::CmpIPredicate> allowed{
      arith::CmpIPredicate::slt, arith::CmpIPredicate::sgt,
      arith::CmpIPredicate::eq, arith::CmpIPredicate::ult,
      arith::CmpIPredicate::ugt};
  // collect predicates under consideration
  for (auto it = body.begin(); it != body.end(); ++it) {
    if (auto op = dyn_cast<arith::CmpIOp>(*it)) {
      auto pred = op.getPredicate();
      if (allowed.find(pred) == allowed.end()) {
        continue;
      }
      preds.push_back(pred);
    }
  }
  // check if sequence of predicates matches any sequence for min/max
  // leftmost/rightmost
  if (m.find(preds) != m.end()) {
    auto [type, tie_break] = m[preds];
    reduceOp->setAttr(reduceRef, rewriter.getStringAttr(type));
    reduceOp->setAttr(tieBreakLeftRef, rewriter.getStringAttr(tie_break));
  }

  // FLOAT
  // For float case it's enough to check for OGT/OLT (comparison of elements)
  // and for sgt/slt (comparison for indices)
  std::string floatType;
  std::string tieBreakLeftFloat;
  for (auto it = body.begin(); it != body.end(); ++it) {
    if (auto op = dyn_cast<arith::CmpFOp>(*it)) {
      if (op.getPredicate() != arith::CmpFPredicate::OGT &&
          op.getPredicate() != arith::CmpFPredicate::OLT) {
        continue;
      }
      floatType = op.getPredicate() == arith::CmpFPredicate::OGT
                      ? "max_with_index"
                      : "min_with_index";
    }
    if (auto op = dyn_cast<arith::CmpIOp>(*it)) {
      if (op.getPredicate() != arith::CmpIPredicate::sgt &&
          op.getPredicate() != arith::CmpIPredicate::slt) {
        continue;
      }
      tieBreakLeftFloat =
          op.getPredicate() == arith::CmpIPredicate::sgt ? "false" : "true";
    }
  }
  if (!floatType.empty() && !tieBreakLeftFloat.empty()) {
    reduceOp->setAttr(reduceRef, rewriter.getStringAttr(floatType));
    reduceOp->setAttr(tieBreakLeftRef,
                      rewriter.getStringAttr(tieBreakLeftFloat));
  }

  return success();
}

LogicalResult ReduceConverter::convertToTargetOpExtended(
    triton::ReduceOp op, typename triton::ReduceOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto elemTypes = op.getElementTypes();

  auto valueResultType = dyn_cast<RankedTensorType>(op.getType(0));
  const auto isScalarReduce = valueResultType == nullptr;

  SmallVector<Value> outputs;
  for (auto i = 0; i < op.getResult().size() && i < elemTypes.size(); i++) {
    auto result = dyn_cast<RankedTensorType>(op.getType(i));
    SmallVector<int64_t> resultShape{
        isScalarReduce ? SmallVector<int64_t>{}
                       : SmallVector<int64_t>(result.getShape())};
    outputs.push_back(
        rewriter.create<tensor::EmptyOp>(loc, resultShape, elemTypes[i]));
  }

  auto linalgOp = rewriter.create<linalg::ReduceOp>(
      loc, adaptor.getOperands(), outputs,
      SmallVector<int64_t>{adaptor.getAxis()},
      [&](OpBuilder &b, Location loc, ValueRange inputs) {
        auto tritonReduceBlock = op.getBody();
        IRMapping mapping;
        mapping.map(tritonReduceBlock->getArguments(), inputs);

        for (auto &op : tritonReduceBlock->without_terminator()) {
          b.clone(op, mapping);
        }

        auto tritonYield = tritonReduceBlock->getTerminator();
        auto results =
            llvm::map_to_vector(tritonYield->getOperands(),
                                [&](Value val) { return mapping.lookup(val); });
        b.create<linalg::YieldOp>(loc, results);
      });

  if (failed(addReduceWithIndexAttrIfNeeded(rewriter, linalgOp))) {
    return rewriter.notifyMatchFailure(op, "meaningless reduce operation");
  }

  if (isScalarReduce) {
    SmallVector<Value> reduceResults;
    for (auto i = 0; i < linalgOp.getResults().size() && i < elemTypes.size();
         i++) {
      reduceResults.push_back(rewriter.create<tensor::ExtractOp>(
          loc, elemTypes[i], linalgOp.getResults()[i], ValueRange{}));
    }
    rewriter.replaceOp(op, reduceResults);
  } else {
    rewriter.replaceOp(op, linalgOp);
  }
  return success();
}

bool ScanConverter::isReductionOpSupported(Operation *redOp) const {
  return isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp>(redOp);
}

LogicalResult
ScanConverter::convertToTargetOp(triton::ScanOp op,
                                 typename triton::ScanOp::Adaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto reductionOps = this->getRedOps(op);
  if (reductionOps.empty()) {
    return rewriter.notifyMatchFailure(op,
                                       "No reduction op found in scan body");
  }

  bool reverse = op.getReverse();
  if (reverse) {
    op.emitError("reverse=True is not yet supported for scan op");
    return failure();
  }

  llvm::SmallString<64> funcName;
  auto rop = reductionOps.front();
  if (this->isReductionOpSupported(reductionOps.front())) {
    if (isa<arith::AddFOp, arith::AddIOp>(rop)) {
      funcName = "triton_cumsum";
    } else if (isa<arith::MulFOp, arith::MulIOp>(rop)) {
      funcName = "triton_cumprod";
    }

    auto moduleOp = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPoint(moduleOp.getBody(),
                               std::prev(moduleOp.getBody()->end()));

    auto loc = op.getLoc();
    auto src = adaptor.getOperands().front();
    auto resTy = op.getResult().front().getType();
    auto libFnType = rewriter.getFunctionType(
        {src.getType(), rewriter.getI32Type(), rewriter.getI1Type()}, {resTy});
    auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);

    SymbolTable symTab(moduleOp);
    auto maybePrintFuncNameAttr = symTab.renameToUnique(funcOp, {&symTab});
    if (failed(maybePrintFuncNameAttr)) {
      return op->emitError(
          "failed to create a unique func name for device_print");
    }
    SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

    rewriter.setInsertionPoint(op);
    auto scanAxis = op.getAxis();
    auto scanReverse = op.getReverse();
    Value axis = rewriter.create<arith::ConstantIntOp>(loc, scanAxis, 32);
    Value reverseVal =
        rewriter.create<arith::ConstantIntOp>(loc, scanReverse, 1);
    auto callOp = rewriter.create<func::CallOp>(
        loc, funcOp.getSymNameAttr(), TypeRange({resTy}),
        ValueRange({src, axis, reverseVal}));

    rewriter.replaceOp(op, callOp);

    return success();
  } else {
    // This branch is the associative_scan op.
    auto loc = op.getLoc();

    Value scanInput = op.getOperand(0);

    scanInput.dump();

    for (Value operand : op->getOperands()) {
      operand.dump();
    }

    auto srcType = mlir::dyn_cast<RankedTensorType>(scanInput.getType());
    if (!srcType) {
      return rewriter.notifyMatchFailure(
          op, "Expected RankedTensorType input for associative_scan");
    }

    auto elementType = srcType.getElementType();
    auto shape = srcType.getShape();
    int rank = shape.size();
    int axis = op.getAxis();

    if (axis < 0 || axis >= rank) {
      return rewriter.notifyMatchFailure(op, "Invalid scan axis: " +
                                                 std::to_string(axis));
    }

    if (op->getNumRegions() < 1 || op->getRegion(0).empty()) {
      return rewriter.notifyMatchFailure(op, "Missing combine region");
    }

    OpBuilder::InsertionGuard guard(rewriter);

    auto memrefType = MemRefType::get(shape, elementType);
    Value inputMemRef =
        rewriter.create<bufferization::ToBufferOp>(loc, memrefType, scanInput);
    Value outputMemRef = rewriter.create<memref::AllocOp>(loc, memrefType);

    auto processDimension = [&](ArrayRef<Value> baseIdxsArray) {
      llvm::SmallVector<Value> baseIdxs(baseIdxsArray.begin(),
                                        baseIdxsArray.end());
      llvm::SmallVector<Value> firstIdx = baseIdxs;
      if (axis <= firstIdx.size()) {
        firstIdx.insert(firstIdx.begin() + axis,
                        rewriter.create<arith::ConstantIndexOp>(loc, 0));
      } else {
        firstIdx.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      }

      Value firstVal =
          rewriter.create<memref::LoadOp>(loc, inputMemRef, firstIdx);
      rewriter.create<memref::StoreOp>(loc, firstVal, outputMemRef, firstIdx);

      Value axisSize =
          rewriter.create<memref::DimOp>(loc, inputMemRef, axis).getResult();
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                                 axisSize, one);
      auto ifOp = rewriter.create<scf::IfOp>(loc, cmp, false);

      // Create a loop only when the axis size is greater than 1.
      rewriter.setInsertionPointToStart(ifOp.thenBlock());

      auto forOp = rewriter.create<scf::ForOp>(loc, one, axisSize, one);
      rewriter.setInsertionPointToStart(forOp.getBody());

      Value k = forOp.getInductionVar();
      llvm::SmallVector<Value> currIdx = baseIdxs;
      if (axis <= currIdx.size()) {
        currIdx.insert(currIdx.begin() + axis, k);
      } else {
        currIdx.push_back(k);
      }

      Value km1 = rewriter.create<arith::SubIOp>(loc, k, one);
      llvm::SmallVector<Value> prevIdx = baseIdxs;
      if (axis <= prevIdx.size()) {
        prevIdx.insert(prevIdx.begin() + axis, km1);
      } else {
        prevIdx.push_back(km1);
      }

      Value currentVal =
          rewriter.create<memref::LoadOp>(loc, inputMemRef, currIdx);
      Value prevResult =
          rewriter.create<memref::LoadOp>(loc, outputMemRef, prevIdx);

      Region &combineRegion = op->getRegion(0);
      Block &combineBlock = combineRegion.front();
      IRMapping mapping;
      mapping.map(combineBlock.getArgument(0), prevResult);
      mapping.map(combineBlock.getArgument(1), currentVal);

      for (Operation &innerOp : combineBlock.without_terminator()) {
        rewriter.clone(innerOp, mapping);
      }

      Operation *yieldOp = combineBlock.getTerminator();
      Value resultVal = mapping.lookup(yieldOp->getOperand(0));

      rewriter.create<memref::StoreOp>(loc, resultVal, outputMemRef, currIdx);

      rewriter.setInsertionPointAfter(ifOp);
    };

    // Constructing loops for non-scanning dimensions
    llvm::SmallVector<int> nonScanDims;
    for (int i = 0; i < rank; ++i) {
      if (i != axis)
        nonScanDims.push_back(i);
    }

    createSimpleNestedLoops(rewriter, loc, outputMemRef, nonScanDims,
                            processDimension);

    rewriter.setInsertionPointAfter(op);

    MemRefType memrefTy = cast<MemRefType>(outputMemRef.getType());
    Type tensorTy =
        RankedTensorType::get(memrefTy.getShape(), memrefTy.getElementType());
    Value outputTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorTy, outputMemRef, true, true);
    rewriter.replaceOp(op, outputTensor);
    return success();
  }
}

LogicalResult ScanConverter::convertToTargetOpExtended(
    triton::ScanOp op, typename triton::ScanOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  bool reverse = op.getReverse();
  if (reverse) {
    return op.emitError(
        "reverse=True is not yet supported for extended scan op");
  }

  // 1. Extract all input tensors (supports multiple inputs)
  auto operands = op->getOperands();
  if (operands.empty()) {
    return rewriter.notifyMatchFailure(op,
                                       "No input operands for extended scan");
  }

  // 2. Validate all inputs are of RankedTensorType
  llvm::SmallVector<RankedTensorType> inputTensTypes;
  for (auto operand : operands) {
    auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
    if (!tensorTy) {
      return rewriter.notifyMatchFailure(op,
                                         "All inputs must be RankedTensorType");
    }
    inputTensTypes.push_back(tensorTy);
  }

  // 3. Validate all input tensors have the same shape (scan operation requires
  // matching input dimensions)
  auto baseShape = inputTensTypes[0].getShape();
  int rank = baseShape.size();
  int axis = op.getAxis();
  if (axis < 0 || axis >= rank) {
    return rewriter.notifyMatchFailure(op, "Invalid scan axis: " +
                                               std::to_string(axis));
  }
  for (size_t i = 1; i < inputTensTypes.size(); ++i) {
    if (inputTensTypes[i].getShape() != baseShape) {
      return rewriter.notifyMatchFailure(op,
                                         "All inputs must have the same shape");
    }
  }

  // 4. Prepare MemRefs for multiple inputs/outputs
  llvm::SmallVector<Value> inputMemRefs;
  llvm::SmallVector<Value> outputMemRefs;
  llvm::SmallVector<MemRefType> memRefTypes;
  for (size_t i = 0; i < inputTensTypes.size(); ++i) {
    auto &tensorTy = inputTensTypes[i];
    auto memRefTy =
        MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
    memRefTypes.push_back(memRefTy);
    // Convert input tensors to MemRefs
    inputMemRefs.push_back(
        rewriter.create<bufferization::ToBufferOp>(loc, memRefTy, operands[i]));
    // Allocate MemRefs for outputs
    outputMemRefs.push_back(rewriter.create<memref::AllocOp>(loc, memRefTy));
  }

  // 5. Define scanning logic for multiple inputs/outputs
  LogicalResult loopResult = success();
  auto processDimension = [&](ArrayRef<Value> baseIdxsArray) {
    llvm::SmallVector<Value> baseIdxs(baseIdxsArray.begin(),
                                      baseIdxsArray.end());
    llvm::SmallVector<Value> firstIdx = baseIdxs;
    // Insert start index (0) for the scan axis
    if (axis <= firstIdx.size()) {
      firstIdx.insert(firstIdx.begin() + axis,
                      rewriter.create<arith::ConstantIndexOp>(loc, 0));
    } else {
      firstIdx.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    // 5.1 Process the first element: directly copy multiple inputs to multiple
    // outputs (initialize cumulative results)
    for (size_t i = 0; i < inputMemRefs.size(); ++i) {
      Value firstVal =
          rewriter.create<memref::LoadOp>(loc, inputMemRefs[i], firstIdx);
      rewriter.create<memref::StoreOp>(loc, firstVal, outputMemRefs[i],
                                       firstIdx);
    }

    // 5.2 Calculate the size of the scan axis; create a loop only if the axis
    // size > 1
    Value axisSize =
        rewriter.create<memref::DimOp>(loc, inputMemRefs[0], axis).getResult();
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                               axisSize, one);
    auto ifOp = rewriter.create<scf::IfOp>(loc, cmp, false);

    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    // Loop variable k: ranges from 1 to axisSize-1
    auto forOp = rewriter.create<scf::ForOp>(loc, one, axisSize, one);
    rewriter.setInsertionPointToStart(forOp.getBody());
    Value k = forOp.getInductionVar();

    // 5.3 Calculate current index (k) and previous index (k-1)
    llvm::SmallVector<Value> currIdx = baseIdxs;
    if (axis <= currIdx.size()) {
      currIdx.insert(currIdx.begin() + axis, k);
    } else {
      currIdx.push_back(k);
    }
    Value km1 = rewriter.create<arith::SubIOp>(loc, k, one);
    llvm::SmallVector<Value> prevIdx = baseIdxs;
    if (axis <= prevIdx.size()) {
      prevIdx.insert(prevIdx.begin() + axis, km1);
    } else {
      prevIdx.push_back(km1);
    }

    // 5.4 Load current elements and previous cumulative results
    llvm::SmallVector<Value> currentVals;
    llvm::SmallVector<Value> prevResults;
    for (size_t i = 0; i < inputMemRefs.size(); ++i) {
      currentVals.push_back(
          rewriter.create<memref::LoadOp>(loc, inputMemRefs[i], currIdx));
      prevResults.push_back(
          rewriter.create<memref::LoadOp>(loc, outputMemRefs[i], prevIdx));
    }

    // 5.5 Bind parameters for custom reduction logic
    Region &combineRegion = op->getRegion(0);
    if (combineRegion.empty()) {
      op->emitError("Missing combine region in extended scan");
      loopResult = failure();
      return;
    }
    Block &combineBlock = combineRegion.front();
    // Validate that the number of reduction region arguments matches (number of
    // previous results + number of current elements)
    if (combineBlock.getNumArguments() != 2 * inputMemRefs.size()) {
      op->emitError("Combine region arguments mismatch with input count");
      loopResult = failure();
      return;
    }
    IRMapping mapping;
    for (size_t i = 0; i < inputMemRefs.size(); ++i) {
      // Bind previous results (previous value of the i-th output) to the i-th
      // argument of the reduction region
      mapping.map(combineBlock.getArgument(i), prevResults[i]);
      // Bind current elements (current value of the i-th input) to the i+N-th
      // argument of the reduction region (N is the number of inputs)
      mapping.map(combineBlock.getArgument(i + inputMemRefs.size()),
                  currentVals[i]);
    }

    // 5.6 Clone all operations within the reduction region
    for (Operation &innerOp : combineBlock.without_terminator()) {
      rewriter.clone(innerOp, mapping);
    }

    // 5.7 Extract reduction results and store them in outputMemRef
    Operation *yieldOp = combineBlock.getTerminator();
    if (yieldOp->getNumOperands() != outputMemRefs.size()) {
      op->emitError("Combine region returns mismatch with output count");
      loopResult = failure();
      return;
    }
    for (size_t i = 0; i < outputMemRefs.size(); ++i) {
      Value resultVal = mapping.lookup(yieldOp->getOperand(i));
      rewriter.create<memref::StoreOp>(loc, resultVal, outputMemRefs[i],
                                       currIdx);
    }

    rewriter.setInsertionPointAfter(ifOp);
  };

  // 6. Generate nested loops for non-scan dimensions
  llvm::SmallVector<int> nonScanDims;
  for (int i = 0; i < rank; ++i) {
    if (i != axis)
      nonScanDims.push_back(i);
  }
  createSimpleNestedLoops(rewriter, loc, outputMemRefs[0], nonScanDims,
                          processDimension);

  if (failed(loopResult)) {
    return failure();
  }

  // 7. Convert multiple output MemRefs back to tensors and replace the original
  // tt.scan operation
  llvm::SmallVector<Value> outputTensors;
  for (auto outputMemRef : outputMemRefs) {
    MemRefType memrefTy = cast<MemRefType>(outputMemRef.getType());
    Type tensorTy =
        RankedTensorType::get(memrefTy.getShape(), memrefTy.getElementType());
    outputTensors.push_back(rewriter.create<bufferization::ToTensorOp>(
        loc, tensorTy, outputMemRef, true, true));
  }
  rewriter.replaceOp(op, outputTensors);

  return success();
}

LogicalResult DevicePrintConverter::matchAndRewrite(
    triton::PrintOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));
  SmallVector<Type, 4> inputTypes;
  for (auto arg : op.getArgs()) {
    inputTypes.push_back(arg.getType());
  }
  auto libFnType = rewriter.getFunctionType(inputTypes, {});
  auto funcOp =
      rewriter.create<func::FuncOp>(op.getLoc(), printFuncNameBase, libFnType);
  SymbolTable symTab(moduleOp);
  auto maybePrintFuncNameAttr = symTab.renameToUnique(funcOp, {&symTab});
  if (failed(maybePrintFuncNameAttr)) {
    return op->emitError(
        "failed to create a unique func name for device_print");
  }
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  auto prefixAttr = op.getPrefixAttr();
  funcOp->setAttr(prefixAttrName, prefixAttr);
  auto hexAttr = op.getHexAttr();
  funcOp->setAttr(hexAttrName, hexAttr);

  rewriter.setInsertionPoint(op);
  rewriter.create<func::CallOp>(op.getLoc(), funcOp, op.getArgs());

  rewriter.eraseOp(op);
  return success();
}

LogicalResult DeviceAssertConverter::matchAndRewrite(
    triton::AssertOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto msgAttr = op.getMessageAttr();
  // Filter out automatically inserted assert ops
  if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(msgAttr)) {
    llvm::StringRef msg = strAttr.getValue();
    if (msg.contains("overflow detected for operation")) {
      rewriter.eraseOp(op);
      return success();
    }
  }

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));
  auto conditionType = op.getCondition().getType();

  auto libFnType = rewriter.getFunctionType({conditionType}, {});
  auto funcOp =
      rewriter.create<func::FuncOp>(op.getLoc(), printFuncNameBase, libFnType);
  mlir::SymbolTable symTab(moduleOp);
  auto maybePrintFuncNameAttr = symTab.renameToUnique(funcOp, {&symTab});
  if (failed(maybePrintFuncNameAttr)) {
    return op->emitError(
        "failed to create a unique func name for device_assert");
  }
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  funcOp->setAttr(msgAttrName, msgAttr);

  rewriter.setInsertionPoint(op);
  rewriter.create<func::CallOp>(op.getLoc(), funcOp,
                                ValueRange{op.getCondition()});

  rewriter.eraseOp(op);
  return success();
}