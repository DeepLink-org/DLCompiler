#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"
#include "dicp/Dialect/LinalgExt/Transforms/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

#define DEBUG_TYPE "linalg-if-to-select"

using namespace mlir;
using namespace dicp;
using namespace LinalgExt;

namespace mlir::dicp::LinalgExt {
#define GEN_PASS_DEF_LINALGIFTOSELECT
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace mlir::dicp::LinalgExt

namespace {

// --- Lift scalar arith.select (inside linalg.generic) to
// tensor arith.select before the linalg.generic and add it as an extra
// `ins` operand of generic. ---
struct LiftScalarSelectToTensorPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp->hasAttr("ExtractedLoadOrStore"))
      return rewriter.notifyMatchFailure(
          linalgOp, "missing 'ExtractedLoadOrStore' attribute");

    // Find a scalar arith.select in the body whose operands are either
    // block arguments or constants.
    Block *body = linalgOp.getBody();
    arith::SelectOp scalarSelect;
    for (Operation &op : body->getOperations()) {
      if (auto s = dyn_cast<arith::SelectOp>(op)) {
        // Ensure it is scalar (not tensor)
        if (isa<TensorType, MemRefType>(s.getType()))
          continue;
        // Only consider select whose operands (true/false/cond) are
        // either block args or loop-invariant
        Value t = s.getTrueValue();
        Value f = s.getFalseValue();
        Value c = s.getCondition();

        // 'v' is allowed if it's a BlockArgument OR defined outside the
        // linalg op's body (making it a loop-invariant).
        auto isAllowed = [&](Value v) -> bool {
          // Case 1: Block argument (e.g., %arg0)
          if (isa<BlockArgument>(v))
            return true;

          // Case 2: Any SSA value defined *outside* the linalg body.
          // This includes arith.constant and other loop-invariant values.
          Operation *defOp = v.getDefiningOp();
          if (defOp && defOp->getBlock() != body) {
            return true;
          }

          return false;
        };

        if (isAllowed(t) && isAllowed(f) && isAllowed(c)) {
          scalarSelect = s;
          break;
        }
      }
    }

    if (!scalarSelect)
      return rewriter.notifyMatchFailure(
          linalgOp,
          "could not find a liftable scalar arith.select in the body");

    Location loc = linalgOp.getLoc();

    // Determine the tensor shape to materialize the lifted operands.
    // Prefer first input's shape.
    if (linalgOp.getNumDpsInputs() == 0)
      return rewriter.notifyMatchFailure(
          linalgOp, "linalg.generic has no input operands to determine shape");

    Value firstInput = linalgOp.getDpsInputOperand(0)->get();
    auto firstTy = dyn_cast<RankedTensorType>(firstInput.getType());
    if (!firstTy || !firstTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          linalgOp, "first input is not a RankedTensorType or does not have "
                    "static shape");
    ArrayRef<int64_t> shape = firstTy.getShape();

    // Utility to materialize a tensor from either a constant scalar or a
    // block-argument-mapped input tensor.
    auto materializeTensor = [&](Value v, Type desiredElemType) -> Value {
      // Case 1: BlockArgument (maps to an existing input tensor)
      if (auto ba = dyn_cast<BlockArgument>(v)) {
        unsigned argNo = ba.getArgNumber();
        // block args are ordered as inputs then outputs.
        if (argNo >= linalgOp.getNumDpsInputs()) {
          // Block argument refers to an output tensor, which is not an allowed
          // input for select in this pattern.
          return nullptr;
        }
        Value tensorOperand = linalgOp.getDpsInputOperand(argNo)->get();
        // TODO: Ensure it has same element type or cast. For now, just return.
        return tensorOperand;
      }

      // Case 2: Loop-invariant SSA scalar value (from outside)
      // This now correctly includes arith.constant and any other op.
      Operation *defOp = v.getDefiningOp();
      if (defOp && defOp->getBlock() != body) {
        // 'v' is the loop-invariant scalar value. Broadcast it.
        Value empty =
            rewriter.create<tensor::EmptyOp>(loc, shape, desiredElemType);

        // 'v' is the scalar to fill with.
        auto fill = rewriter.create<linalg::FillOp>(loc, v, empty);
        return fill.getResult(0);
      }

      // 'v' is not a BlockArgument and not defined outside.
      return nullptr;
    };

    // Materialize cond tensor, true tensor, false tensor
    Value condTensor = materializeTensor(scalarSelect.getCondition(),
                                         scalarSelect.getCondition().getType());
    if (!condTensor)
      return rewriter.notifyMatchFailure(
          linalgOp, "failed to materialize condition tensor");

    Type trueElemTy = scalarSelect.getTrueValue().getType();
    Value trueTensor =
        materializeTensor(scalarSelect.getTrueValue(), trueElemTy);
    if (!trueTensor)
      return rewriter.notifyMatchFailure(
          linalgOp, "failed to materialize true-value tensor");

    Type falseElemTy = scalarSelect.getFalseValue().getType();
    Value falseTensor =
        materializeTensor(scalarSelect.getFalseValue(), falseElemTy);
    if (!falseTensor)
      return rewriter.notifyMatchFailure(
          linalgOp, "failed to materialize false-value tensor");

    // Create the lifted tensor select before the linalg.generic
    rewriter.setInsertionPoint(linalgOp);
    RankedTensorType condType =
        dyn_cast<RankedTensorType>(condTensor.getType());
    RankedTensorType trueType =
        dyn_cast<RankedTensorType>(trueTensor.getType());
    if (!condType || !trueType)
      return rewriter.notifyMatchFailure(
          linalgOp, "materialized tensor type is not a RankedTensorType");

    // arith.select signature: (cond, true, false) -> result type (trueType)
    Value tensorSelect = rewriter.create<arith::SelectOp>(
        loc, /*result type*/ trueTensor.getType(), condTensor, trueTensor,
        falseTensor);

    // --- Fix: Rebuild LinalgOp ---
    unsigned numInputs = linalgOp.getNumDpsInputs();
    unsigned numOutputs = linalgOp.getNumDpsInits();

    // 1. Prepare new operand list
    auto newIns = llvm::to_vector<4>(
        llvm::map_range(linalgOp.getDpsInputOperands(),
                        [](OpOperand *opnd) { return opnd->get(); }));

    newIns.push_back(
        tensorSelect); // Add the new tensor select as the last input
    ValueRange newOuts = linalgOp.getDpsInits();

    // 2. Prepare new indexing_maps attribute
    SmallVector<Attribute> newMapAttrs;
    for (Attribute a : linalgOp.getIndexingMaps())
      newMapAttrs.push_back(a);

    // Add an identity map for the new input (reusing the first map or creating
    // new)
    if (!linalgOp.getIndexingMaps().empty()) {
      newMapAttrs.push_back(linalgOp.getIndexingMaps()[0]);
    } else {
      // Fallback: create an identity map if original maps were empty
      // This is necessary because we checked for firstInput above.
      if (!firstTy)
        return rewriter.notifyMatchFailure(
            linalgOp, "cannot determine rank for identity map fallback");
      newMapAttrs.push_back(
          AffineMapAttr::get(AffineMap::getMultiDimIdentityMap(
              firstTy.getRank(), rewriter.getContext())));
    }

    ArrayAttr newMapsAttr = rewriter.getArrayAttr(newMapAttrs);

    // 3. Prepare new operandSegmentSizes attribute
    auto newSegmentSizes = rewriter.getDenseI32ArrayAttr(
        {static_cast<int32_t>(numInputs + 1), // New number of Inputs
         static_cast<int32_t>(numOutputs)});

    // 4. Create the new linalg.generic op and rebuild the region using
    // bodyBuilder
    auto newLinalgOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/linalgOp.getResultTypes(),
        /*inputs=*/newIns,
        /*outputs=*/newOuts,
        /*indexing_maps=*/newMapsAttr,
        /*iterator_types=*/linalgOp.getIteratorTypes(),
        /*doc=*/nullptr,
        /*library_call=*/nullptr,
        /*bodyBuilder=*/
        [&](OpBuilder &b, Location bodyLoc, ValueRange newBlockArgs) {
          // newBlockArgs now contains (old_ins, new_tensor_select_arg,
          // old_outs)
          IRMapping bvm;
          unsigned oldIdx = 0;
          unsigned newIdx = 0;

          // Map old input block args
          for (oldIdx = 0; oldIdx < numInputs; ++oldIdx, ++newIdx) {
            bvm.map(linalgOp.getBody()->getArgument(oldIdx),
                    newBlockArgs[newIdx]);
          }

          // This is the new block arg corresponding to new_tensor_select
          Value newArgForTensorSelect = newBlockArgs[newIdx++];

          // Map old output block args
          for (; oldIdx < linalgOp.getBody()->getNumArguments();
               ++oldIdx, ++newIdx) {
            bvm.map(linalgOp.getBody()->getArgument(oldIdx),
                    newBlockArgs[newIdx]);
          }

          // Clone the linalgOp body, skipping the old scalarSelect
          for (Operation &op : linalgOp.getBody()->without_terminator()) {
            // Check if this op is the scalarSelect we are replacing
            if (auto oldSel = dyn_cast<arith::SelectOp>(op)) {
              if (oldSel == scalarSelect) {
                // Yes! Map the result of the old select to the new block arg
                bvm.map(oldSel.getResult(), newArgForTensorSelect);
                // Do not clone this op
                continue;
              }
            }
            // Clone all other ops
            b.clone(op, bvm);
          }

          // Clone the terminator (linalg.yield)
          b.clone(*linalgOp.getBody()->getTerminator(), bvm);
        });

    // 5. Copy other attributes from linalgOp (e.g., "ExtractedLoadOrStore")
    for (const auto &attr : linalgOp->getAttrs()) {
      if (attr.getName() == linalgOp.getOperandSegmentSizesAttrName() ||
          attr.getName() == linalgOp.getIndexingMapsAttrName()) {
        continue;
      }
      newLinalgOp->setAttr(attr.getName(), attr.getValue());
    }

    // 6. Replace the old op results with the new op results
    rewriter.replaceOp(linalgOp, newLinalgOp.getResults());

    return success();
  }
};

// --- If the linalg.yield returns a scalar select that chooses
// between a constant and a scalar op inside the region, lift the select to
// after the linalg.generic as a tensor select. ---
struct LiftYieldSelectOutPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp->hasAttr("ExtractedLoadOrStore"))
      return rewriter.notifyMatchFailure(
          linalgOp, "missing 'ExtractedLoadOrStore' attribute");

    Block *body = linalgOp.getBody();
    auto linalgYield =
        mlir::dyn_cast_or_null<linalg::YieldOp>(body->getTerminator());
    if (!linalgYield || linalgYield.getNumOperands() == 0)
      return rewriter.notifyMatchFailure(
          linalgOp,
          "body terminator is not a linalg.yield or yields no values");

    Value yielded = linalgYield.getOperand(0);
    auto sel = yielded.getDefiningOp<arith::SelectOp>();
    if (!sel)
      return rewriter.notifyMatchFailure(
          linalgOp, "yielded value is not an arith.select op");

    // Conditions: select is single-use (used only by linalg.yield), condition
    // comes from a block arg, and one operand is a constant while the other is
    // a scalar op inside the region with one use.
    if (!llvm::hasSingleElement(sel->getUsers()))
      return rewriter.notifyMatchFailure(
          linalgOp,
          "arith.select has more than one user (not only linalg.yield)");

    Value cond = sel.getCondition();
    if (!isa<BlockArgument>(cond))
      return rewriter.notifyMatchFailure(
          linalgOp, "select condition is not a block argument");

    rewriter.setInsertionPointAfter(linalgOp);

    Value trueV = sel.getTrueValue();
    Value falseV = sel.getFalseValue();

    // Identify which is constant and which is an internal op
    Value constScalar;
    Operation *internalOp = nullptr;
    if (trueV.getDefiningOp<arith::ConstantOp>()) {
      constScalar = trueV;
      internalOp = falseV.getDefiningOp();
    } else if (falseV.getDefiningOp<arith::ConstantOp>()) {
      constScalar = falseV;
      internalOp = trueV.getDefiningOp();
    } else {
      return rewriter.notifyMatchFailure(
          linalgOp, "neither select operand is an arith.constant op");
    }

    if (!internalOp)
      return rewriter.notifyMatchFailure(
          linalgOp, "internal operand of select is not a defined operation");

    // internalOp must be single-use and produce a scalar compatible with yield
    if (!internalOp->hasOneUse())
      return rewriter.notifyMatchFailure(linalgOp,
                                         "internal op has more than one use");
    if (internalOp->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          linalgOp, "internal op does not have exactly one result");

    Type scalarTy = internalOp->getResult(0).getType();
    if (isa<TensorType>(scalarTy))
      return rewriter.notifyMatchFailure(
          linalgOp, "internal op result is a TensorType (expected scalar)");

    // Replace yield's operand with the internalOp's result so that the generic
    // returns the "internal" values instead of the select results.
    linalgYield->setOperand(0, internalOp->getResult(0));

    // After the linalg.generic, create a tensor constant from constScalar and
    // then a tensor select between the generic's result tensor and the
    // constant tensor using the cond input tensor.
    Location loc = linalgOp.getLoc();

    // Determine result type (assume first result) and element type
    if (linalgOp.getOperation()->getNumResults() == 0)
      return rewriter.notifyMatchFailure(linalgOp,
                                         "linalg.generic op has no results");

    Value genericResult = linalgOp.getResult(0);
    auto resultTy = dyn_cast<RankedTensorType>(genericResult.getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(
          linalgOp, "first result of linalg.generic is not a RankedTensorType");

    ArrayRef<int64_t> shape = resultTy.getShape();
    Type elemTy = resultTy.getElementType();

    // materialize tensor from scalar constant
    Value empty = rewriter.create<tensor::EmptyOp>(loc, shape, elemTy);
    // scalar constant
    auto constOp = constScalar.getDefiningOp<arith::ConstantOp>();
    if (!constOp)
      return rewriter.notifyMatchFailure(
          linalgOp, "constant scalar is not a valid arith.constant op");

    Value scalarConst =
        rewriter.create<arith::ConstantOp>(loc, elemTy, constOp.getValueAttr());
    Value filled =
        rewriter.create<linalg::FillOp>(loc, scalarConst, empty).getResult(0);

    // cond tensor: map block argument to corresponding input tensor
    auto condBA = cast<BlockArgument>(cond);
    unsigned condArgNo = condBA.getArgNumber();
    if (condArgNo >= linalgOp.getNumDpsInputs())
      return rewriter.notifyMatchFailure(
          linalgOp,
          "condition block argument index is out of bounds for dps_inputs");

    Value condTensor = linalgOp.getDpsInputOperand(condArgNo)->get();

    // Insert the select after the generic op
    Value tensorSelect = rewriter.create<arith::SelectOp>(
        loc, /*result type*/ genericResult.getType(), condTensor, genericResult,
        filled);

    // Replace uses of the generic result (outside the generic) with the
    // tensorSelect result. Do not replace uses inside the generic (there
    // shouldn't be any, but guard anyway).
    genericResult.replaceAllUsesExcept(tensorSelect,
                                       tensorSelect.getDefiningOp());

    return success();
  }
};

/**
 * @brief Converts scf.if within linalg.generic to arith.select.
 *
 * This pattern matches an scf.if inside a linalg.generic op.
 *
 * Requirements (Request 1 & 2):
 * 1. The if op must be inside a linalg.generic op.
 * 2. The linalg.generic op must have all "parallel" iterators and all
 * "identity" maps.
 * 3. All operations within the if regions (then/else) must produce scalar
values.  * 4. If the if op has results and an else block: the else block must
contain  * only one scf.yield whose yielded value is a constant.  * 5. If the if
op has no results: the else block must be empty (or contain  * only scf.yield).
 *
 * Conversion Logic:
 * 1. Iterate over all operations (op) in the `then` block.
 * 2. For each operand (val) of the op, check if val is a block argument (arg)
 * of the linalg.generic.
 * 3. If it is and the arg hasn't been mapped yet:
 * a. Create a zero constant %zero (of the same type as arg).
 * b. Create %sel_arg = arith.select %cond, %arg, %zero {recheck}.
 * c. Store the mapping { %arg -> %sel_arg } in IRMapping (bvm).
 * 4. Clone the body of the `then` block (excluding the terminator) before
 * the ifOp using bvm for value mapping.
 * 5. (Case A: ifOp has results, see Example 1)
 * a. Get the `then` block's yield value (%then_val) and the `else` block's
 * constant yield value (%else_const).
 * b. Find the mapped value for %then_val in bvm (i.e., %cloned_then_val).
 * c. Create the final select:
 * `%final_sel = arith.select %cond, %cloned_then_val, %else_const {recheck}`
 * d. Replace all uses of the ifOp with %final_sel and erase the ifOp.
 * 6. (Case B: ifOp has no results, see Example 2)
 * a. Erase the ifOp directly (since the `then` block's body has been cloned).
 */
struct LinalgIfToSelectPattern : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const final {

    // --- 0. Find the parent linalg.generic operation ---
    auto linalgOp = ifOp->getParentOfType<linalg::GenericOp>();
    if (!linalgOp) {
      // This scf.if is not inside a linalg.generic, match fails
      return failure();
    }
    Block *linalgBody = linalgOp.getBody();

    // --- 1. Check linalg.generic constraints (Request 2) ---
    // Linalg op must have identity maps
    if (!llvm::all_of(linalgOp.getIteratorTypesArray(),
                      linalg::isParallelIterator)) {
      return rewriter.notifyMatchFailure(
          linalgOp, "Failed match: expected all parallel iterators.");
    }
    for (AffineMap map : linalgOp.getIndexingMapsArray()) {
      if (!map.isIdentity()) {
        return rewriter.notifyMatchFailure(
            linalgOp, "Failed match: expected identity maps.");
      }
    }

    // --- 2. Check scf.if constraints ---
    // Check: all results of ops in the if are scalar
    bool allScalars = true;
    ifOp.walk([&](Operation *innerOp) {
      // Skip terminators (scf.yield) and the IfOp itself
      if (isa<scf::YieldOp>(innerOp) || innerOp == ifOp.getOperation())
        return;

      for (Value res : innerOp->getResults()) {
        if (isa<TensorType, MemRefType>(res.getType())) {
          allScalars = false;
        }
      }
    });
    if (!allScalars) {
      return rewriter.notifyMatchFailure(
          linalgOp, "Failed match: if region contains non-scalar ops.");
    }

    // --- 3. Get condition, location, and set insertion point ---
    Location loc = ifOp.getLoc();
    Value cond = ifOp.getCondition();
    // Set insertion point *before* the ifOp
    rewriter.setInsertionPoint(ifOp);

    // --- 4. Create arith.select only for linalg block args *used* in 'then'
    // block ---
    IRMapping bvm;
    Block *thenBlock = ifOp.thenBlock();

    // Iterate over ops in 'then' block
    for (Operation &op : thenBlock->without_terminator()) {
      // Iterate over all operands of the op
      for (Value val : op.getOperands()) {
        auto ba = dyn_cast<BlockArgument>(val);
        // Check: 1. Is a block argument
        //        2. Belongs to linalgOp (not a nested op)
        //        3. Select has not been created for this block argument yet
        if (ba && ba.getOwner() == linalgBody && !bvm.contains(ba)) {
          Type argType = ba.getType();
          // Create %false_value (zero constant)
          Value falseValue;
          if (isa<IndexType>(argType)) {
            falseValue = rewriter.create<arith::ConstantIndexOp>(loc, 0);

          } else if (auto intTy = dyn_cast<IntegerType>(argType)) {
            falseValue = rewriter.create<arith::ConstantOp>(
                loc, argType, rewriter.getIntegerAttr(argType, 0));

          } else if (auto floatTy = dyn_cast<FloatType>(argType)) {
            falseValue = rewriter.create<arith::ConstantOp>(
                loc, argType, rewriter.getFloatAttr(argType, 0.0));

          } else {
            return rewriter.notifyMatchFailure(
                linalgOp,
                "Failed match: unsupported block arg type for zero init.");
            return failure();
          }

          // Create arith.select cond, v, false_value
          Value newSel = rewriter.create<arith::SelectOp>(loc, /*type*/ argType,
                                                          cond, ba, falseValue);
          newSel.getDefiningOp()->setAttr("recheck", rewriter.getUnitAttr());
          // Map: old_arg -> new_select
          bvm.map(ba, newSel);
        }
      }
    }

    // --- 5. Perform transformation (based on whether ifOp has results) ---
    if (ifOp.getNumResults() > 0) {
      // --- Case 1: Has results (e.g., Example 1) ---

      // Check else block: must exist, contain only one scf.yield, and yield
      // a constant (Request 4)
      Block *elseBlock = ifOp.elseBlock();
      if (!elseBlock || elseBlock->empty()) {
        return rewriter.notifyMatchFailure(
            linalgOp, "Failed match: if with results has no else block.");
      }
      if (!llvm::hasSingleElement(*elseBlock)) {
        return rewriter.notifyMatchFailure(
            linalgOp, "Failed match: else block has more than one op.");
      }
      auto elseYield = dyn_cast<scf::YieldOp>(elseBlock->getTerminator());
      if (!elseYield || elseYield.getNumOperands() == 0) {
        return rewriter.notifyMatchFailure(
            linalgOp, "Failed match: else block has no valid yield.");
      }

      Value elseValue = elseYield.getOperand(0);
      if (!elseValue.getDefiningOp<arith::ConstantOp>()) {
        return rewriter.notifyMatchFailure(
            linalgOp, "Failed match: else block does not yield a constant.");
        return failure();
      }

      // Get the `then` block's yield value
      auto thenYield = dyn_cast<scf::YieldOp>(thenBlock->getTerminator());
      if (!thenYield || thenYield->getNumOperands() == 0) {
        return rewriter.notifyMatchFailure(
            linalgOp, "Failed match: then block has no valid yield.");
        return failure();
      }
      Value thenValue = thenYield->getOperand(0);

      // Clone the operations of the 'then' block (excluding terminator) before
      // ifOp
      // BVM will automatically replace block-args
      // BVM will also be updated during cloning: old_op_result -> new_op_result
      for (Operation &op : thenBlock->without_terminator()) {
        rewriter.clone(op, bvm);
      }

      // Find the (potentially remapped) 'then' yield value
      Value newThenValue = bvm.lookupOrDefault(thenValue);

      // Create the final arith.select (to replace ifOp)
      Value finalSel = rewriter.create<arith::SelectOp>(
          loc, newThenValue.getType(), cond, newThenValue, elseValue);

      // Replace ifOp results and erase ifOp
      rewriter.replaceOp(ifOp, finalSel);

    } else {
      // --- Case 2: No results (e.g., Example 2) ---

      // Check else block: must be empty (or only yield)
      Block *elseBlock = ifOp.elseBlock();
      if (elseBlock && !elseBlock->empty()) {
        // If the else block is not empty, it *must* have only one scf.yield
        if (!isa<scf::YieldOp>(elseBlock->front()) ||
            !llvm::hasSingleElement(*elseBlock)) {
          return rewriter.notifyMatchFailure(
              linalgOp,
              "Failed match: if with no results has non-empty else block.");
        }
      }

      // Clone the operations of the 'then' block (excluding terminator) before
      // ifOp
      // BVM will automatically replace block-args
      for (Operation &op : thenBlock->without_terminator()) {
        rewriter.clone(op, bvm);
      }

      // Erase the if op
      rewriter.eraseOp(ifOp);
    }

    return success();
  }
};

/**
 * @brief Fixes Linalg select's false value being zero for a store operation.
 *
 * Motivation:
 * The `LinalgIfToSelectPattern` transforms `scf.if` into `arith.select`s.
 * In this conversion, it defaults the 'false' value (when the condition is
false)  * for a select operation to a zero constant (0).  *  * This zero value
is incorrect when the result of this `select` is used as the  * *value* for a
`memref.store`.  *  * The correct 'false' value should be the *original* value
in the memref, as  * a false condition means we should not modify the memory
location (i.e., we  * store the old value).  *  * This pattern aims to correct
this error. It finds the "value select" (the select  * providing the value to
store), finds the corresponding "index select" (providing  * the index), and
replaces the incorrect zero with:  * `memref.load(%memref, %false_index)`  *
(i.e., the original value at the "false" index).  *  * Before Transformation
(IR):  *  * // "index select" (provides index for store)  * %sel_idx =
arith.select %cond, %idx_t, %idx_f {recheck}  * %idx_cast = arith.index_cast
%sel_idx : i32 to index  *  * // "value select" (provides value for store)  *
%cst_0 = arith.constant 0.0 : f32 // <-- Incorrect 0  * %sel_val = arith.select
%cond, %val_t, %cst_0 {recheck}  *  * memref.store %sel_val, %memref[%idx_cast]
 *
 * After Transformation (Corrected IR):
 *
 * %sel_idx = arith.select %cond, %idx_t, %idx_f {recheck}
 * %idx_cast = arith.index_cast %sel_idx : i32 to index
 *
 * // Correction:
 * %idx_f_cast = arith.index_cast %idx_f : i32 to index
 * // Load original value at the "false" index, hoisted outside linalg if
invariant  * %new_val_f = memref.load %memref[%idx_f_cast]  * // Replace 0 with
the original value  * %sel_val = arith.select %cond, %val_t, %new_val_f  *  *
memref.store %sel_val, %memref[%idx_cast]  */
struct FixLinalgSelectZeroForStorePattern
    : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp valueSel,
                                PatternRewriter &rewriter) const final {
    // 1. Must have the "recheck" attribute
    if (!valueSel->hasAttr("recheck"))
      return failure();

    // 2. Check for "skip": if a user is arith.index_cast, this is an index
    // select, not a value select
    if (llvm::any_of(valueSel->getUsers(),

                     [](Operation *u) { return isa<arith::IndexCastOp>(u); })) {
      return rewriter.notifyMatchFailure(valueSel,
                                         "Has index_cast user, skipping.");
    }

    // 3. Find the "memref.store" use
    memref::StoreOp storeOp;
    for (Operation *user : valueSel->getUsers()) {
      if (auto s = dyn_cast<memref::StoreOp>(user)) {
        if (s.getValue() == valueSel.getResult()) {
          storeOp = s;
          break;
        }
      }
    }
    if (!storeOp) {
      return rewriter.notifyMatchFailure(
          storeOp, "Not used as value in a memref.store.");
      return failure();
    }

    // 4. Analyze store indices (assume 1D)
    if (storeOp.getIndices().size() != 1) {
      return rewriter.notifyMatchFailure(storeOp, "Store is not 1D, skipping.");
    }
    Value storeIndex = storeOp.getIndices().front();

    // 5. Find the "index select"
    arith::SelectOp indexSel;
    if (auto castOp = storeIndex.getDefiningOp<arith::IndexCastOp>()) {
      indexSel = castOp.getIn().getDefiningOp<arith::SelectOp>();

    } else {
      indexSel = storeIndex.getDefiningOp<arith::SelectOp>();
    }
    if (!indexSel) {
      return rewriter.notifyMatchFailure(
          storeOp, "Store index does not come from a select or cast(select).");
    }

    // --- 6. Check invariants and set insertion point ---
    Value falseIndex = indexSel.getFalseValue();
    Value memref = storeOp.getMemRef();
    Location loc = valueSel.getLoc();

    auto linalgOp = valueSel->getParentOfType<linalg::GenericOp>();

    // Helper function: Check if a value v is defined *outside* the linalgOp's
    // body
    auto isInvariant = [&](Value v) {
      if (!linalgOp)
        return false; // If not inside linalg, cannot be invariant w.r.t linalg
                      // body
      Operation *defOp = v.getDefiningOp();
      if (!defOp) {
        // Is a BlockArgument
        auto ba = cast<BlockArgument>(v);
        // It's invariant if its owner is the linalg.generic's body
        return ba.getOwner() == linalgOp.getBody();
      }
      // If defined by an op, the op must be outside the linalgOp's region
      return !linalgOp.getRegion().isProperAncestor(defOp->getParentRegion());
    };

    if (linalgOp && isInvariant(memref) && isInvariant(falseIndex)) {
      // **Hoisting Path**: Insertion point is before linalg.generic
      rewriter.setInsertionPoint(linalgOp);

    } else {
      // **Internal Path**: Insertion point is before valueSel
      rewriter.setInsertionPoint(valueSel);
    }

    // 7. Create load index (ensure it is 'index' type)
    Type indexType = rewriter.getIndexType();
    Value loadIndex;
    if (falseIndex.getType() == indexType) {
      loadIndex = falseIndex;

    } else if (isa<IntegerType>(falseIndex.getType())) {
      loadIndex =
          rewriter.create<arith::IndexCastOp>(loc, indexType, falseIndex);

    } else {
      return valueSel.emitError("False index type is not integer or index.");
    }

    // 8. Create memref.load (position determined by setInsertionPoint)
    // %new_val_f = memref.load %memref[%idx_f_cast]
    Value newLoad =
        rewriter.create<memref::LoadOp>(loc, memref, ValueRange{loadIndex});
    rewriter.setInsertionPoint(valueSel);
    // 9. Replace the old select
    // replaceOpWithNewOp automatically inserts the new op at the *original*
    // op's (valueSel) location even if the rewriter's "default" insertion point
    // is elsewhere
    rewriter.replaceOpWithNewOp<arith::SelectOp>(
        valueSel, valueSel.getCondition(), valueSel.getTrueValue(), newLoad);

    return success();
  }
};

struct LinalgIfToSelectPass
    : mlir::dicp::LinalgExt::impl::LinalgIfToSelectBase<LinalgIfToSelectPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();
    UnitAttr unitAttr = UnitAttr::get(context);
    moduleOp.walk([&](linalg::GenericOp linalgOp) {
      // Skip if already tagged
      if (linalgOp->hasAttr("ExtractedLoadOrStore"))
        return;

      // Condition 1: Must be all "parallel" iterators
      if (!llvm::all_of(linalgOp.getIteratorTypesArray(),
                        linalg::isParallelIterator)) {
        return;
      }

      // Condition 2: All indexing maps must be identity
      bool allIdentity = true;
      for (AffineMap map : linalgOp.getIndexingMapsArray()) {
        if (!map.isIdentity()) {
          return;
        }
      }

      // Condition 3 & 4: Must contain scf.if and (load/store) inside the body
      bool hasScfIf = false;
      bool hasLoadStore = false;

      linalgOp.getBody()->walk([&](Operation *innerOp) {
        // Check for scf.if
        if (isa<scf::IfOp>(innerOp)) {
          hasScfIf = true;
        }

        // Check for various load/store operations
        if (isa<tensor::ExtractOp, tensor::InsertOp, memref::LoadOp,
                memref::StoreOp, memref::CopyOp>(innerOp)) {
          hasLoadStore = true;
        }
      });

      // Tag the op if all conditions are met
      if (hasScfIf && hasLoadStore) {
        linalgOp->setAttr("ExtractedLoadOrStore", unitAttr);
      }
    });
    RewritePatternSet patterns(context);
    patterns.add<LinalgIfToSelectPattern>(context);
    patterns.add<FixLinalgSelectZeroForStorePattern>(context);
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
    // Apply Patterns
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }

    {
      RewritePatternSet patterns(context);
      populateLinalgLiftSelectPattern(patterns);
      // Apply Patterns
      if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
        signalPassFailure();
      }

      PassManager pm(&getContext(), moduleOp.getOperationName());
      // Erase dead code and fold constants created during lowering
      pm.addPass(createCSEPass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createSymbolDCEPass());
      if (failed(runPipeline(pm, getOperation()))) {
        signalPassFailure();
      }
    }

  } // namespace
};

} // namespace

void mlir::dicp::LinalgExt::populateLinalgLiftSelectPattern(
    RewritePatternSet &patterns) {
  patterns.add<LiftScalarSelectToTensorPattern>(patterns.getContext());
  patterns.add<LiftYieldSelectOutPattern>(patterns.getContext());
  linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::dicp::LinalgExt::createLinalgIfToSelectPass() {
  return std::make_unique<LinalgIfToSelectPass>();
}