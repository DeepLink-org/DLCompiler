
#ifndef TRITON_TO_STRUCTURED_MASKANALYSIS_H
#define TRITON_TO_STRUCTURED_MASKANALYSIS_H

#include <cstddef>
#include <set>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace TritonToStructured {
using namespace mlir;
using namespace triton;

struct dimInfo {
  OpFoldResult offset;
  OpFoldResult shape;
  OpFoldResult rhs;
  size_t dimIndex;
  bool hasBroadCast = false;

  enum class CompareType { slt, sge, ult, uge, deafaultType };

  CompareType currentType = CompareType::deafaultType;

  dimInfo(size_t dimIndex = 0, bool hasBroadCast = false)
      : dimIndex(dimIndex), hasBroadCast(hasBroadCast) {}

  dimInfo(OpFoldResult offset, OpFoldResult shape, size_t dimIndex = 0,
          bool hasBroadCast = false,
          CompareType Type = CompareType::deafaultType,
          OpFoldResult rhs = nullptr)
      : offset(offset), shape(shape), dimIndex(dimIndex),
        hasBroadCast(hasBroadCast), currentType(Type), rhs(rhs) {}

  bool setType(arith::CmpIPredicate Type);
  bool compareTypeIsLess() const;
  void dump() const;
};

struct MaskState {
  SmallVector<dimInfo> stateInfo;
  OpFoldResult scalar;
  Value newMask;

  // Recursively parse a Value; call the corresponding function based on the
  // defining operation and Value type
  LogicalResult parse(Value operand, const Location loc, OpBuilder &builder);

  bool isEmpty() const { return stateInfo.empty() && !scalar; }
  void dump() const;

  // Operand is the result of a constant
  // Get the value of the constant and assign it to scalar.
  LogicalResult parseConstant(arith::ConstantOp constOp, const Location loc,
                              OpBuilder &builder);

  LogicalResult parseIntScalar(Value scalar, const Location loc,
                               OpBuilder &builder);

  LogicalResult parseMakeRange(triton::MakeRangeOp rangeOp, const Location loc,
                               OpBuilder &builder);

  LogicalResult parseExtSI(arith::ExtSIOp op, const Location loc,
                           OpBuilder &builder);

  LogicalResult parseSplat(triton::SplatOp splatOp, const Location loc,
                           OpBuilder &builder);

  LogicalResult parseExpandDims(triton::ExpandDimsOp expandDimsOp,
                                const Location loc, OpBuilder &builder);

  LogicalResult parseAdd(arith::AddIOp addOp, const Location loc,
                         OpBuilder &builder);

  LogicalResult parseBroadcast(triton::BroadcastOp broadcastOp,
                               const Location loc, OpBuilder &builder);

  LogicalResult addStates(const MaskState &lhsState, const MaskState &rhsState,
                          Location loc, OpBuilder &builder);

  LogicalResult addStateScalar(const MaskState &state,
                               const OpFoldResult scalar, Location loc,
                               OpBuilder &builder);

  LogicalResult parseCmp(arith::CmpIOp cmpOp, const Location loc,
                         OpBuilder &builder);

  LogicalResult parseRem(arith::RemSIOp remOp, const Location loc,
                         OpBuilder &builder);

  LogicalResult parseDiv(arith::DivSIOp divOp, const Location loc,
                         OpBuilder &builder);

  LogicalResult parseAnd(arith::AndIOp andOp, const Location loc,
                         OpBuilder &builder);

  LogicalResult analysisMask(Value operand);

  Value createNewMask(const Location loc, OpBuilder &builder);
};

} // namespace TritonToStructured

#endif