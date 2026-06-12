

#ifndef TRITON_ADAPTER_TRITONTOLINALG_HOISTBROADCAST_H
#define TRITON_ADAPTER_TRITONTOLINALG_HOISTBROADCAST_H

#include "dicp/TritonToLinalg/BlockPtrAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-to-linalg"

namespace HoistBroadcast {
using namespace mlir;
using namespace triton;

class BroadcastConverter : public OpConversionPattern<triton::BroadcastOp> {
public:
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class BroadcastHoister {
public:
  BroadcastHoister(triton::BroadcastOp op);
  LogicalResult parse(Value operand, const Location &loc,
                      ConversionPatternRewriter &rewriter);
  LogicalResult parseAddptr(triton::AddPtrOp op, const Location &loc,
                            ConversionPatternRewriter &rewriter);
  LogicalResult parseBroadcast(triton::BroadcastOp op, const Location &loc,
                               ConversionPatternRewriter &rewriter);
  LogicalResult parseSplat(triton::SplatOp op, const Location &loc,
                           ConversionPatternRewriter &rewriter);
  LogicalResult findSrc(Value operand);
  LogicalResult replaceBroadcastOp(triton::BroadcastOp op,
                                   ConversionPatternRewriter &rewriter);
  bool canBroadcast();

private:
  Value source;
  triton::BroadcastOp opToHoist;
  SmallVector<int64_t> tensorSizes;
  llvm::SmallDenseMap<Value, Value> broadcastMap;
};
} // namespace HoistBroadcast

#endif // TRITON_ADAPTER_TRITONTOLINALG_HOISTBROADCAST_H
