#include "dicp/Conversion/TritonToLinalgNPU/TritonArithToLinalg/NPUSpecific.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>
#include <type_traits>

#define DEBUG_TYPE "triton-arith-to-linalg-npu"
#include "dicp/Conversion/TritonToLinalgNPU/TritonArithToLinalg/ConversionPatterns.hpp"

using namespace mlir;
using namespace triton;
using namespace mlir::dicp::linked;

void mlir::dicp::linked::populateTritonArithToLinalgNPUConversionPatterns(
    bool pidsToFuncArgs, bool addptrToLinalg, bool assertToCf,
    bool transposeReduceToRank0, RewritePatternSet &patterns) {

  if (pidsToFuncArgs) {
    patterns.add<GetProgramIDConverter, GetNumProgramsConverter>(
        patterns.getContext());
  }
  if (addptrToLinalg) {
    patterns.add<AddPtrConverter>(patterns.getContext());
  }
  if (assertToCf) {
    patterns.add<AssertConverter>(patterns.getContext());
  }
  patterns.add<BroadcastNPUConverter>(patterns.getContext());
  patterns.add<TransposeConverter>(patterns.getContext());
  patterns.add<MakeRangeConverter>(patterns.getContext());
  patterns.add<ExpandDimsConverter>(patterns.getContext());
  patterns.add<BitcastNPUConverter>(patterns.getContext());
  patterns.add<CallConverter>(patterns.getContext());
  patterns.add<MulHiUIOpConverter>(patterns.getContext());
  patterns.add<PreciseSqrtConverter>(patterns.getContext());
  patterns.add<PreciseDivConverter>(patterns.getContext());
  patterns.add<CatConverter>(patterns.getContext());
  patterns.add<SplitConverter>(patterns.getContext());
  patterns.add<JoinConverter>(patterns.getContext());
  patterns.add<FpToFpConverter>(patterns.getContext());
  patterns.add<ClampConverter>(patterns.getContext());
  patterns.add<MatmulNPUConverter>(patterns.getContext());
  patterns.add<SplatConverter>(patterns.getContext());
  patterns.add<UnsplatConverter>(patterns.getContext());
  patterns.add<DenseConstantConverter>(patterns.getContext());
  patterns.add<ReshapeConverter>(patterns.getContext());

  populateExternElementwiseOpToMLIROps(patterns);

  // Reduce converters
  // Triton's reduce op is idential to linalg.reduce op, so we can clone
  // `tt.reduce` body to `linalg.reduce`. Unfortunately, we still need to
  // perform pattern matching to know what reduce ops we are dealing with
  // so that we know how to initialize the initial reduce values correctly.
  //
  // We can do this in a generic way without pattern matching by always using
  // the first elements along the reduction axis and perform the reduction on
  // the remaining elements. However, this results in creatings sub-tensors that
  // aren't always multiple of 2s, which are sub-optimal for certain hardwares.
  patterns.add<ArgMinConverter>(patterns.getContext());
  patterns.add<ArgMaxConverter>(patterns.getContext());
  patterns.add<ReduceNPUConverter>(patterns.getContext(),
                                   transposeReduceToRank0);

  // linalg::populateElementwiseToLinalgConversionPatterns(patterns);
}

bool mlir::dicp::linked::isLegalConstantAndTensorArithmeticOpForNPU(
    Operation *op) {
  // Check for arith::ConstantOp
  if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
    // 1. Scalar constants are always legal (handled elsewhere).
    if (!isa<RankedTensorType>(constOp.getResult().getType())) {
      return true;
    }
    // 2. RankedTensor constant check:
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
      // Dense splat constants of float/integer type are ILLEGAL (must be
      // lowered, e.g., to linalg.fill).
      if (denseAttr.isSplat() &&
          isa<FloatType, IntegerType>(denseAttr.getElementType())) {
        return false; // **ILLEGAL**: needs lowering (e.g., to linalg.fill).
      }
    }
    // All other RankedTensor constants (non-splat, non-float/int elements) are
    // legal.
    return true;
  }
  // 3. All non-constant arith/math operations are currently considered legal
  //    (i.e., they should not be lowered by the constant-related pass).
  return true;
}