#include "dicp/Conversion/LinalgToNPU/LinalgToNPU.h"
#include "dicp/Dialect/NPU/IR/NPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>
#include <type_traits>

#define DEBUG_TYPE "linalg-to-npu"
#include "dicp/Conversion/LinalgToNPU/ConversionPatterns.hpp"

using namespace mlir;
using namespace dicp;

#define GEN_PASS_CLASSES
#include "dicp/Conversion/LinalgToNPU/Passes.h.inc"

void npu::populateLinalgToNPUConversionPatterns(RewritePatternSet &patterns) {
  // patterns.add<AddFConverter>(patterns.getContext());
  // patterns.add<CopyConverter>(patterns.getContext());
  // patterns.add<SubViewConverter>(patterns.getContext());
  // patterns.add<LinalgGenericConverter>(patterns.getContext());
  patterns.add<ConvertLinalgGenericToArith>(patterns.getContext());
  // patterns.add<ConvertLinalgGenericToBroadcast>(patterns.getContext());
}
