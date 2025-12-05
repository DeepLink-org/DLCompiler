#ifndef TRITON_CONVERSION_TRITONARITHTOLINALGNPU_H
#define TRITON_CONVERSION_TRITONARITHTOLINALGNPU_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::dicp::linked {

void populateTritonArithToLinalgNPUConversionPatterns(
    bool pidsToFuncArgs, bool addptrToLinalg, bool assertToCf,
    bool transposeReduceToRank0, RewritePatternSet &patterns);

// NPU Specialization: Dynamically determines if an arith/math operation is
// legal for lowering. An operation is legal if it does NOT require
// conversion/lowering by the current pass.
bool isLegalConstantAndTensorArithmeticOpForNPU(Operation *op);

} // namespace mlir::dicp::linked

#endif // TRITON_CONVERSION_TritonArithToLinalgNPU_TritonArithToLinalgNPU_H
