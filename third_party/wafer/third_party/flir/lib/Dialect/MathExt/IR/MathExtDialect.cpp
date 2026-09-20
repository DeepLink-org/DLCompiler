#include "mlir-ext/Dialect/MathExt/IR/MathExt.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::mathext;

#include "mlir-ext/Dialect/MathExt/IR/MathExtDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// MathExt dialect.
//===----------------------------------------------------------------------===//

void MathExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-ext/Dialect/MathExt/IR/MathExtOps.cpp.inc"
      >();
}
