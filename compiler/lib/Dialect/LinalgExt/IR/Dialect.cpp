#include "Dialect/LinalgExt/IR/LinalgExtOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/SmallVector.h"

#include "Dialect/LinalgExt/IR/LinalgExtDialect.cpp.inc"

using namespace mlir;
namespace mlir::dicp::LinalgExt {

void LinalgExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/LinalgExt/IR/LinalgExtOps.cpp.inc"
      >();
}
} // namespace mlir::dicp::LinalgExt
