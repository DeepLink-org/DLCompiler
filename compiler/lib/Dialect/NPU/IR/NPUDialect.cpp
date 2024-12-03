#include "compiler/include/Dialect/NPU/IR/NPUDialect.h"

using namespace mlir;
using namespace mlir::npu;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void NPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "compiler/include/Dialect/NPU/IR/NPUOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "compiler/include/Dialect/NPU/IR/NPUDialect.cpp.inc"
#include "compiler/include/Dialect/NPU/IR/NPUOps.cpp.inc"
