#include "incubated/Dialect/TritonStructuredIncubated/IR/TritonStructuredDialectIncubated.h"

using namespace mlir;
using namespace mlir::tts::Incubated;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void TritonStructuredDialectIncubated::initialize() {
  addOperations<
#define GET_OP_LIST
#include "incubated/Dialect/TritonStructuredIncubated/IR/TritonStructuredOpsIncubated.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "incubated/Dialect/TritonStructuredIncubated/IR/TritonStructuredOpsIncubated.cpp.inc"

#include "incubated/Dialect/TritonStructuredIncubated/IR/TritonStructuredDialectIncubated.cpp.inc"
