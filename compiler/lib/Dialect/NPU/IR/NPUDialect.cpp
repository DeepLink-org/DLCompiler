#include "dicp/Dialect/NPU/IR/NPUDialect.h"
#include "dicp/Dialect/NPU/IR/NPUDialect.cpp.inc"
#include "dicp/Dialect/NPU/IR/NPUTypes.h"
#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`

#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::dicp::npu;

// Type parseType(DialectAsmParser &parser) const {
//   StringRef typeKind;
//   if (parser.parseKeyword(&typeKind))
//     return {};
//   auto type = llvm::StringSwitch<Type>(typeKind)
//                   .Case("tqueue", TQueueType::get(getContext()))
//                   .Default(nullptr);
//   if (!type) {
//     parser.emitError(parser.getCurrentLocation())
//         << "unknown NPU type: " << typeKind;
//   }
//   return type;
// }

// void printType(Type type, DialectAsmPrinter &p) const {
//   if (llvm::isa<TQueueType>(type)) {
//     p << "tqueue";
//   } else {
//     assert(false && "unknown tqueue type");
//   }
// }

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void NPUDialect::initialize() {
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "dicp/Dialect/NPU/IR/NPUOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//
