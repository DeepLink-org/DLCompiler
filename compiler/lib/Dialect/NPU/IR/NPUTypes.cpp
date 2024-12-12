#include "dicp/Dialect/NPU/IR/NPUTypes.h"
#include "dicp/Dialect/NPU/IR/NPUDialect.h"
#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`

using namespace mlir;
using namespace mlir::dicp::npu;

#define GET_TYPEDEF_CLASSES
#include "dicp/Dialect/NPU/IR/NPUTypes.cpp.inc"

Type TQueueType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  int position = 1;
  if (parser.parseInteger(position)) {
    return Type();
  }
  int bufferNumber = 1;
  if (parser.parseInteger(bufferNumber)) {
    return Type();
  }

  if (parser.parseGreater()) {
    return Type();
  }

  return TQueueType::get(parser.getContext(), position, bufferNumber);
}

void TQueueType::print(AsmPrinter &printer) const {
  printer << "npu.tqueue<" << getPosition() << ", " << getBufferNumber() << ">";
}

Type TPipType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  int type = 1;
  if (parser.parseInteger(type)) {
    return Type();
  }

  if (parser.parseGreater()) {
    return Type();
  }

  return TPipType::get(parser.getContext(), type);
}

void TPipType::print(AsmPrinter &printer) const {
  printer << "npu.tpip<" << getType() << ">";
}

Type GlobalTensorType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  int type = 1;
  if (parser.parseInteger(type)) {
    return Type();
  }

  if (parser.parseGreater()) {
    return Type();
  }

  return GlobalTensorType::get(parser.getContext(), type);
}

void GlobalTensorType::print(AsmPrinter &printer) const {
  printer << "npu.global_tensor<" << getType() << ">";
}

//===----------------------------------------------------------------------===//
void NPUDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "dicp/Dialect/NPU/IR/NPUTypes.cpp.inc"
      >();
}
