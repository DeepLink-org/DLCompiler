#include "dicp/Dialect/NPU/IR/NPUDialect.h"
#include "dicp/Dialect/NPU/IR/NPUTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <string>

// #define GET_OP_CLASSES
// #include "dicp/Dialect/NPU/IR/NPUOps.h.inc"

// #define GET_OP_CLASSES
// #include "dicp/Dialect/NPU/IR/NPUDialect.cpp.inc"
#define GET_OP_CLASSES
#include "dicp/Dialect/NPU/IR/NPUOps.cpp.inc"

using namespace mlir;
using namespace mlir::dicp;

namespace mlir::dicp::npu {

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
  SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (FunctionType funcType = llvm::dyn_cast<FunctionType>(type)) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                               result.operands))
      return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // If all of the types are the same, print the type directly.
  Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  // Otherwise, print a functional type.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddFOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF16Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult AddFOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void AddFOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

// void CreateTQueueOp::print(mlir::OpAsmPrinter &p) {
//   //  printBinaryOp(p, *this);
//    p.printFunctionalType((*this)->getResultTypes());
// }

// void Copy::build(OpBuilder &b, OperationState &state,
//                 ArrayRef<int64_t> offsets,
//                             ArrayRef<int64_t> sizes,
//                             ArrayRef<int64_t> strides
//                             ) {
//   SmallVector<int64_t> staticStrides, staticOffsets, staticShape;
//   SmallVector<Value> dynamicStrides, dynamicOffsets, dynamicShape;

//   dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
//   dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
//   dispatchIndexOpFoldResults(shape, dynamicShape, staticShape);

//   Type resType;
//   auto basePtr = cast<triton::PointerType>(base.getType());
//   auto elemType = basePtr.getPointeeType();
//   // non-block pointer
//   if (order.empty()) {
//     resType = RankedTensorType::get(sizes, basePtr);
//   }
//   // block pointer
//   else {
//     resType = triton::PointerType::get(RankedTensorType::get(sizes,
//     elemType),
//                                        basePtr.getAddressSpace());
//   }

//   build(b, state, resType, base, sizes, dynamicStrides, dynamicOffsets,
//         dynamicShape, b.getDenseI64ArrayAttr(staticStrides),
//         b.getDenseI64ArrayAttr(staticOffsets),
//         b.getDenseI64ArrayAttr(staticShape), order);
// }
} // namespace mlir::dicp::npu
