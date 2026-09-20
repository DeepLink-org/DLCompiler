#ifndef MLIR_DIALECT_MATHEXT_IR_MATHEXT_H_
#define MLIR_DIALECT_MATHEXT_IR_MATHEXT_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

//===----------------------------------------------------------------------===//
// MathExt Dialect
//===----------------------------------------------------------------------===//
#include "mlir-ext/Dialect/MathExt/IR/MathExtDialect.h.inc"

//===----------------------------------------------------------------------===//
// MathExt Dialect Operations
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "mlir-ext/Dialect/MathExt/IR/MathExtOps.h.inc"

#endif // MLIR_DIALECT_MATHEXT_IR_MATHEXT_H_