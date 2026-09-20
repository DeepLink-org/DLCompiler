#ifndef MLIR_DIALECT_TRITON_STRUCTURED_INCUBATED_IR_TRITON_STRUCTURED_DIALECT_INCUBATED_H_
#define MLIR_DIALECT_TRITON_STRUCTURED_INCUBATED_IR_TRITON_STRUCTURED_DIALECT_INCUBATED_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
//===----------------------------------------------------------------------===//
// TritonStructured Operations
//===----------------------------------------------------------------------===//
#include "incubated/Dialect/TritonStructuredIncubated/IR/TritonStructuredDialectIncubated.h.inc"

// Include the auto-generated header file containing the declarations of the
// TritonStructured operations.
#define GET_OP_CLASSES

#include "incubated/Dialect/TritonStructuredIncubated/IR/TritonStructuredOpsIncubated.h.inc"

#endif
