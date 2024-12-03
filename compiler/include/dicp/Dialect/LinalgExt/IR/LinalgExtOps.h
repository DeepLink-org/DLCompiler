#ifndef DIALECT_LINALGEXT_IR_LINALGEXT_OPS_H_
#define DIALECT_LINALGEXT_IR_LINALGEXT_OPS_H_

#include "dicp/Dialect/LinalgExt/IR/Traits.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LLVM.h"

#include "dicp/Dialect/LinalgExt/IR/LinalgExtDialect.h.inc"

#define GET_OP_CLASSES
#include "dicp/Dialect/LinalgExt/IR/LinalgExtOps.h.inc"
#define GET_TYPEDEF_CLASSES
#include "dicp/Dialect/LinalgExt/IR/LinalgExtTypes.h.inc"

#endif // DIALECT_LINALGEXT_IR_LINALGEXT_OPS_H_