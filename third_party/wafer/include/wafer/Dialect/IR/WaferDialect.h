//===-------------------------- WaferDialect.h -----------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WAFER_IR_DIALECT_H
#define MLIR_DIALECT_WAFER_IR_DIALECT_H

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

//===----------------------------------------------------------------------===//
// Wafer Wafer Operations
//===----------------------------------------------------------------------===//
#include "wafer/Dialect/IR/WaferDialect.h.inc"

// Include the auto-generated header file containing the declarations of the
// TritonStructured operations.
#define GET_OP_CLASSES
#include "wafer/Dialect/IR/WaferEnums.h.inc"
#include "wafer/Dialect/IR/WaferOps.h.inc"

#endif // MLIR_DIALECT_WAFER_IR_DIALECT_H
