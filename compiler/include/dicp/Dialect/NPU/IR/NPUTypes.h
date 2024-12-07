#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "dicp/Dialect/NPU/IR/NPUTypes.h.inc"

// namespace mlir::dicp::npu {

// struct TQueueType : public Type::TypeBase<TQueueType, Type, TypeStorage> {
//   using Base::Base;

//   static constexpr StringLiteral name = "npu.tqueue";
// };
// }
