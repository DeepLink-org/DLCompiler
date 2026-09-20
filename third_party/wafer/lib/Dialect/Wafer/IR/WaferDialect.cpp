//===-------------------------- WaferDialect.cpp ---------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "wafer/Dialect/IR/WaferDialect.h"

using namespace mlir;
using namespace mlir::wafer;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void WaferDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "wafer/Dialect/IR/WaferOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "wafer/Dialect/IR/WaferEnums.cpp.inc"
#include "wafer/Dialect/IR/WaferOps.cpp.inc"

#include "wafer/Dialect/IR/WaferDialect.cpp.inc"
