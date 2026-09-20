//===----------------------- WaferDialect.td -------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef WAFER_DIALECT
#define WAFER_DIALECT

include "mlir/IR/OpBase.td"

def WaferDialect : Dialect {
  let name = "wafer";

  let cppNamespace = "::mlir::wafer";

  let summary = "The Wafer Wafer IR in MLIR";

  let description = [{
    Wafer Wafer Dialect.

    Dependent Dialects:
      * MK
      * Memref
      * Bufferization
  }];

  let dependentDialects = [
  ];

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  // let hasConstantMaterializer = 1;
  // let useDefaultTypePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

include "wafer/Dialect/IR/WaferTypes.td"

#endif // WAFER_DIALECT
