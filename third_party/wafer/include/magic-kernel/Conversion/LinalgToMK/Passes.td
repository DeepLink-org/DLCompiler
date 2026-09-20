//===------------------- Passes.td ----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_TO_MK_CONVERSION_PASSES
#define LINALG_TO_MK_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def LinalgToMK : Pass<"linalg-to-mk", "mlir::ModuleOp"> {
  let summary = "Convert linalg operations into magic kernel operations";

  let options = [
     Option<"precisionPriority", "precision-priority", "bool", /*default*/"false",
       "Enable precision priority mode, translating integer-related operations to RISC-V instructions (legacy alias for mode 2)">,
     Option<"precisionMode", "precision-mode", "int", /*default*/"0",
       "Integer precision: 0 permits fp32 lowering, 1 preserves i64, 2 preserves i32/i64 and division">
  ];
}

#endif
