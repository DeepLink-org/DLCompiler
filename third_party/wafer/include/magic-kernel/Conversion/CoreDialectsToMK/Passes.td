//===------------------- Passes.td ----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef CORE_DIALECTS_TO_MK_CONVERSION_PASSES
#define CORE_DIALECTS_TO_MK_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def CoreDialectsToMK : Pass<"core-dialects-to-mk", "mlir::ModuleOp"> {
  let summary = "Convert core dialects including Linalg, Memref etc to MK";
  let constructor = "triton::createCoreDialectsToMKPass()";
  let options = [
     Option<"precisionPriority", "precision-priority", "bool", /*default*/"false",
       "Enable precision priority mode, translating integer-related operations to RISC-V instructions (legacy alias for mode 2)">,
     Option<"precisionMode", "precision-mode", "int", /*default*/"0",
       "Integer precision: 0 permits fp32 lowering, 1 preserves i64, 2 preserves i32/i64 and division">
  ];
}

#endif
