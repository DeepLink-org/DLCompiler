//===------------------------ Barrier.c -----------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Barrier see WaferOps.td for detail.
//
//===----------------------------------------------------------------------===//

#include "wafer.h"

void __Barrier() {
  INTRNISIC_RUN_SWITCH;
  TsmWaitfinish();
}
