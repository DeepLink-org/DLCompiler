//===------------------------ AtomicBarrierIn.c
//----------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::AtomicBarrierIn see WaferOps.td for detail.
//
//===----------------------------------------------------------------------===//

#include "wafer.h"

void __AtomicBarrierIn() {
#ifdef USE_SIM_MODE
#else
  atomic_barrier_in();
  SYNCHRONOUS_INTRINSIC_SWITCH;
#endif
}
