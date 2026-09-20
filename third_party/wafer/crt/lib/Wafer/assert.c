// ===------------------------ assert.c
// ------------------------------------===//

// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.

// ===---------------------------------------------------------------------===//

// Enable wafer kernel assert support

#include "wafer.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

void __Assert(const char *message, ...) {
  INTRNISIC_RUN_SWITCH;
  va_list args;
  va_start(args, message);

  char *file = va_arg(args, char *);
  int line = va_arg(args, int);
  int col = va_arg(args, int);
  int pidX = va_arg(args, int);
  int pidY = va_arg(args, int);
  int pidZ = va_arg(args, int);
  va_end(args);

#ifdef USE_SIM_MODE
  printf("%s(line %d, col %d)::tile (%d, %d, %d): %s\n", file, line, col, pidX,
         pidY, pidZ, message);
  abort();
#else
  tsm_ep_log(__FILE__, __func__, __LINE__, KCORE_LOG_ERROR,
             "%s(line %d, col %d)::tile (%d, %d, %d): %s\n", file, line, col,
             pidX, pidY, pidZ, message);
  // RT_ASSERT is an RT-Thread macro, not an exported firmware function.
  // Call the SDK's non-returning newlib assertion entry directly: assert(0)
  // would disappear from Release builds when NDEBUG is defined.
  __assert_func(file, line, __func__, message);
#endif
}
