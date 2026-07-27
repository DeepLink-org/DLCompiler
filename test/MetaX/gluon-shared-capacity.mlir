// RUN: not triton-opt \
// RUN:   --tritonmetaxgpu-insert-gvm-arrive-barrier-shared \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // One allocation already exceeds the complete 64-KiB C500 capacity. The
  // final allocation verifier reports bytes, not an inferred occupancy value.
  // CHECK: error: final compiler-managed shared-memory allocation exceeds the C500 hard capacity: required 131072 bytes, capacity 65536 bytes
  // CHECK-SAME: reduce live shared buffers or select a lower-scratch layout transfer
  tt.func public @over_capacity() {
    %buffer = ttg.local_alloc : () -> !ttg.memdesc<256x256xf16, #shared, #smem, mutable>
    tt.return
  }
}
