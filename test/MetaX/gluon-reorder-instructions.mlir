// RUN: triton-opt --split-input-file \
// RUN:   --tritonmetaxgpu-gluon-reorder-instructions \
// RUN:   --tritonmetaxgpu-gluon-reorder-instructions %s | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // A same-buffer write in the region must keep the load outside.
  // CHECK-LABEL: tt.func public @aliasing_store_blocks_sink
  tt.func public @aliasing_store_blocks_sink(
      %buffer: !ttg.memdesc<16xf32, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32, #blocked>
    // CHECK: %[[LOAD:.*]] = ttg.local_load %{{.*}}
    // CHECK-NEXT: scf.for
    %loaded = ttg.local_load %buffer : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #blocked>
    scf.for %i = %c0 to %c1 step %c1 {
      ttg.local_store %zero, %buffer : tensor<16xf32, #blocked> -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
      %sum = arith.addf %loaded, %zero : tensor<16xf32, #blocked>
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // A read-only RegionBranch has no conflicting dependence.
  // CHECK-LABEL: tt.func public @read_only_loop_sinks_load
  tt.func public @read_only_loop_sinks_load(
      %buffer: !ttg.memdesc<16xf32, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32, #blocked>
    %loaded = ttg.local_load %buffer : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #blocked>
    // CHECK: scf.for
    // CHECK-NEXT: %[[LOAD:.*]] = ttg.local_load %{{.*}}
    scf.for %i = %c0 to %c1 step %c1 {
      %sum = arith.addf %loaded, %zero : tensor<16xf32, #blocked>
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // Distinct local_alloc roots refine a generic MayAlias result.
  // CHECK-LABEL: tt.func public @distinct_allocations_allow_sink
  tt.func public @distinct_allocations_allow_sink() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32, #blocked>
    // CHECK: %[[READ_BUFFER:.*]] = ttg.local_alloc
    // CHECK-NEXT: %[[WRITE_BUFFER:.*]] = ttg.local_alloc
    %read_buffer = ttg.local_alloc : () -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    %write_buffer = ttg.local_alloc : () -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    %loaded = ttg.local_load %read_buffer : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #blocked>
    // CHECK: scf.for
    // CHECK: ttg.local_store %{{.*}}, %[[WRITE_BUFFER]]
    // CHECK: %[[LOAD:.*]] = ttg.local_load %[[READ_BUFFER]]
    scf.for %i = %c0 to %c1 step %c1 {
      ttg.local_store %zero, %write_buffer : tensor<16xf32, #blocked> -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
      %sum = arith.addf %loaded, %zero : tensor<16xf32, #blocked>
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // Synchronization operations are scheduling boundaries even without a write.
  // CHECK-LABEL: tt.func public @barrier_blocks_sink
  tt.func public @barrier_blocks_sink(
      %buffer: !ttg.memdesc<16xf32, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32, #blocked>
    // CHECK: %[[LOAD:.*]] = ttg.local_load %{{.*}}
    // CHECK-NEXT: scf.for
    %loaded = ttg.local_load %buffer : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #blocked>
    scf.for %i = %c0 to %c1 step %c1 {
      ttg.barrier_shared
      %sum = arith.addf %loaded, %zero : tensor<16xf32, #blocked>
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // Distinct function arguments are not distinct allocation proofs.
  // CHECK-LABEL: tt.func public @distinct_arguments_remain_may_alias
  tt.func public @distinct_arguments_remain_may_alias(
      %read_buffer: !ttg.memdesc<16xf32, #shared, #smem, mutable>,
      %write_buffer: !ttg.memdesc<16xf32, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32, #blocked>
    // CHECK: %[[LOAD:.*]] = ttg.local_load %{{.*}}
    // CHECK-NEXT: scf.for
    %loaded = ttg.local_load %read_buffer : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #blocked>
    scf.for %i = %c0 to %c1 step %c1 {
      ttg.local_store %zero, %write_buffer : tensor<16xf32, #blocked> -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
      %sum = arith.addf %loaded, %zero : tensor<16xf32, #blocked>
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // Multiple uses prevent cloning or speculative duplication.
  // CHECK-LABEL: tt.func public @multiple_uses_stay_outside
  tt.func public @multiple_uses_stay_outside(
      %buffer: !ttg.memdesc<16xf32, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32, #blocked>
    // CHECK: %[[LOAD:.*]] = ttg.local_load %{{.*}}
    // CHECK-NEXT: scf.for
    %loaded = ttg.local_load %buffer : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #blocked>
    scf.for %i = %c0 to %c1 step %c1 {
      %sum0 = arith.addf %loaded, %zero : tensor<16xf32, #blocked>
      %sum1 = arith.mulf %loaded, %zero : tensor<16xf32, #blocked>
    }
    tt.return
  }
}
