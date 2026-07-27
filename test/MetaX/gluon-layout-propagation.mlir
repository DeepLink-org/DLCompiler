// RUN: split-file %s %t
// RUN: triton-opt --split-input-file \
// RUN:   --tritonmetaxgpu-gluon-insert-require-layout='compute-capability=80' \
// RUN:   --tritonmetaxgpu-gluon-propagate-layout \
// RUN:   --tritonmetaxgpu-gluon-resolve-placeholder-layouts \
// RUN:   --tritonmetaxgpu-gluon-legalize-c500-async-copy-layout \
// RUN:   --tritonmetaxgpu-gluon-legalize-register-slices \
// RUN:   --tritonmetaxgpu-gluon-verify-layout-contracts \
// RUN:   --tritonmetaxgpu-insert-gvm-arrive-barrier-shared \
// RUN:   --tritonmetaxgpu-gluon-verify-synchronization \
// RUN:   %t/local-store.mlir | FileCheck %t/local-store.mlir
// RUN: triton-opt \
// RUN:   --tritonmetaxgpu-gluon-insert-require-layout='compute-capability=80' \
// RUN:   %t/memdesc-region.mlir | \
// RUN:   FileCheck %t/memdesc-region.mlir --check-prefix=FIXED-MEMDESC \
// RUN:   --implicit-check-not='gluon.require_layout {{.*}} : !ttg.memdesc'
// RUN: triton-opt \
// RUN:   --tritonmetaxgpu-gluon-propagate-layout \
// RUN:   --tritonmetaxgpu-gluon-resolve-placeholder-layouts \
// RUN:   %t/async-source.mlir | FileCheck %t/async-source.mlir \
// RUN:   --implicit-check-not='#gluon.auto_encoding'
// RUN: triton-opt \
// RUN:   --tritonmetaxgpu-gluon-insert-require-layout='compute-capability=80' \
// RUN:   %t/global-preservation.mlir | FileCheck %t/global-preservation.mlir \
// RUN:   --implicit-check-not=gluon.set_auto_layout \
// RUN:   --implicit-check-not=gluon.require_layout
// RUN: triton-opt \
// RUN:   --tritonmetaxgpu-gluon-insert-require-layout='compute-capability=80' \
// RUN:   --tritonmetaxgpu-gluon-propagate-layout \
// RUN:   --tritonmetaxgpu-gluon-resolve-placeholder-layouts \
// RUN:   --tritonmetaxgpu-gluon-verify-layout-contracts \
// RUN:   %t/global-load-auto.mlir | FileCheck %t/global-load-auto.mlir \
// RUN:   --implicit-check-not='#gluon.auto_encoding' \
// RUN:   --implicit-check-not=gluon.set_auto_layout \
// RUN:   --implicit-check-not=ttg.convert_layout
// RUN: triton-opt \
// RUN:   --tritonmetaxgpu-gluon-insert-require-layout='compute-capability=80' \
// RUN:   --tritonmetaxgpu-gluon-propagate-layout \
// RUN:   --tritonmetaxgpu-gluon-resolve-placeholder-layouts \
// RUN:   --tritonmetaxgpu-gluon-verify-layout-contracts \
// RUN:   %t/global-rank1-producer.mlir | \
// RUN:   FileCheck %t/global-rank1-producer.mlir \
// RUN:   --implicit-check-not='#gluon.auto_encoding' \
// RUN:   --implicit-check-not=gluon.set_auto_layout \
// RUN:   --implicit-check-not=gluon.require_layout \
// RUN:   --implicit-check-not=ttg.convert_layout
// RUN: triton-opt \
// RUN:   --tritonmetaxgpu-gluon-propagate-layout \
// RUN:   --tritonmetaxgpu-gluon-resolve-placeholder-layouts \
// RUN:   --tritonmetaxgpu-gluon-verify-layout-contracts \
// RUN:   %t/loop-dot-explicit-seed.mlir | \
// RUN:   FileCheck %t/loop-dot-explicit-seed.mlir \
// RUN:   --implicit-check-not='#gluon.auto_encoding' \
// RUN:   --implicit-check-not=gluon.set_auto_layout \
// RUN:   --implicit-check-not=gluon.require_layout
// RUN: triton-opt \
// RUN:   --tritonmetaxgpu-gluon-insert-require-layout='compute-capability=80' \
// RUN:   --tritonmetaxgpu-gluon-propagate-layout \
// RUN:   --tritonmetaxgpu-gluon-resolve-placeholder-layouts \
// RUN:   --tritonmetaxgpu-gluon-verify-layout-contracts \
// RUN:   %t/expand-multi-parent-conflict.mlir | \
// RUN:   FileCheck %t/expand-multi-parent-conflict.mlir \
// RUN:   --implicit-check-not='#gluon.auto_encoding' \
// RUN:   --implicit-check-not=gluon.deferred_tensor_layout_boundary
// RUN: triton-opt \
// RUN:   --tritonmetaxgpu-gluon-insert-require-layout='compute-capability=80' \
// RUN:   --tritonmetaxgpu-gluon-propagate-layout \
// RUN:   --tritonmetaxgpu-gluon-resolve-placeholder-layouts \
// RUN:   --tritongpu-optimize-dot-operands='hoist-layout-conversion=true' \
// RUN:   --tritongpu-remove-layout-conversions \
// RUN:   --cse --canonicalize \
// RUN:   --tritonmetaxgpu-gluon-verify-layout-contracts \
// RUN:   %t/expand-multi-parent-conflict.mlir -o /dev/null
// RUN: not triton-opt \
// RUN:   --tritonmetaxgpu-gluon-propagate-layout \
// RUN:   --tritonmetaxgpu-gluon-resolve-placeholder-layouts \
// RUN:   --tritonmetaxgpu-gluon-legalize-c500-async-copy-layout \
// RUN:   --tritonmetaxgpu-gluon-legalize-register-slices \
// RUN:   --tritonmetaxgpu-gluon-verify-layout-contracts \
// RUN:   --tritonmetaxgpu-insert-gvm-arrive-barrier-shared \
// RUN:   --tritonmetaxgpu-gluon-verify-synchronization \
// RUN:   %t/register-to-shared-invalid.mlir 2>&1 | \
// RUN:   FileCheck %t/register-to-shared-invalid.mlir
// RUN: triton-opt \
// RUN:   --tritonmetaxgpu-gluon-propagate-layout \
// RUN:   --tritonmetaxgpu-gluon-resolve-placeholder-layouts \
// RUN:   %t/hard-fixed-conflict.mlir | \
// RUN:   FileCheck %t/hard-fixed-conflict.mlir \
// RUN:   --check-prefix=SET-AUTO-CONFLICT \
// RUN:   --implicit-check-not='#gluon.auto_encoding' \
// RUN:   --implicit-check-not='gluon.set_auto_layout'

//--- local-store.mlir

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @local_store_source_is_layout_sink
  tt.func public @local_store_source_is_layout_sink(%arg0: !ttg.memdesc<16xf32, #shared, #smem, mutable>) -> tensor<16xf32, #blocked> {
    // CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32, #blocked>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: ttg.local_store %[[CST]], %arg0 : tensor<16xf32, #blocked> -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    // CHECK: tt.return %[[CST]]
    %0 = arith.constant dense<0.000000e+00> : tensor<16xf32, #gluon.auto_encoding>
    %1 = gluon.set_auto_layout %0 : tensor<16xf32, #gluon.auto_encoding> -> tensor<16xf32, #blocked>
    ttg.local_store %0, %arg0 : tensor<16xf32, #gluon.auto_encoding> -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    tt.return %1 : tensor<16xf32, #blocked>
  }

  // CHECK-LABEL: @local_store_only_local_load_gets_default_register_layout
  tt.func public @local_store_only_local_load_gets_default_register_layout(
      %arg0: !ttg.memdesc<16xf32, #shared, #smem, mutable>,
      %arg1: !ttg.memdesc<16xf32, #shared, #smem, mutable>) {
    // CHECK: %[[LOAD:.*]] = ttg.local_load %arg0 : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #[[DEFAULT:blocked[0-9]*]]>
    // CHECK: ttg.local_store %[[LOAD]], %arg1 : tensor<16xf32, #[[DEFAULT]]> -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    %0 = ttg.local_load %arg0 : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #gluon.auto_encoding>
    ttg.local_store %0, %arg1 : tensor<16xf32, #gluon.auto_encoding> -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @local_load_store_load_roundtrip_then_global_store
  tt.func public @local_load_store_load_roundtrip_then_global_store(
      %arg0: !ttg.memdesc<16xf32, #shared, #smem, mutable>,
      %arg1: !ttg.memdesc<16xf32, #shared, #smem, mutable>,
      %arg2: !tt.ptr<f32>) {
    // CHECK: %[[MASK:.*]] = arith.constant dense<true> : tensor<16xi1, #blocked>
    // CHECK: %[[PTR:.*]] = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    // The first load is governed by its immediate shared-to-register producer
    // boundary; the later global store independently constrains the second.
    // CHECK: %[[LOAD0:.*]] = ttg.local_load %arg0 : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #[[SOURCE:blocked[0-9]*]]>
    // CHECK: ttg.local_store %[[LOAD0]], %arg1 : tensor<16xf32, #[[SOURCE]]> -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    // CHECK: %[[LOAD1:.*]] = ttg.local_load %arg1 : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #blocked>
    // CHECK: tt.store %[[PTR]], %[[LOAD1]], %[[MASK]] : tensor<16x!tt.ptr<f32>, #blocked>
    %ptr = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #gluon.auto_encoding>
    %ptr_seed = gluon.set_auto_layout %ptr : tensor<16x!tt.ptr<f32>, #gluon.auto_encoding> -> tensor<16x!tt.ptr<f32>, #blocked>
    %mask = arith.constant dense<true> : tensor<16xi1, #gluon.auto_encoding>
    %0 = ttg.local_load %arg0 : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #gluon.auto_encoding>
    ttg.local_store %0, %arg1 : tensor<16xf32, #gluon.auto_encoding> -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    %1 = ttg.local_load %arg1 : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #gluon.auto_encoding>
    tt.store %ptr, %1, %mask : tensor<16x!tt.ptr<f32>, #gluon.auto_encoding>
    tt.return
  }

  // A manual-layout pipeline skips requirement insertion, but the first
  // concrete-layout legalizer still removes frontend-only provenance.
  // CHECK-LABEL: @manual_default_shared_marker_is_consumed
  // CHECK: ttg.local_alloc
  // CHECK-NOT: ttg.gluon.default-shared-layout
  // CHECK: ttg.local_store
  // CHECK: tt.return
  tt.func public @manual_default_shared_marker_is_consumed() {
    %value = arith.constant dense<0.000000e+00> : tensor<16xf32, #blocked>
    %buffer = ttg.local_alloc {"ttg.gluon.default-shared-layout"} : () -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    ttg.local_store %value, %buffer : tensor<16xf32, #blocked> -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // Conflicting requirements use the target's deterministic blocked fallback;
  // each non-matching consumer retains an explicit conversion boundary.
  // CHECK: #blocked2 = #ttg.blocked<{{.*}}sizePerThread = [1, 1]{{.*}}threadsPerWarp = [2, 32]
  // CHECK-LABEL: @conflicting_requirements_use_target_default
  tt.func public @conflicting_requirements_use_target_default() -> (tensor<32x32xf32, #blocked0>, tensor<32x32xf32, #blocked1>) {
    // CHECK: %[[VALUE:.*]] = arith.constant {{.*}} : tensor<32x32xf32, #blocked2>
    // CHECK: %[[FIRST:.*]] = ttg.convert_layout %[[VALUE]] : tensor<32x32xf32, #blocked2> -> tensor<32x32xf32, #blocked>
    // CHECK: %[[SECOND:.*]] = ttg.convert_layout %[[VALUE]] : tensor<32x32xf32, #blocked2> -> tensor<32x32xf32, #blocked1>
    // CHECK: tt.return %[[FIRST]], %[[SECOND]]
    %value = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #gluon.auto_encoding>
    %first = gluon.require_layout %value : tensor<32x32xf32, #gluon.auto_encoding> -> tensor<32x32xf32, #blocked0>
    %second = gluon.require_layout %value : tensor<32x32xf32, #gluon.auto_encoding> -> tensor<32x32xf32, #blocked1>
    tt.return %first, %second : tensor<32x32xf32, #blocked0>, tensor<32x32xf32, #blocked1>
  }
}

// -----

#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], elementsMNK = [1, 4, 8], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // A pure local_alloc(init)->local_load staging round trip is a layout conversion.
  // CHECK-LABEL: @computed_mma_alloc_init_to_dot_shared_folds
  tt.func public @computed_mma_alloc_init_to_dot_shared_folds(
      %arg0: tensor<32x32xf32, #mma>) -> tensor<32x32xf32, #dot_a> {
    // CHECK-NOT: ttg.local_alloc
    // CHECK: %[[CONVERT:.*]] = ttg.convert_layout %arg0
    // CHECK: tt.return %[[CONVERT]]
    %0 = ttg.local_alloc %arg0 : (tensor<32x32xf32, #mma>) -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #dot_a>
    tt.return %1 : tensor<32x32xf32, #dot_a>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#seeded = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // A sink fallback must not override a concrete seed across a view.
  // CHECK-LABEL: @local_store_extract_uses_parent_seed
  tt.func public @local_store_extract_uses_parent_seed(
      %dst: !ttg.memdesc<32x128xf16, #shared, #smem, mutable>) -> tensor<128x128xf32, #blocked> {
    // CHECK: %[[PARENT:.*]] = arith.constant {{.*}} : tensor<128x128xf32, #blocked>
    // CHECK: %[[SLICE:.*]] = "ttg.extract_tensor"(%[[PARENT]])
    // CHECK: ttg.local_store %{{.*}}, %{{.*}}
    // CHECK-NOT: #gluon.auto_encoding
    %parent = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #gluon.auto_encoding>
    %seed = gluon.set_auto_layout %parent : tensor<128x128xf32, #gluon.auto_encoding> -> tensor<128x128xf32, #blocked>
    %slice = "gluon.extract_slice"(%parent) <{offsets = array<i64: 0, 0>}> : (tensor<128x128xf32, #gluon.auto_encoding>) -> tensor<32x128xf32, #gluon.auto_encoding>
    %stage = arith.truncf %slice : tensor<32x128xf32, #gluon.auto_encoding> to tensor<32x128xf16, #gluon.auto_encoding>
    ttg.local_store %stage, %dst : tensor<32x128xf16, #gluon.auto_encoding> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    tt.return %seed : tensor<128x128xf32, #blocked>
  }

  // The update is the only concrete predecessor seed for the logical insert.
  // CHECK-LABEL: tt.func public @local_store_insert_uses_update_seed
  // CHECK-SAME: -> tensor<16x128xf16, #[[$INSERT_SEED:[A-Za-z0-9_]+]]>
  tt.func public @local_store_insert_uses_update_seed(
      %dst: !ttg.memdesc<32x128xf16, #shared, #smem, mutable>)
      -> tensor<16x128xf16, #seeded> {
    // CHECK: %[[BASE:.*]] = arith.constant {{.*}} : tensor<32x128xf16, #[[$INSERT_SEED]]>
    // CHECK: %[[UPDATE:.*]] = arith.constant {{.*}} : tensor<16x128xf16, #[[$INSERT_SEED]]>
    // CHECK-NOT: #gluon.auto_encoding
    %base = arith.constant dense<0.000000e+00>
        : tensor<32x128xf16, #gluon.auto_encoding>
    %update = arith.constant dense<1.000000e+00>
        : tensor<16x128xf16, #gluon.auto_encoding>
    %merged = "gluon.insert_slice"(%base, %update)
        <{offsets = array<i64: 0, 0>}>
        : (tensor<32x128xf16, #gluon.auto_encoding>,
           tensor<16x128xf16, #gluon.auto_encoding>)
          -> tensor<32x128xf16, #gluon.auto_encoding>
    %seed = gluon.set_auto_layout %update
        : tensor<16x128xf16, #gluon.auto_encoding>
          -> tensor<16x128xf16, #seeded>
    ttg.local_store %merged, %dst
        : tensor<32x128xf16, #gluon.auto_encoding>
          -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    tt.return %seed : tensor<16x128xf16, #seeded>
  }
}

// -----

#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], elementsMNK = [1, 4, 8], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // RegionBranch traversal must preserve the init seed and terminate on backedges.
  // CHECK-LABEL: @local_store_region_carrier_uses_init_seed
  tt.func public @local_store_region_carrier_uses_init_seed(
      %dst: !ttg.memdesc<32x32xf16, #shared, #smem, mutable>) -> tensor<32x32xf32, #mma> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: %[[ZERO:.*]] = arith.constant {{.*}} : tensor<32x32xf32, #mma>
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #gluon.auto_encoding>
    %seed = gluon.set_auto_layout %zero : tensor<32x32xf32, #gluon.auto_encoding> -> tensor<32x32xf32, #mma>
    // CHECK: %[[LOOP:.*]] = scf.for {{.*}} -> (tensor<32x32xf32, #mma>)
    %loop = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %zero) -> (tensor<32x32xf32, #gluon.auto_encoding>) {
      %next = arith.addf %acc, %zero : tensor<32x32xf32, #gluon.auto_encoding>
      scf.yield %next : tensor<32x32xf32, #gluon.auto_encoding>
    }
    %stage = arith.truncf %loop : tensor<32x32xf32, #gluon.auto_encoding> to tensor<32x32xf16, #gluon.auto_encoding>
    // CHECK: ttg.local_store %{{.*}}, %{{.*}} : tensor<32x32xf16, #mma>
    // CHECK-NOT: #gluon.auto_encoding
    ttg.local_store %stage, %dst : tensor<32x32xf16, #gluon.auto_encoding> -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    tt.return %seed : tensor<32x32xf32, #mma>
  }
}

// -----

#terminal = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @terminal_store_local_load_keeps_terminal_layout
  tt.func public @terminal_store_local_load_keeps_terminal_layout(
      %arg0: !ttg.memdesc<32x32xf16, #shared, #smem, mutable>,
      %arg1: !tt.ptr<f16>) {
    // CHECK: %[[MASK:.*]] = arith.constant dense<true> : tensor<32x32xi1, #blocked>
    // CHECK: %[[PTR:.*]] = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    // CHECK: %[[LOAD:.*]] = ttg.local_load %arg0 : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #blocked>
    // CHECK: tt.store %[[PTR]], %[[LOAD]], %[[MASK]] : tensor<32x32x!tt.ptr<f16>, #blocked>
    %ptr = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #gluon.auto_encoding>
    %ptr_seed = gluon.set_auto_layout %ptr : tensor<32x32x!tt.ptr<f16>, #gluon.auto_encoding> -> tensor<32x32x!tt.ptr<f16>, #terminal>
    %mask = arith.constant dense<true> : tensor<32x32xi1, #gluon.auto_encoding>
    %0 = ttg.local_load %arg0 : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #gluon.auto_encoding>
    tt.store %ptr, %0, %mask : tensor<32x32x!tt.ptr<f16>, #gluon.auto_encoding>
    tt.return
  }

  // Shared memory is a physical layout boundary: the producer-side load/store
  // pair may use a target-default register layout while the downstream load
  // independently honors its terminal consumer. Each local_store operand must
  // still exactly match its defining local_load, without an implicit relayout.
  // CHECK-LABEL: @local_store_roundtrip_keeps_boundary_layouts_independent
  tt.func public @local_store_roundtrip_keeps_boundary_layouts_independent(
      %arg0: !ttg.memdesc<32x32xf16, #shared, #smem, mutable>,
      %arg1: !ttg.memdesc<32x32xf16, #shared, #smem, mutable>,
      %arg2: !tt.ptr<f16>) {
    // CHECK: %[[MASK:.*]] = arith.constant dense<true> : tensor<32x32xi1, #blocked>
    // CHECK: %[[PTR:.*]] = tt.splat %arg2 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    // CHECK: %[[LOAD0:.*]] = ttg.local_load %arg0 : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #[[$STAGE_LAYOUT:[A-Za-z0-9_]+]]>
    // CHECK: ttg.local_store %[[LOAD0]], %arg1 : tensor<32x32xf16, #[[$STAGE_LAYOUT]]> -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    // CHECK: %[[LOAD1:.*]] = ttg.local_load %arg1 : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #blocked>
    // CHECK: tt.store %[[PTR]], %[[LOAD1]], %[[MASK]] : tensor<32x32x!tt.ptr<f16>, #blocked>
    %ptr = tt.splat %arg2 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #gluon.auto_encoding>
    %ptr_seed = gluon.set_auto_layout %ptr : tensor<32x32x!tt.ptr<f16>, #gluon.auto_encoding> -> tensor<32x32x!tt.ptr<f16>, #terminal>
    %mask = arith.constant dense<true> : tensor<32x32xi1, #gluon.auto_encoding>
    %0 = ttg.local_load %arg0 : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #gluon.auto_encoding>
    ttg.local_store %0, %arg1 : tensor<32x32xf16, #gluon.auto_encoding> -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %1 = ttg.local_load %arg1 : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #gluon.auto_encoding>
    tt.store %ptr, %1, %mask : tensor<32x32x!tt.ptr<f16>, #gluon.auto_encoding>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @no_dot_async_copy_infers_wait_and_barrier
  tt.func public @no_dot_async_copy_infers_wait_and_barrier(
      %arg0: tensor<32x32x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>},
      %arg1: tensor<32x32xi1, #blocked>) {
    // CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #blocked>
    // CHECK: %[[SMEM:.*]] = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    // A layout-only conversion must preserve the tt.contiguity proof attached
    // to %arg0; the post-rewrite physical verifier consumes this exact value.
    // CHECK: %[[COPY_PTR:.*]] = ttg.convert_layout %arg0 : tensor<32x32x!tt.ptr<f16>, #blocked> -> tensor<32x32x!tt.ptr<f16>, #{{[A-Za-z0-9_]+}}>
    // CHECK: ttg.async_copy_global_to_local %[[COPY_PTR]], %[[SMEM]] mask %{{.*}} other %{{.*}}
    // CHECK: ttg.gvm_arrive {{.*}}num = 0 : i32
    // CHECK: ttg.barrier_shared
    // CHECK: %[[OUT:.*]] = ttg.local_load %[[SMEM]] {{.*}} : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #blocked>
    // CHECK: tt.store %arg0, %[[OUT]], %arg1 : tensor<32x32x!tt.ptr<f16>, #blocked>
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %0 = ttg.async_copy_global_to_local %arg0, %smem mask %arg1 other %zero {intrinsic = true} : tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #smem, mutable>
    %1 = ttg.local_load %smem {intrinsic = true, isConstantOffs = true} : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #gluon.auto_encoding>
    %seed = gluon.set_auto_layout %1 : tensor<32x32xf16, #gluon.auto_encoding> -> tensor<32x32xf16, #blocked>
    tt.store %arg0, %seed, %arg1 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }

}

// -----

// Layout costs are independent of MLIR's intrusive SSA use-list order. These
// paired functions reverse only boundary-user discovery order; each pair must
// still select the same concrete producer layout.

#blocked_m = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_n = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared_m = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared_n = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func public @register_to_shared_m_then_n
  // CHECK: %[[VALUE_MN:.*]] = arith.constant {{.*}} : tensor<32x32xf32, #[[$SELECTED:[A-Za-z0-9_]+]]>
  tt.func public @register_to_shared_m_then_n(
      %dst_m: !ttg.memdesc<32x32xf32, #shared_m, #smem, mutable>,
      %dst_n: !ttg.memdesc<32x32xf32, #shared_n, #smem, mutable>)
      -> (tensor<32x32xf32, #blocked_m>, tensor<32x32xf32, #blocked_n>) {
    %value = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #gluon.auto_encoding>
    ttg.local_store %value, %dst_m : tensor<32x32xf32, #gluon.auto_encoding> -> !ttg.memdesc<32x32xf32, #shared_m, #smem, mutable>
    ttg.local_store %value, %dst_n : tensor<32x32xf32, #gluon.auto_encoding> -> !ttg.memdesc<32x32xf32, #shared_n, #smem, mutable>
    %read_m = ttg.local_load %dst_m : !ttg.memdesc<32x32xf32, #shared_m, #smem, mutable> -> tensor<32x32xf32, #blocked_m>
    %read_n = ttg.local_load %dst_n : !ttg.memdesc<32x32xf32, #shared_n, #smem, mutable> -> tensor<32x32xf32, #blocked_n>
    tt.return %read_m, %read_n : tensor<32x32xf32, #blocked_m>, tensor<32x32xf32, #blocked_n>
  }

  // CHECK-LABEL: tt.func public @register_to_shared_n_then_m
  // CHECK: %[[VALUE_NM:.*]] = arith.constant {{.*}} : tensor<32x32xf32, #[[$SELECTED]]>
  tt.func public @register_to_shared_n_then_m(
      %dst_m: !ttg.memdesc<32x32xf32, #shared_m, #smem, mutable>,
      %dst_n: !ttg.memdesc<32x32xf32, #shared_n, #smem, mutable>)
      -> (tensor<32x32xf32, #blocked_m>, tensor<32x32xf32, #blocked_n>) {
    %value = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #gluon.auto_encoding>
    ttg.local_store %value, %dst_n : tensor<32x32xf32, #gluon.auto_encoding> -> !ttg.memdesc<32x32xf32, #shared_n, #smem, mutable>
    ttg.local_store %value, %dst_m : tensor<32x32xf32, #gluon.auto_encoding> -> !ttg.memdesc<32x32xf32, #shared_m, #smem, mutable>
    %read_m = ttg.local_load %dst_m : !ttg.memdesc<32x32xf32, #shared_m, #smem, mutable> -> tensor<32x32xf32, #blocked_m>
    %read_n = ttg.local_load %dst_n : !ttg.memdesc<32x32xf32, #shared_n, #smem, mutable> -> tensor<32x32xf32, #blocked_n>
    tt.return %read_m, %read_n : tensor<32x32xf32, #blocked_m>, tensor<32x32xf32, #blocked_n>
  }

}

//--- expand-multi-parent-conflict.mlir

#left = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#right = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // One logical row may feed two independent full-rank ownership domains.
  // The shared source stays generic and each conflicting projection gets a
  // local conversion before expand_dims.
  // CHECK-DAG: #[[$LEFT:[A-Za-z0-9_]+]] = #ttg.blocked<{{.*}}sizePerThread = [1, 4]{{.*}}order = [1, 0]
  // CHECK-DAG: #[[$RIGHT:[A-Za-z0-9_]+]] = #ttg.blocked<{{.*}}sizePerThread = [4, 1]{{.*}}order = [0, 1]
  // CHECK-LABEL: @expand_multi_parent_conflict_is_local
  // CHECK: %[[ROW:.*]] = arith.constant {{.*}} : tensor<32xf32, #[[$GENERIC:[A-Za-z0-9_]+]]>
  // CHECK: %[[LEFT_ROW:.*]] = ttg.convert_layout %[[ROW]] : tensor<32xf32, #[[$GENERIC]]> -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #[[$LEFT]]}>>
  // CHECK: %[[LEFT_COLUMN:.*]] = tt.expand_dims %[[LEFT_ROW]] {{.*}} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #[[$LEFT]]}>> -> tensor<32x1xf32, #[[$LEFT]]>
  // CHECK: %[[RIGHT_ROW:.*]] = ttg.convert_layout %[[ROW]] : tensor<32xf32, #[[$GENERIC]]> -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #[[$RIGHT]]}>>
  // CHECK: %[[RIGHT_COLUMN:.*]] = tt.expand_dims %[[RIGHT_ROW]] {{.*}} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #[[$RIGHT]]}>> -> tensor<32x1xf32, #[[$RIGHT]]>
  tt.func public @expand_multi_parent_conflict_is_local()
      -> (tensor<32x32xf32, #left>, tensor<32x32xf32, #right>) {
    %row = arith.constant dense<0.000000e+00>
        : tensor<32xf32, #gluon.auto_encoding>
    %left_column = tt.expand_dims %row {axis = 1 : i32}
        : tensor<32xf32, #gluon.auto_encoding>
          -> tensor<32x1xf32, #gluon.auto_encoding>
    %left_matrix = tt.broadcast %left_column
        : tensor<32x1xf32, #gluon.auto_encoding>
          -> tensor<32x32xf32, #gluon.auto_encoding>
    %left_seed = gluon.set_auto_layout %left_matrix
        : tensor<32x32xf32, #gluon.auto_encoding>
          -> tensor<32x32xf32, #left>
    %right_column = tt.expand_dims %row {axis = 1 : i32}
        : tensor<32xf32, #gluon.auto_encoding>
          -> tensor<32x1xf32, #gluon.auto_encoding>
    %right_matrix = tt.broadcast %right_column
        : tensor<32x1xf32, #gluon.auto_encoding>
          -> tensor<32x32xf32, #gluon.auto_encoding>
    %right_seed = gluon.set_auto_layout %right_matrix
        : tensor<32x32xf32, #gluon.auto_encoding>
          -> tensor<32x32xf32, #right>
    tt.return %left_seed, %right_seed
        : tensor<32x32xf32, #left>, tensor<32x32xf32, #right>
  }
}

//--- memdesc-region.mlir

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // Both local loads alias the same fixed shared family through a region
  // carrier. The memdesc encoding remains source-authored; each load may use a
  // different lowerable register view selected for its dot operand.
  // FIXED-MEMDESC-LABEL: tt.func public @same_family_through_if
  // FIXED-MEMDESC: scf.if
  // FIXED-MEMDESC: ttg.local_load
  // FIXED-MEMDESC: ttg.local_load
  // FIXED-MEMDESC: gluon.set_auto_layout
  // FIXED-MEMDESC: gluon.require_layout
  // FIXED-MEMDESC: gluon.require_layout
  // FIXED-MEMDESC: tt.dot
  tt.func public @same_family_through_if(
      %condition: i1,
      %storage: !ttg.memdesc<32x32xf16, #shared, #smem, mutable>) {
    %carried = scf.if %condition -> (!ttg.memdesc<32x32xf16, #shared, #smem, mutable>) {
      scf.yield %storage : !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    } else {
      scf.yield %storage : !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    }
    %a = ttg.local_load %storage : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #gluon.auto_encoding>
    %b = ttg.local_load %carried : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #gluon.auto_encoding>
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #gluon.auto_encoding>
    %result = tt.dot %a, %b, %zero, inputPrecision = tf32 : tensor<32x32xf16, #gluon.auto_encoding> * tensor<32x32xf16, #gluon.auto_encoding> -> tensor<32x32xf32, #gluon.auto_encoding>
    tt.return
  }

}

//--- async-source.mlir

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // The async source operands are one exact distributed component. Resolve
  // chooses one generic blocked root and closes ptr/mask/other together.
  // CHECK-LABEL: tt.func public @async_source_resolves_mask_and_other
  // CHECK: %[[ASYNC_PTR:.*]] = tt.splat %{{.*}} {{.*}} : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #[[$ASYNC_DIST:[A-Za-z0-9_]+]]>
  // CHECK: %[[ASYNC_MASK:.*]] = tt.splat %{{.*}} : i1 -> tensor<32x32xi1, #[[$ASYNC_DIST]]>
  // CHECK: %[[ASYNC_OTHER:.*]] = tt.splat %{{.*}} : f16 -> tensor<32x32xf16, #[[$ASYNC_DIST]]>
  // CHECK: ttg.async_copy_global_to_local %[[ASYNC_PTR]], %{{.*}} mask %[[ASYNC_MASK]] other %[[ASYNC_OTHER]]
  tt.func public @async_source_resolves_mask_and_other(
      %base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %predicate: i1,
      %storage: !ttg.memdesc<32x32xf16, #shared, #smem, mutable>) {
    %ptr = tt.splat %base
        {tt.contiguity = dense<[1, 16]> : tensor<2xi32>}
        : !tt.ptr<f16>
          -> tensor<32x32x!tt.ptr<f16>, #gluon.auto_encoding>
    %mask = tt.splat %predicate
        : i1 -> tensor<32x32xi1, #gluon.auto_encoding>
    %zero = arith.constant 0.000000e+00 : f16
    %other = tt.splat %zero
        : f16 -> tensor<32x32xf16, #gluon.auto_encoding>
    %token = ttg.async_copy_global_to_local
        %ptr, %storage mask %mask other %other {intrinsic = true}
        : tensor<32x32x!tt.ptr<f16>, #gluon.auto_encoding>
          -> <32x32xf16, #shared, #smem, mutable>
    tt.return
  }
}

//--- global-preservation.mlir
// A concrete rank-2 global-memory layout is an input contract, not a request
// for target re-inference.  In particular, a valid register subview used by a
// normal global load is not subject to the stricter contiguous-segment rule of
// C500 async-copy issue planning.

#parent = #ttg.blocked<{sizePerThread = [8, 4], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#fragment = #ttg.blocked<{sizePerThread = [4, 2], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func public @preserve_concrete_global_subview
  // CHECK: %[[SUB:.*]] = "ttg.extract_tensor"(%{{.*}}) <{ctaIdx = array<i64: 0, 1>, elemIdx = array<i64: 0, 2, 4, 6, 8, 10, 12, 14>}> : (tensor<128x128x!tt.ptr<f16>, #[[$PARENT:[A-Za-z0-9_]+]]>) -> tensor<64x64x!tt.ptr<f16>, #[[$FRAGMENT:[A-Za-z0-9_]+]]>
  // CHECK: tt.load %[[SUB]] : tensor<64x64x!tt.ptr<f16>, #[[$FRAGMENT]]>
  tt.func public @preserve_concrete_global_subview(
      %source: tensor<128x128x!tt.ptr<f16>, #parent>) {
    %sub = "ttg.extract_tensor"(%source)
        <{ctaIdx = array<i64: 0, 1>, elemIdx = array<i64: 0, 2, 4, 6, 8, 10, 12, 14>}>
        : (tensor<128x128x!tt.ptr<f16>, #parent>)
          -> tensor<64x64x!tt.ptr<f16>, #fragment>
    %value = tt.load %sub : tensor<64x64x!tt.ptr<f16>, #fragment>
    tt.return
  }
}

//--- global-load-auto.mlir

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // The normal global-load bridge is target-fixed, not an autotune axis.
  // Pointer contiguity selects an eight-element vector along dimension 1,
  // and the load's exact-layout trait closes ptr/mask/other/result together.
  // CHECK: #[[$GLOBAL_LOAD:blocked[0-9]*]] = #ttg.blocked<{{.*}}sizePerThread = [1, 8]{{.*}}threadsPerWarp = [8, 8]{{.*}}warpsPerCTA = [4, 1]{{.*}}order = [1, 0]
  // CHECK-LABEL: tt.func public @auto_global_load_uses_coalesced_component
  // CHECK: %[[PTR:.*]] = tt.splat %{{.*}} {{.*}} : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #[[$GLOBAL_LOAD]]>
  // CHECK: %[[MASK:.*]] = tt.splat %{{.*}} : i1 -> tensor<32x128xi1, #[[$GLOBAL_LOAD]]>
  // CHECK: %[[OTHER:.*]] = tt.splat %{{.*}} : f16 -> tensor<32x128xf16, #[[$GLOBAL_LOAD]]>
  // CHECK: %[[VALUE:.*]] = tt.load %[[PTR]], %[[MASK]], %[[OTHER]] : tensor<32x128x!tt.ptr<f16>, #[[$GLOBAL_LOAD]]>
  tt.func public @auto_global_load_uses_coalesced_component(
      %base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %predicate: i1) {
    %ptr = tt.splat %base
        {tt.contiguity = dense<[1, 16]> : tensor<2xi32>}
        : !tt.ptr<f16>
          -> tensor<32x128x!tt.ptr<f16>, #gluon.auto_encoding>
    %mask = tt.splat %predicate
        : i1 -> tensor<32x128xi1, #gluon.auto_encoding>
    %zero = arith.constant 0.000000e+00 : f16
    %other = tt.splat %zero
        : f16 -> tensor<32x128xf16, #gluon.auto_encoding>
    %value = tt.load %ptr, %mask, %other
        : tensor<32x128x!tt.ptr<f16>, #gluon.auto_encoding>
    tt.return
  }
}

//--- global-rank1-producer.mlir

#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], elementsMNK = [1, 2, 8], colMajor = 1, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>
#row = #ttg.slice<{dim = 1, parent = #mma}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // Rank-1 has no separately proven coalesced bridge. Preserve the concrete
  // producer ownership across the exact store component instead of forcing a
  // generic blocked conversion and shared exchange.
  // CHECK-LABEL: tt.func public @rank1_store_preserves_producer_layout
  // CHECK: %[[PTR:.*]] = tt.addptr {{.*}} : tensor<128x!tt.ptr<f32>, #ttg.slice<{{.*}}>>, tensor<128xi32, #ttg.slice<{{.*}}>>
  // CHECK: %[[MASK:.*]] = tt.splat %{{.*}} : i1 -> tensor<128xi1, #ttg.slice<{{.*}}>>
  // CHECK: %[[VALUE:.*]] = arith.constant {{.*}} : tensor<128xf32, #ttg.slice<{{.*}}>>
  // CHECK: tt.store %[[PTR]], %[[VALUE]], %[[MASK]] : tensor<128x!tt.ptr<f32>, #ttg.slice<{{.*}}>>
  tt.func public @rank1_store_preserves_producer_layout(
      %base: !tt.ptr<f32>, %predicate: i1) {
    %offset = tt.make_range {start = 0 : i32, end = 128 : i32}
        : tensor<128xi32, #gluon.auto_encoding>
    %base_tensor = tt.splat %base
        : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #gluon.auto_encoding>
    %ptr = tt.addptr %base_tensor, %offset
        : tensor<128x!tt.ptr<f32>, #gluon.auto_encoding>,
          tensor<128xi32, #gluon.auto_encoding>
    %mask = tt.splat %predicate
        : i1 -> tensor<128xi1, #gluon.auto_encoding>
    %value = arith.constant dense<0.000000e+00>
        : tensor<128xf32, #gluon.auto_encoding>
    %seed = gluon.set_auto_layout %value
        : tensor<128xf32, #gluon.auto_encoding> -> tensor<128xf32, #row>
    tt.store %ptr, %value, %mask
        : tensor<128x!tt.ptr<f32>, #gluon.auto_encoding>
    tt.return
  }
}

//--- loop-dot-explicit-seed.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], elementsMNK = [1, 4, 8], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma}>
#a_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#b_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // The loop carrier and its local_load initializer are one exact-layout
  // component. The explicit blocked seed owns that component; the dot
  // requirement remains a local conversion instead of retagging the carrier.
  // CHECK-LABEL: tt.func public @loop_carried_dot_uses_component_seed
  // CHECK: %[[A_INIT:.*]] = ttg.local_load {{.*}} -> tensor<32x32xf16, #blocked>
  // CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[A:.*]] = %[[A_INIT]]) -> (tensor<32x32xf16, #blocked>) {
  // CHECK: %[[DOT_A:.*]] = ttg.convert_layout %[[A]] : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{{.*}}opIdx = 0{{.*}}>>
  // CHECK: tt.dot %[[DOT_A]], {{.*}} : tensor<32x32xf16, #ttg.dot_op<{{.*}}opIdx = 0{{.*}}>> * tensor<32x32xf16, #ttg.dot_op<{{.*}}opIdx = 1{{.*}}>> -> tensor<32x32xf32, #mma>
  // CHECK: scf.yield %[[A]] : tensor<32x32xf16, #blocked>
  tt.func public @loop_carried_dot_uses_component_seed(
      %a_smem: !ttg.memdesc<32x32xf16, #a_shared, #smem, mutable>,
      %b_smem: !ttg.memdesc<32x32xf16, #b_shared, #smem, mutable>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a_init = ttg.local_load %a_smem
        : !ttg.memdesc<32x32xf16, #a_shared, #smem, mutable>
          -> tensor<32x32xf16, #gluon.auto_encoding>
    %a_seed = gluon.set_auto_layout %a_init
        : tensor<32x32xf16, #gluon.auto_encoding>
          -> tensor<32x32xf16, #blocked>
    %b = ttg.local_load %b_smem
        : !ttg.memdesc<32x32xf16, #b_shared, #smem, mutable>
          -> tensor<32x32xf16, #dot_b>
    %zero = arith.constant dense<0.000000e+00>
        : tensor<32x32xf32, #mma>
    %loop = scf.for %i = %c0 to %c1 step %c1
        iter_args(%a = %a_init)
        -> (tensor<32x32xf16, #gluon.auto_encoding>) {
      %dot_a = gluon.require_layout %a
          : tensor<32x32xf16, #gluon.auto_encoding>
            -> tensor<32x32xf16, #dot_a>
      %result = tt.dot %dot_a, %b, %zero, inputPrecision = tf32
          : tensor<32x32xf16, #dot_a> * tensor<32x32xf16, #dot_b>
            -> tensor<32x32xf32, #mma>
      scf.yield %a : tensor<32x32xf16, #gluon.auto_encoding>
    }
    tt.return
  }
}

//--- register-to-shared-invalid.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0]]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[0, 1]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @register_to_shared_rejects_cross_cta_mapping(
      %src: tensor<32x32xf32, #blocked>,
      %dst: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // CHECK: register-to-shared sink register layout is not linearly lowerable to shared layout
    ttg.local_store %src, %dst : tensor<32x32xf32, #blocked> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

//--- hard-fixed-conflict.mlir

#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // Conflicting producer anchors widen the source to Unknown. Each residual
  // anchor becomes a conversion from Resolve's generic blocked source.
  // SET-AUTO-CONFLICT: #blocked = #ttg.blocked<{{.*}}sizePerThread = [1, 1]{{.*}}threadsPerWarp = [2, 32]
  // SET-AUTO-CONFLICT-LABEL: tt.func public @set_auto_conflict_is_local
  // SET-AUTO-CONFLICT: %[[VALUE:.*]] = arith.constant {{.*}} : tensor<32x32xf32, #blocked>
  // SET-AUTO-CONFLICT: ttg.convert_layout %[[VALUE]] : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
  // SET-AUTO-CONFLICT: ttg.convert_layout %[[VALUE]] : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked2>
  tt.func public @set_auto_conflict_is_local() {
    %value = arith.constant dense<0.000000e+00>
        : tensor<32x32xf32, #gluon.auto_encoding>
    %first = gluon.set_auto_layout %value
        : tensor<32x32xf32, #gluon.auto_encoding>
          -> tensor<32x32xf32, #blocked0>
    %second = gluon.set_auto_layout %value
        : tensor<32x32xf32, #gluon.auto_encoding>
          -> tensor<32x32xf32, #blocked1>
    tt.return
  }
}
