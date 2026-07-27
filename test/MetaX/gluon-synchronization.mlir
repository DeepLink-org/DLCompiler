// RUN: triton-opt --split-input-file \
// RUN:   --tritonmetaxgpu-gluon-propagate-layout \
// RUN:   --tritonmetaxgpu-gluon-resolve-placeholder-layouts \
// RUN:   --tritonmetaxgpu-gluon-legalize-c500-async-copy-layout \
// RUN:   --tritonmetaxgpu-gluon-legalize-register-slices \
// RUN:   --tritonmetaxgpu-gluon-verify-layout-contracts \
// RUN:   --tritonmetaxgpu-insert-gvm-arrive-barrier-shared \
// RUN:   --tritonmetaxgpu-gluon-verify-synchronization \
// RUN:   %s | FileCheck %s


#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @async_copy_then_load
  tt.func public @async_copy_then_load(
      %ptr: tensor<32x32x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>},
      %mask: tensor<32x32xi1, #blocked>) -> tensor<32x32xf16, #blocked> {
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    // CHECK: ttg.async_copy_global_to_local
    %token = ttg.async_copy_global_to_local %ptr, %smem mask %mask other %zero {intrinsic = true} : tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #smem, mutable>
    // CHECK-NEXT: ttg.gvm_arrive {{.*}}num = 0 : i32
    // CHECK-NEXT: ttg.barrier_shared
    // CHECK-NEXT: %[[ASYNC_LOAD:.*]] = ttg.local_load %{{.*}}
    %load = ttg.local_load %smem {intrinsic = true, isConstantOffs = true} : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #blocked>
    tt.return %load : tensor<32x32xf16, #blocked>
  }

  // CHECK-LABEL: @synchronous_write_then_load
  tt.func public @synchronous_write_then_load(
      %value: tensor<32x32xf16, #blocked>,
      %smem: !ttg.memdesc<32x32xf16, #shared, #smem, mutable>) -> tensor<32x32xf16, #blocked> {
    // CHECK: ttg.local_store %{{.*}}, %{{.*}}
    ttg.local_store %value, %smem : tensor<32x32xf16, #blocked> -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    // CHECK-NEXT: ttg.barrier_shared
    // CHECK-NEXT: %[[WRITE_LOAD:.*]] = ttg.local_load %{{.*}}
    %load = ttg.local_load %smem : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #blocked>
    tt.return %load : tensor<32x32xf16, #blocked>
  }

  // CHECK-LABEL: @load_then_synchronous_overwrite
  tt.func public @load_then_synchronous_overwrite(
      %value: tensor<32x32xf16, #blocked>,
      %smem: !ttg.memdesc<32x32xf16, #shared, #smem, mutable>) -> tensor<32x32xf16, #blocked> {
    // CHECK: %[[READ_LOAD:.*]] = ttg.local_load %{{.*}}
    %load = ttg.local_load %smem : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #blocked>
    // CHECK-NEXT: ttg.barrier_shared
    // CHECK-NEXT: ttg.local_store %{{.*}}, %{{.*}}
    ttg.local_store %value, %smem : tensor<32x32xf16, #blocked> -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    tt.return %load : tensor<32x32xf16, #blocked>
  }

  // User synchronization remains concrete input. Re-planning may preserve it,
  // but must neither delete it nor claim it with compiler ownership markers.
  // CHECK-LABEL: @user_authored_sync_survives_replan
  // CHECK: ttg.gvm_arrive {num = 0 : i32, test.user_owned_sync}
  // CHECK-NEXT: ttg.barrier_shared {test.user_owned_sync}
  tt.func public @user_authored_sync_survives_replan(
      %ptr: tensor<32x32x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>},
      %mask: tensor<32x32xi1, #blocked>) -> tensor<32x32xf16, #blocked> {
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %token = ttg.async_copy_global_to_local %ptr, %smem mask %mask other %zero {intrinsic = true} : tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #smem, mutable>
    ttg.gvm_arrive {num = 0 : i32, test.user_owned_sync}
    ttg.barrier_shared {test.user_owned_sync}
    %load = ttg.local_load %smem {intrinsic = true} : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #blocked>
    tt.return %load : tensor<32x32xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // Backedge reuse of one physical shared allocation requires synchronization.
  // CHECK-LABEL: @cross_iteration_write_reuse_needs_barrier
  tt.func public @cross_iteration_write_reuse_needs_barrier(
      %trip_count: i32,
      %value: tensor<32x32xf16, #blocked>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %smem = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    // CHECK: scf.for
    // CHECK: ttg.barrier_shared {{.*}}metax.gluon.inferred_barrier_shared
    // CHECK-NEXT: ttg.local_store
    scf.for %i = %c0 to %trip_count step %c1 : i32 {
      ttg.local_store %value, %smem : tensor<32x32xf16, #blocked> -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    }
    tt.return
  }
}

// -----

// Address-contiguity proofs must survive register-slice legalization.  The
// full offset tensor and the replacement tile carry the same affine range
// coefficients, so ttg.insert_tensor preserves the unit-stride column axis.
// Register ownership alone is not used as address evidence.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#row = #ttg.slice<{dim = 1, parent = #blocked}>
#col = #ttg.slice<{dim = 0, parent = #blocked}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func public @inserted_offset_preserves_address_contiguity
  // CHECK: %[[INSERTED:.*]] = "ttg.insert_tensor"
  // CHECK: %[[PTRS:.*]] = tt.addptr {{.*}}, %[[INSERTED]]
  // CHECK: %[[SLICE:.*]] = "ttg.extract_tensor"(%[[PTRS]])
  // CHECK: ttg.async_copy_global_to_local %[[SLICE]]
  tt.func public @inserted_offset_preserves_address_contiguity(
      %base: !tt.ptr<f16>) {
    %c128 = arith.constant 128 : i32
    %rows = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #row>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<128xi32, #row> -> tensor<128x1xi32, #blocked>
    %row_scale = tt.splat %c128 : i32 -> tensor<128x1xi32, #blocked>
    %scaled_rows = arith.muli %rows_2d, %row_scale : tensor<128x1xi32, #blocked>
    %scaled_rows_full = tt.broadcast %scaled_rows : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>

    %cols = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #col>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<128xi32, #col> -> tensor<1x128xi32, #blocked>
    %cols_full = tt.broadcast %cols_2d : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %offsets = arith.addi %scaled_rows_full, %cols_full : tensor<128x128xi32, #blocked>

    %tile = "gluon.extract_slice"(%offsets) <{offsets = array<i64: 0, 0>}> : (tensor<128x128xi32, #blocked>) -> tensor<32x128xi32, #blocked>
    %updated = "gluon.insert_slice"(%offsets, %tile) <{offsets = array<i64: 0, 0>}> : (tensor<128x128xi32, #blocked>, tensor<32x128xi32, #blocked>) -> tensor<128x128xi32, #blocked>

    %base_tensor = tt.splat %base : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    %ptrs = tt.addptr %base_tensor, %updated : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
    %copy_ptrs = "gluon.extract_slice"(%ptrs) <{offsets = array<i64: 0, 0>}> : (tensor<128x128x!tt.ptr<f16>, #blocked>) -> tensor<32x128x!tt.ptr<f16>, #blocked>
    %mask = arith.constant dense<true> : tensor<32x128xi1, #blocked>
    %other = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked>
    %buffer = ttg.local_alloc : () -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    %token = ttg.async_copy_global_to_local %copy_ptrs, %buffer mask %mask other %other {intrinsic = true} : tensor<32x128x!tt.ptr<f16>, #blocked> -> <32x128xf16, #shared, #smem, mutable>
    tt.return
  }
}
