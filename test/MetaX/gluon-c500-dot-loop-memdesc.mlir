// RUN: triton-opt \
// RUN:   --tritonmetaxgpu-gluon-insert-require-layout='compute-capability=80' \
// RUN:   --tritonmetaxgpu-gluon-propagate-layout \
// RUN:   --tritonmetaxgpu-gluon-resolve-placeholder-layouts \
// RUN:   --tritonmetaxgpu-gluon-verify-layout-contracts %s | FileCheck %s \
// RUN:   --implicit-check-not='#gluon.auto_encoding' \
// RUN:   --implicit-check-not='gluon.set_auto_layout' \
// RUN:   --implicit-check-not='gluon.require_layout'

#a_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#b_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // A RegionBranch predecessor is not itself a join. The initial and yielded
  // memdesc_index values must retain their defining view paths, allowing the
  // loop block arguments and downstream subslices to reach one exact root.
  // CHECK-LABEL: tt.func public @loop_carried_index_subslice_dot
  // CHECK: %[[A_ROOT:.*]] = ttg.local_alloc {{.*}} !ttg.memdesc<2x32x32xf16, #[[$A_SHARED:[A-Za-z0-9_]+]], #smem, mutable>
  // CHECK: %[[B_ROOT:.*]] = ttg.local_alloc {{.*}} !ttg.memdesc<2x32x32xf16, #[[$B_SHARED:[A-Za-z0-9_]+]], #smem, mutable>
  // CHECK: scf.for {{.*}} iter_args({{.*}}) -> (tensor<32x32xf32, {{.*}}>, !ttg.memdesc<32x32xf16, #[[$A_SHARED]], #smem, mutable>, !ttg.memdesc<32x32xf16, #[[$B_SHARED]], #smem, mutable>)
  // CHECK: ttg.memdesc_subslice
  // CHECK: ttg.local_load
  // CHECK: tt.dot
  // CHECK: ttg.memdesc_index %[[A_ROOT]]
  // CHECK: ttg.memdesc_index %[[B_ROOT]]
  tt.func public @loop_carried_index_subslice_dot() {
    %a_root = ttg.local_alloc {"ttg.gluon.default-shared-layout"} : () -> !ttg.memdesc<2x32x32xf16, #a_shared, #smem, mutable>
    %b_root = ttg.local_alloc {"ttg.gluon.default-shared-layout"} : () -> !ttg.memdesc<2x32x32xf16, #b_shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %a_init = ttg.memdesc_index %a_root[%c0] : !ttg.memdesc<2x32x32xf16, #a_shared, #smem, mutable> -> !ttg.memdesc<32x32xf16, #a_shared, #smem, mutable>
    %b_init = ttg.memdesc_index %b_root[%c0] : !ttg.memdesc<2x32x32xf16, #b_shared, #smem, mutable> -> !ttg.memdesc<32x32xf16, #b_shared, #smem, mutable>
    %a_tile = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #gluon.auto_encoding>
    %b_tile = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #gluon.auto_encoding>
    ttg.local_store %a_tile, %a_init : tensor<32x32xf16, #gluon.auto_encoding> -> !ttg.memdesc<32x32xf16, #a_shared, #smem, mutable>
    ttg.local_store %b_tile, %b_init : tensor<32x32xf16, #gluon.auto_encoding> -> !ttg.memdesc<32x32xf16, #b_shared, #smem, mutable>
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #gluon.auto_encoding>
    %result:3 = scf.for %iv = %c0 to %c2 step %c1 iter_args(%acc = %zero, %a_stage = %a_init, %b_stage = %b_init) -> (tensor<32x32xf32, #gluon.auto_encoding>, !ttg.memdesc<32x32xf16, #a_shared, #smem, mutable>, !ttg.memdesc<32x32xf16, #b_shared, #smem, mutable>)  : i32 {
      %a_slice = ttg.memdesc_subslice %a_stage[0, 0] : !ttg.memdesc<32x32xf16, #a_shared, #smem, mutable> -> !ttg.memdesc<32x16xf16, #a_shared, #smem, mutable, 32x32>
      %b_slice = ttg.memdesc_subslice %b_stage[0, 0] : !ttg.memdesc<32x32xf16, #b_shared, #smem, mutable> -> !ttg.memdesc<16x32xf16, #b_shared, #smem, mutable, 32x32>
      %a = ttg.local_load %a_slice {intrinsic = true} : !ttg.memdesc<32x16xf16, #a_shared, #smem, mutable, 32x32> -> tensor<32x16xf16, #gluon.auto_encoding>
      %b = ttg.local_load %b_slice {intrinsic = true} : !ttg.memdesc<16x32xf16, #b_shared, #smem, mutable, 32x32> -> tensor<16x32xf16, #gluon.auto_encoding>
      %next_acc = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<32x16xf16, #gluon.auto_encoding> * tensor<16x32xf16, #gluon.auto_encoding> -> tensor<32x32xf32, #gluon.auto_encoding>
      %next_index = arith.remsi %iv, %c2 : i32
      %next_a = ttg.memdesc_index %a_root[%next_index] : !ttg.memdesc<2x32x32xf16, #a_shared, #smem, mutable> -> !ttg.memdesc<32x32xf16, #a_shared, #smem, mutable>
      %next_b = ttg.memdesc_index %b_root[%next_index] : !ttg.memdesc<2x32x32xf16, #b_shared, #smem, mutable> -> !ttg.memdesc<32x32xf16, #b_shared, #smem, mutable>
      ttg.local_store %a_tile, %next_a : tensor<32x32xf16, #gluon.auto_encoding> -> !ttg.memdesc<32x32xf16, #a_shared, #smem, mutable>
      ttg.local_store %b_tile, %next_b : tensor<32x32xf16, #gluon.auto_encoding> -> !ttg.memdesc<32x32xf16, #b_shared, #smem, mutable>
      scf.yield %next_acc, %next_a, %next_b : tensor<32x32xf32, #gluon.auto_encoding>, !ttg.memdesc<32x32xf16, #a_shared, #smem, mutable>, !ttg.memdesc<32x32xf16, #b_shared, #smem, mutable>
    }
    tt.return
  }
}
