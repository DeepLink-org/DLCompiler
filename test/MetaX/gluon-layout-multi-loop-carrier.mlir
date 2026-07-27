// RUN: triton-opt \
// RUN:   --tritonmetaxgpu-gluon-propagate-layout \
// RUN:   --tritonmetaxgpu-gluon-resolve-placeholder-layouts \
// RUN:   %s | FileCheck %s --implicit-check-not='#gluon.auto_encoding'

#left = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#right = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-DAG: #[[$LEFT:blocked[0-9]*]] = #ttg.blocked<{{.*}}order = [1, 0]{{.*}}>
  // CHECK-DAG: #[[$RIGHT:blocked[0-9]*]] = #ttg.blocked<{{.*}}order = [0, 1]{{.*}}>
  // CHECK-LABEL: @independent_loop_carriers_keep_their_positions
  // CHECK: %[[LEFT_INIT:.*]] = arith.constant {{.*}} : tensor<32x32xf32, #[[$LEFT]]>
  // CHECK: %[[RIGHT_INIT:.*]] = arith.constant {{.*}} : tensor<32x32xf32, #[[$RIGHT]]>
  // CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[LEFT_INIT]], %{{.*}} = %[[RIGHT_INIT]]) -> (tensor<32x32xf32, #[[$LEFT]]>, tensor<32x32xf32, #[[$RIGHT]]>)
  tt.func public @independent_loop_carriers_keep_their_positions() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %left_init = arith.constant dense<0.000000e+00>
        : tensor<32x32xf32, #gluon.auto_encoding>
    %right_init = arith.constant dense<0.000000e+00>
        : tensor<32x32xf32, #gluon.auto_encoding>
    %left_seed = gluon.set_auto_layout %left_init
        : tensor<32x32xf32, #gluon.auto_encoding>
          -> tensor<32x32xf32, #left>
    %right_seed = gluon.set_auto_layout %right_init
        : tensor<32x32xf32, #gluon.auto_encoding>
          -> tensor<32x32xf32, #right>
    %result:2 = scf.for %i = %c0 to %c1 step %c1
        iter_args(%left = %left_init, %right = %right_init)
        -> (tensor<32x32xf32, #gluon.auto_encoding>,
            tensor<32x32xf32, #gluon.auto_encoding>) {
      scf.yield %left, %right
          : tensor<32x32xf32, #gluon.auto_encoding>,
            tensor<32x32xf32, #gluon.auto_encoding>
    }
    tt.return
  }
}
