// RUN: split-file %s %t
// RUN: %PYTHON %S/Inputs/verify_gluon_layout_candidate_bundle.py \
// RUN:   --no-dot %t/no-dot.mlir \
// RUN:   --single-dot %t/single-dot.mlir \
// RUN:   --two-dot %t/two-dot.mlir

// The production entry is the C++ CandidateBundle binding. Python receives
// finalized whole-module sources; there is no Transform interpreter, ordinal,
// operation path, or candidate marker protocol.

//--- no-dot.mlir

module attributes {
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 8 : i32,
  "ttg.target" = "cuda:80",
  "ttg.threads-per-warp" = 64 : i32
} {
  tt.func public @no_dot(%source: !tt.ptr<f16>,
                         %destination: !tt.ptr<f16>) {
    %offset = tt.make_range {start = 0 : i32, end = 64 : i32}
        : tensor<64xi32, #gluon.auto_encoding>
    %source_base = tt.splat %source
        : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>, #gluon.auto_encoding>
    %source_pointer = tt.addptr %source_base, %offset
        : tensor<64x!tt.ptr<f16>, #gluon.auto_encoding>,
          tensor<64xi32, #gluon.auto_encoding>
    %value = tt.load %source_pointer
        : tensor<64x!tt.ptr<f16>, #gluon.auto_encoding>
    %destination_base = tt.splat %destination
        : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>, #gluon.auto_encoding>
    %destination_pointer = tt.addptr %destination_base, %offset
        : tensor<64x!tt.ptr<f16>, #gluon.auto_encoding>,
          tensor<64xi32, #gluon.auto_encoding>
    tt.store %destination_pointer, %value
        : tensor<64x!tt.ptr<f16>, #gluon.auto_encoding>
    tt.return
  }
}

//--- single-dot.mlir

#single_a_shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#single_b_shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1]}>
#single_output = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 2], order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 8 : i32,
  "ttg.target" = "cuda:80",
  "ttg.threads-per-warp" = 64 : i32
} {
  tt.func public @single_dot(
      %a_smem: !ttg.memdesc<32x32xf16, #single_a_shared, #smem, mutable>,
      %b_smem: !ttg.memdesc<32x128xf16, #single_b_shared, #smem, mutable>,
      %output: !tt.ptr<f32>) {
    %a = ttg.local_load %a_smem :
        !ttg.memdesc<32x32xf16, #single_a_shared, #smem, mutable>
        -> tensor<32x32xf16, #gluon.auto_encoding>
    %b = ttg.local_load %b_smem :
        !ttg.memdesc<32x128xf16, #single_b_shared, #smem, mutable>
        -> tensor<32x128xf16, #gluon.auto_encoding>
    %c = arith.constant dense<0.000000e+00>
        : tensor<32x128xf32, #gluon.auto_encoding>
    %d = tt.dot %a, %b, %c, inputPrecision = tf32 :
        tensor<32x32xf16, #gluon.auto_encoding> *
        tensor<32x128xf16, #gluon.auto_encoding>
        -> tensor<32x128xf32, #gluon.auto_encoding>
    %output_tensor = tt.splat %output :
        !tt.ptr<f32> -> tensor<32x128x!tt.ptr<f32>, #single_output>
    %stored = ttg.convert_layout %d :
        tensor<32x128xf32, #gluon.auto_encoding>
        -> tensor<32x128xf32, #single_output>
    tt.store %output_tensor, %stored :
        tensor<32x128x!tt.ptr<f32>, #single_output>
    tt.return
  }
}

//--- two-dot.mlir

#two_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#two_output = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#two_smem = #ttg.shared_memory

module attributes {
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  "ttg.target" = "cuda:80",
  "ttg.threads-per-warp" = 64 : i32
} {
  // This is a producer chain rather than two independent/accumulating dots:
  // QK-C crosses an elementwise cast and becomes PV-A. Candidate finalization
  // must keep one common MMA profile and materialize only the local boundary.
  tt.func public @two_dots(%output: !tt.ptr<f32>) {
    %q_smem = ttg.local_alloc {"ttg.gluon.default-shared-layout"} : () ->
        !ttg.memdesc<32x32xf16, #two_shared, #two_smem, mutable>
    %k_smem = ttg.local_alloc {"ttg.gluon.default-shared-layout"} : () ->
        !ttg.memdesc<32x32xf16, #two_shared, #two_smem, mutable>
    %v_smem = ttg.local_alloc {"ttg.gluon.default-shared-layout"} : () ->
        !ttg.memdesc<32x32xf16, #two_shared, #two_smem, mutable>
    %q_init = arith.constant dense<1.000000e+00> :
        tensor<32x32xf16, #gluon.auto_encoding>
    %k_init = arith.constant dense<2.000000e+00> :
        tensor<32x32xf16, #gluon.auto_encoding>
    %v_init = arith.constant dense<3.000000e+00> :
        tensor<32x32xf16, #gluon.auto_encoding>
    ttg.local_store %q_init, %q_smem :
        tensor<32x32xf16, #gluon.auto_encoding> ->
        !ttg.memdesc<32x32xf16, #two_shared, #two_smem, mutable>
    ttg.local_store %k_init, %k_smem :
        tensor<32x32xf16, #gluon.auto_encoding> ->
        !ttg.memdesc<32x32xf16, #two_shared, #two_smem, mutable>
    ttg.local_store %v_init, %v_smem :
        tensor<32x32xf16, #gluon.auto_encoding> ->
        !ttg.memdesc<32x32xf16, #two_shared, #two_smem, mutable>
    ttg.barrier_shared
    %q = ttg.local_load %q_smem :
        !ttg.memdesc<32x32xf16, #two_shared, #two_smem, mutable>
        -> tensor<32x32xf16, #gluon.auto_encoding>
    %k = ttg.local_load %k_smem :
        !ttg.memdesc<32x32xf16, #two_shared, #two_smem, mutable>
        -> tensor<32x32xf16, #gluon.auto_encoding>
    %v = ttg.local_load %v_smem :
        !ttg.memdesc<32x32xf16, #two_shared, #two_smem, mutable>
        -> tensor<32x32xf16, #gluon.auto_encoding>
    %qk_zero = arith.constant dense<0.000000e+00>
        : tensor<32x32xf32, #gluon.auto_encoding>
    %qk = tt.dot %q, %k, %qk_zero, inputPrecision = tf32 :
        tensor<32x32xf16, #gluon.auto_encoding> *
        tensor<32x32xf16, #gluon.auto_encoding>
        -> tensor<32x32xf32, #gluon.auto_encoding>
    %p = arith.truncf %qk :
        tensor<32x32xf32, #gluon.auto_encoding>
        to tensor<32x32xf16, #gluon.auto_encoding>
    %pv_zero = arith.constant dense<0.000000e+00>
        : tensor<32x32xf32, #gluon.auto_encoding>
    %out = tt.dot %p, %v, %pv_zero, inputPrecision = tf32 :
        tensor<32x32xf16, #gluon.auto_encoding> *
        tensor<32x32xf16, #gluon.auto_encoding>
        -> tensor<32x32xf32, #gluon.auto_encoding>
    %output_tensor = tt.splat %output :
        !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #two_output>
    %stored = ttg.convert_layout %out :
        tensor<32x32xf32, #gluon.auto_encoding>
        -> tensor<32x32xf32, #two_output>
    tt.store %output_tensor, %stored :
        tensor<32x32x!tt.ptr<f32>, #two_output>
    tt.return
  }
}
