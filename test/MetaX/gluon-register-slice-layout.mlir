// RUN: split-file %s %t
// RUN: triton-opt --split-input-file \
// RUN:   --tritonmetaxgpu-gluon-propagate-layout \
// RUN:   --tritonmetaxgpu-gluon-resolve-placeholder-layouts \
// RUN:   --tritonmetaxgpu-gluon-legalize-c500-async-copy-layout \
// RUN:   --tritonmetaxgpu-gluon-legalize-register-slices \
// RUN:   --tritonmetaxgpu-gluon-verify-layout-contracts \
// RUN:   --tritonmetaxgpu-insert-gvm-arrive-barrier-shared \
// RUN:   --tritonmetaxgpu-gluon-verify-synchronization \
// RUN:   %t/positive.mlir | FileCheck %t/positive.mlir
// RUN: triton-opt --verify-diagnostics \
// RUN:   --tritonmetaxgpu-gluon-legalize-register-slices \
// RUN:   %t/target-invalid.mlir -o /dev/null

//--- positive.mlir

#full = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#same_logical = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], elementsMNK = [1, 1, 8], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // Gluon slices are logical views: shape may change, but the source, update,
  // and result must retain one exact encoding. The target may introduce a
  // different lowering-only carrier around ttg.extract/insert_tensor.
  // CHECK-LABEL: @blocked_gluon_slices
  // CHECK: ttg.convert_layout
  // CHECK: "ttg.extract_tensor"
  // CHECK-SAME: ctaIdx = array<i64: 1, 3, 5, 7>, elemIdx = array<i64: 0>
  // CHECK: ttg.convert_layout
  // CHECK: ttg.convert_layout
  // CHECK: "ttg.insert_tensor"
  // CHECK-SAME: ctaIdx = array<i64: 0, 2, 4, 6>, elemIdx = array<i64: 0>
  // CHECK: ttg.convert_layout
  tt.func public @blocked_gluon_slices(
      %source: tensor<16x128xf16, #full>,
      %base: tensor<16x128xf16, #full>,
      %update: tensor<16x64xf16, #full>)
      -> (tensor<16x64xf16, #full>, tensor<16x128xf16, #full>) {
    %slice = "gluon.extract_slice"(%source) <{offsets = array<i64: 0, 64>}> : (tensor<16x128xf16, #full>) -> tensor<16x64xf16, #full>
    %result = "gluon.insert_slice"(%base, %update) <{offsets = array<i64: 0, 0>}> : (tensor<16x128xf16, #full>, tensor<16x64xf16, #full>) -> tensor<16x128xf16, #full>
    tt.return %slice, %result : tensor<16x64xf16, #full>, tensor<16x128xf16, #full>
  }

  // The logical parent and child intentionally retain one ownership
  // encoding. The N subview cuts through a lane-owned dimension, so it cannot
  // be represented by TTG extract/insert indices directly. C500 must introduce
  // a lowering-only Blocked carrier, select a complete physical partition,
  // and convert back to the unchanged logical types.
  // CHECK-LABEL: @same_logical_layout_uses_physical_carrier
  // CHECK: ttg.convert_layout
  // CHECK: "ttg.extract_tensor"
  // CHECK-SAME: ctaIdx = array<i64: 5, 7>, elemIdx = array<i64: 0>
  // CHECK: ttg.convert_layout
  // CHECK: "ttg.insert_tensor"
  // CHECK-SAME: ctaIdx = array<i64: 5, 7>, elemIdx = array<i64: 0>
  // CHECK: ttg.convert_layout
  tt.func public @same_logical_layout_uses_physical_carrier(
      %source: tensor<32x64xf16, #same_logical>,
      %base: tensor<32x64xf16, #same_logical>,
      %update: tensor<16x32xf16, #same_logical>)
      -> (tensor<16x32xf16, #same_logical>,
          tensor<32x64xf16, #same_logical>) {
    %slice = "gluon.extract_slice"(%source)
        <{offsets = array<i64: 16, 32>}>
        : (tensor<32x64xf16, #same_logical>)
          -> tensor<16x32xf16, #same_logical>
    %result = "gluon.insert_slice"(%base, %update)
        <{offsets = array<i64: 16, 32>}>
        : (tensor<32x64xf16, #same_logical>,
           tensor<16x32xf16, #same_logical>)
          -> tensor<32x64xf16, #same_logical>
    tt.return %slice, %result
        : tensor<16x32xf16, #same_logical>,
          tensor<32x64xf16, #same_logical>
  }

  // Four logical fragments keep the parent's exact encoding. The target
  // chooses a lowering-only carrier because the logical encoding cannot
  // directly represent each 64-column physical selection.
  // CHECK-LABEL: @blocked_repeated_parent_slices
  // CHECK: ttg.convert_layout
  // CHECK: "ttg.extract_tensor"
  // CHECK-SAME: ctaIdx = array<i64: 0, 4, 8, 12>, elemIdx = array<i64: 0>
  // CHECK: ttg.convert_layout
  // CHECK: "ttg.extract_tensor"
  // CHECK-SAME: ctaIdx = array<i64: 1, 5, 9, 13>, elemIdx = array<i64: 0>
  // CHECK: ttg.convert_layout
  // CHECK: "ttg.extract_tensor"
  // CHECK-SAME: ctaIdx = array<i64: 2, 6, 10, 14>, elemIdx = array<i64: 0>
  // CHECK: ttg.convert_layout
  // CHECK: "ttg.extract_tensor"
  // CHECK-SAME: ctaIdx = array<i64: 3, 7, 11, 15>, elemIdx = array<i64: 0>
  // CHECK: ttg.convert_layout
  tt.func public @blocked_repeated_parent_slices(
      %source: tensor<16x256xf16, #full>)
      -> (tensor<16x64xf16, #full>,
          tensor<16x64xf16, #full>,
          tensor<16x64xf16, #full>,
          tensor<16x64xf16, #full>) {
    %s0 = "gluon.extract_slice"(%source)
        <{offsets = array<i64: 0, 0>}>
        : (tensor<16x256xf16, #full>) -> tensor<16x64xf16, #full>
    %s1 = "gluon.extract_slice"(%source)
        <{offsets = array<i64: 0, 64>}>
        : (tensor<16x256xf16, #full>) -> tensor<16x64xf16, #full>
    %s2 = "gluon.extract_slice"(%source)
        <{offsets = array<i64: 0, 128>}>
        : (tensor<16x256xf16, #full>) -> tensor<16x64xf16, #full>
    %s3 = "gluon.extract_slice"(%source)
        <{offsets = array<i64: 0, 192>}>
        : (tensor<16x256xf16, #full>) -> tensor<16x64xf16, #full>
    tt.return %s0, %s1, %s2, %s3
        : tensor<16x64xf16, #full>, tensor<16x64xf16, #full>,
          tensor<16x64xf16, #full>, tensor<16x64xf16, #full>
  }

  // A balanced MACA fragment is itself a complete exact-same-encoding
  // carrier. The Gluon view relation lowers directly without a second logical
  // layout or a conversion boundary.
  // CHECK-LABEL: @mma_same_logical_layout_is_physical_carrier
  // CHECK-NOT: ttg.convert_layout
  // CHECK: "ttg.extract_tensor"
  // CHECK-SAME: ctaIdx = array<i64: 15>, elemIdx = array<i64: 0>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: "ttg.insert_tensor"
  // CHECK-SAME: ctaIdx = array<i64: 15>, elemIdx = array<i64: 0>
  // CHECK-NOT: ttg.convert_layout
  tt.func public @mma_same_logical_layout_is_physical_carrier(
      %full: tensor<128x128xf32, #mma>) -> tensor<128x128xf32, #mma> {
    %sub = "gluon.extract_slice"(%full)
        <{offsets = array<i64: 96, 96>}>
        : (tensor<128x128xf32, #mma>) -> tensor<32x32xf32, #mma>
    %result = "gluon.insert_slice"(%full, %sub)
        <{offsets = array<i64: 96, 96>}>
        : (tensor<128x128xf32, #mma>, tensor<32x32xf32, #mma>)
          -> tensor<128x128xf32, #mma>
    tt.return %result : tensor<128x128xf32, #mma>
  }

}

// -----

#mma_operand_parent = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], elementsMNK = [1, 1, 8], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>
#dot_a_fragment = #ttg.dot_op<{opIdx = 0, parent = #mma_operand_parent}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // Backward inference may reuse a dot-operand encoding only after the target
  // register-slice planner proves a complete repeated-fragment partition.
  // CHECK-LABEL: @infer_proven_dot_operand_parent
  // CHECK-NOT: ttg.convert_layout
  // CHECK: "ttg.extract_tensor"
  // CHECK-SAME: ctaIdx = array<i64: 1>, elemIdx = array<i64: 0, 1, 2, 3, 4, 5, 6, 7>
  // CHECK-NOT: #gluon.auto_encoding
  tt.func public @infer_proven_dot_operand_parent()
      -> tensor<32x32xf16, #dot_a_fragment> {
    %full = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #gluon.auto_encoding>
    %fragment = "gluon.extract_slice"(%full) <{offsets = array<i64: 0, 32>}> : (tensor<32x128xf16, #gluon.auto_encoding>) -> tensor<32x32xf16, #gluon.auto_encoding>
    %seed = gluon.set_auto_layout %fragment : tensor<32x32xf16, #gluon.auto_encoding> -> tensor<32x32xf16, #dot_a_fragment>
    tt.return %seed : tensor<32x32xf16, #dot_a_fragment>
  }
}

//--- target-invalid.mlir

// Register-slice legalization delegates physical ABI selection to the target.
// A bounded but fragment-misaligned logical range must receive the target's
// unsupported-contract diagnostic even though source and result use the same
// logical encoding.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @reject_misaligned_logical_slice(
      %source: tensor<16x128xf16, #blocked>) {
    // expected-error@+1 {{C500 cannot form a complete physical register partition}}
    %slice = "gluon.extract_slice"(%source)
        <{offsets = array<i64: 0, 32>}>
        : (tensor<16x128xf16, #blocked>) -> tensor<16x64xf16, #blocked>
    tt.return
  }
}
