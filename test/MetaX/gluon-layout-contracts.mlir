// RUN: split-file %s %t
// RUN: triton-opt --split-input-file --verify-diagnostics \
// RUN:   --tritonmetaxgpu-gluon-verify-layout-contracts \
// RUN:   %t/layout-contracts.mlir -o /dev/null
// RUN: triton-opt --split-input-file --verify-diagnostics \
// RUN:   --tritonmetaxgpu-gluon-verify-layout-contracts \
// RUN:   %t/maca-verifier.mlir

//--- layout-contracts.mlir

#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 2], elementsMNK = [1, 4, 8], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // This is the C500 B-fragment ABI used by Flash dV/dK: both the packed i32
  // carrier and the f16 result contain exactly 32 registers per thread.
  tt.func public @accept_exact_b_fragment(
      %src: tensor<32x128xi32, #b>) {
    %result = "ttg.bsm_perm"(%src) : (tensor<32x128xi32, #b>) -> tensor<32x128xf16, #b>
    tt.return
  }
}

// -----

#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 4], elementsMNK = [1, 2, 4], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @reject_non_32_register_domain(
      %src: tensor<32x128xi32, #b>) {
    // expected-error@+1 {{bsm_perm does not satisfy the C500 physical permutation contract}}
    %result = "ttg.bsm_perm"(%src) : (tensor<32x128xi32, #b>) -> tensor<32x128xf16, #b>
    tt.return
  }
}

// -----

#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 2], elementsMNK = [1, 4, 8], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @reject_operand_a_role(
      %src: tensor<32x128xi32, #a>) {
    // expected-error@+1 {{bsm_perm does not satisfy the C500 physical permutation contract}}
    %result = "ttg.bsm_perm"(%src) : (tensor<32x128xi32, #a>) -> tensor<32x128xf16, #a>
    tt.return
  }
}

// -----

#src_mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 2], elementsMNK = [1, 4, 8], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>
#result_mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 4], elementsMNK = [1, 4, 8], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>
#src_b = #ttg.dot_op<{opIdx = 1, parent = #src_mma}>
#result_b = #ttg.dot_op<{opIdx = 1, parent = #result_mma}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @reject_parent_mismatch(
      %src: tensor<32x128xi32, #src_b>) {
    // expected-error@+1 {{bsm_perm does not satisfy the C500 physical permutation contract}}
    %result = "ttg.bsm_perm"(%src) : (tensor<32x128xi32, #src_b>) -> tensor<32x128xf16, #result_b>
    tt.return
  }
}

// -----

#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], elementsMNK = [1, 1, 4], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // The raw D64 accumulator layout replicates ownership across four warps.
  // Tensor-atomic lowering predicates away the upper two warps, leaving an
  // exact 2048-owner cover of the 32x64 logical tensor.
  tt.func public @predicated_maca_atomic_exact_cover(
      %ptr: tensor<32x64x!tt.ptr<f32>, #mma>,
      %value: tensor<32x64xf32, #mma>,
      %mask: tensor<32x64xi1, #mma>) {
    %unused = tt.atomic_rmw fadd, relaxed, gpu, %ptr, %value, %mask :
        (tensor<32x64x!tt.ptr<f32>, #mma>, tensor<32x64xf32, #mma>,
         tensor<32x64xi1, #mma>) -> tensor<32x64xf32, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], elementsMNK = [1, 1, 4], colMajor = 1, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @reject_unsupported_f32_col_major(
      %lhs: tensor<16x16xf32, #a>,
      %rhs: tensor<16x16xf32, #b>,
      %acc: tensor<16x16xf32, #mma>) {
    // expected-error@+1 {{C500 MMA accumulator order colMajor=1 is not supported for operand element types 'f32' and 'f32'}}
    %result = tt.dot %lhs, %rhs, %acc : tensor<16x16xf32, #a> * tensor<16x16xf32, #b> -> tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @reject_register_internal_replication(
      %ptr: tensor<1x!tt.ptr<f32>, #blocked>,
      %value: tensor<1xf32, #blocked>,
      %mask: tensor<1xi1, #blocked>) {
    // expected-error@+1 {{tensor atomic_rmw has replicated distributed ownership that is not made injective by the C500 tensor-atomic active-thread predicate}}
    %unused = tt.atomic_rmw fadd, relaxed, gpu, %ptr, %value, %mask :
        (tensor<1x!tt.ptr<f32>, #blocked>, tensor<1xf32, #blocked>,
         tensor<1xi1, #blocked>) -> tensor<1xf32, #blocked>
    tt.return
  }
}

// -----

// Physical register selections must be decodable and duplicate-free.

#full = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], elementsMNK = [4, 4, 4], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>
#sub = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], elementsMNK = [1, 1, 4], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @reject_duplicate_register_selection(
      %full: tensor<128x128xf32, #full>) {
    // expected-error@+1 {{cannot decode extract_tensor register indices}}
    %sub = "ttg.extract_tensor"(%full)
        <{ctaIdx = array<i64: 0>, elemIdx = array<i64: 0, 0>}>
        : (tensor<128x128xf32, #full>) -> tensor<32x32xf32, #sub>
    tt.return
  }
}

// -----

#full = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], elementsMNK = [4, 4, 4], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>
#wrong_sub = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], elementsMNK = [1, 1, 4], colMajor = 1, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @reject_insert_subtype_mismatch(
      %full: tensor<128x128xf32, #full>,
      %sub: tensor<32x32xf32, #wrong_sub>) {
    // expected-error@+1 {{ttg.insert_tensor input type does not match the physical subview inferred from its destination layout}}
    %result = "ttg.insert_tensor"(%full, %sub)
        <{ctaIdx = array<i64: 0>, elemIdx = array<i64: 0>}>
        : (tensor<128x128xf32, #full>, tensor<32x32xf32, #wrong_sub>)
          -> tensor<128x128xf32, #full>
    tt.return
  }
}

//--- maca-verifier.mlir

#mma = #ttg.maca_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], elementsMNK = [1, 1, 8], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>
module {
  // expected-error@+1 {{rejected malformed MACA MMA encoding in block argument #0: versionMajor must be 2, got 3}}
  tt.func public @bad_version(%arg0: tensor<32x32xf32, #mma>) {
    tt.return
  }
}

// -----

#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], elementsMNK = [1, 8], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>
module {
  // expected-error@+1 {{rejected malformed MACA MMA encoding in block argument #0: elementsMNK must contain M, N, and K, got 2 elements}}
  tt.func public @bad_elements_mnk_rank(%arg0: tensor<32x32xf32, #mma>) {
    tt.return
  }
}

// -----

#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], elementsMNK = [1, 0, 8], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>
module {
  // expected-error@+1 {{rejected malformed MACA MMA encoding in block argument #0: elementsMNK values must be nonzero powers of two}}
  tt.func public @bad_zero_element(%arg0: tensor<32x32xf32, #mma>) {
    tt.return
  }
}

// -----

#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 6, warpsPerCTA = [2, 2], elementsMNK = [1, 1, 8], colMajor = 0, isATrans = true, isBTrans = false, elementsStride = [3, 1]}>
module {
  // expected-error@+1 {{rejected malformed MACA MMA encoding in block argument #0: A transpose stride 3 must divide elementsMNK K 8}}
  tt.func public @bad_transpose_stride(%arg0: tensor<32x32xf32, #mma>) {
    tt.return
  }
}

// -----

#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 6, warpsPerCTA = [2, 2], elementsMNK = [1, 1, 8], colMajor = 0, isATrans = true, isBTrans = false, elementsStride = [4, 1]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma}>
module {
  // expected-error@+1 {{does not support MACA dot-operand layouts whose parent uses isATrans/isBTrans}}
  tt.func public @unsupported_dot_parent_transpose(
      %arg0: tensor<32x32xf32, #dot_a>) {
    tt.return
  }
}
