// RUN: triton-opt --split-input-file \
// RUN:   --tritonmetaxgpu-gluon-select-c500-layout-transfers \
// RUN:   --tritonmetaxgpu-gluon-select-c500-layout-transfers %s | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], elementsMNK = [2, 16, 4], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // Both lowerings have two-element shared transactions and two-way read/write
  // conflicts. Temporal decomposition therefore preserves the modeled access
  // quality while reducing scratch from 32768 to 16384 elements. The generic
  // 64-KiB allocation fits, so this specifically tests the Pareto policy.
  // CHECK-LABEL: tt.func public @no_block_terminal_mma_store
  // CHECK: ttg.convert_layout %arg0 {
  // CHECK-SAME: mxg.gluon_c500_repeated_shared
  // CHECK-SAME: mxg.shared_mem_force_no_vec
  tt.func public @no_block_terminal_mma_store(
      %value: tensor<128x256xf16, #mma>,
      %pointer: tensor<128x256x!tt.ptr<f16>, #blocked>,
      %mask: tensor<128x256xi1, #blocked>) {
    %converted = ttg.convert_layout %value : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked>
    tt.store %pointer, %converted, %mask : tensor<128x256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], elementsMNK = [2, 16, 4], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // Scratch would fall from 32768 to 4096 elements, but the repeated lowering
  // reduces each shared transaction from eight elements to two. Scratch alone
  // is not sufficient evidence when the generic 64-KiB allocation still fits,
  // so a stale selection is removed.
  // CHECK-LABEL: tt.func public @vector_width_regression
  // CHECK: ttg.convert_layout %arg0 :
  // CHECK-NOT: mxg.gluon_c500_repeated_shared
  // CHECK-NOT: mxg.shared_mem_force_no_vec
  // CHECK: tt.store
  tt.func public @vector_width_regression(
      %value: tensor<128x256xf16, #mma>,
      %pointer: tensor<128x256x!tt.ptr<f16>, #blocked>,
      %mask: tensor<128x256xi1, #blocked>) {
    %converted = ttg.convert_layout %value {mxg.gluon_c500_repeated_shared, mxg.shared_mem_force_no_vec} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked>
    tt.store %pointer, %converted, %mask : tensor<128x256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], elementsMNK = [2, 16, 4], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // The generic scratch is 65536 f16 elements, or 128 KiB, and therefore
  // cannot run on C500 regardless of other allocations. The 16-KiB repeated
  // form is selected as a hard-capacity rescue even though its shared vector
  // width is narrower. Final allocation analysis still checks the whole
  // kernel against 64 KiB.
  // CHECK-LABEL: tt.func public @hard_capacity_rescue
  // CHECK: ttg.convert_layout %arg0 {
  // CHECK-SAME: mxg.gluon_c500_repeated_shared
  // CHECK-SAME: mxg.shared_mem_force_no_vec
  tt.func public @hard_capacity_rescue(
      %value: tensor<256x256xf16, #mma>,
      %pointer: tensor<256x256x!tt.ptr<f16>, #blocked>,
      %mask: tensor<256x256xi1, #blocked>) {
    %converted = ttg.convert_layout %value : tensor<256x256xf16, #mma> -> tensor<256x256xf16, #blocked>
    tt.store %pointer, %converted, %mask : tensor<256x256x!tt.ptr<f16>, #blocked>
    tt.return
  }

  // Capacity rescue is a legality decision, so it also applies when the
  // converted value has a non-store consumer. The terminal-only restriction
  // remains a profitability guard for transfers whose generic form fits.
  // CHECK-LABEL: tt.func public @nonterminal_hard_capacity_rescue
  // CHECK: ttg.convert_layout %arg0 {
  // CHECK-SAME: mxg.gluon_c500_repeated_shared
  // CHECK-SAME: mxg.shared_mem_force_no_vec
  // CHECK: arith.addf
  tt.func public @nonterminal_hard_capacity_rescue(
      %value: tensor<256x256xf16, #mma>) {
    %converted = ttg.convert_layout %value : tensor<256x256xf16, #mma> -> tensor<256x256xf16, #blocked>
    %sum = arith.addf %converted, %converted : tensor<256x256xf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], elementsMNK = [1, 4, 8], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // The generic transfer already has temporal repetition, so the selector
  // preserves its vectorized lowering.
  // CHECK-LABEL: tt.func public @already_repeated_terminal_store
  // CHECK: ttg.convert_layout %arg0 :
  // CHECK-NOT: mxg.shared_mem_force_no_vec
  // CHECK: tt.store
  tt.func public @already_repeated_terminal_store(
      %value: tensor<128x128xf16, #mma>,
      %pointer: tensor<128x128x!tt.ptr<f16>, #blocked>,
      %mask: tensor<128x128xi1, #blocked>) {
    %converted = ttg.convert_layout %value : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #blocked>
    tt.store %pointer, %converted, %mask : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], elementsMNK = [2, 16, 4], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // A stale selection owned by this pass is removed when profitability facts
  // change. Running the pass twice must leave the same clean result.
  // CHECK-LABEL: tt.func public @owned_nonterminal_mma_transfer
  // CHECK: %[[CONVERTED:.*]] = ttg.convert_layout %arg0 :
  // CHECK-NOT: mxg.gluon_c500_repeated_shared
  // CHECK-NOT: mxg.shared_mem_force_no_vec
  // CHECK: %{{.*}} = arith.addf %[[CONVERTED]], %[[CONVERTED]]
  tt.func public @owned_nonterminal_mma_transfer(
      %value: tensor<128x256xf16, #mma>) {
    %converted = ttg.convert_layout %value {mxg.gluon_c500_repeated_shared, mxg.shared_mem_force_no_vec} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked>
    %sum = arith.addf %converted, %converted : tensor<128x256xf16, #blocked>
    tt.return
  }

  // A lowering request without this pass's provenance is externally owned and
  // must be preserved even when this planner would choose the generic form.
  // CHECK-LABEL: tt.func public @external_nonterminal_mma_transfer
  // CHECK: ttg.convert_layout %arg0 {mxg.shared_mem_force_no_vec}
  tt.func public @external_nonterminal_mma_transfer(
      %value: tensor<128x256xf16, #mma>) {
    %converted = ttg.convert_layout %value {mxg.shared_mem_force_no_vec} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked>
    %sum = arith.addf %converted, %converted : tensor<128x256xf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], elementsMNK = [2, 16, 4], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // Repeated shared transfer adds synchronization. Even with terminal store
  // users, a transfer nested in a loop remains on a repeated critical path.
  // CHECK-LABEL: tt.func public @loop_terminal_store
  // CHECK: scf.for
  // CHECK: ttg.convert_layout %arg0 :
  // CHECK-NOT: mxg.gluon_c500_repeated_shared
  // CHECK-NOT: mxg.shared_mem_force_no_vec
  // CHECK: tt.store
  tt.func public @loop_terminal_store(
      %value: tensor<128x256xf16, #mma>,
      %pointer: tensor<128x256x!tt.ptr<f16>, #blocked>,
      %mask: tensor<128x256xi1, #blocked>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %iv = %c0 to %c1 step %c1 {
      %converted = ttg.convert_layout %value {mxg.gluon_c500_repeated_shared, mxg.shared_mem_force_no_vec} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked>
      tt.store %pointer, %converted, %mask : tensor<128x256x!tt.ptr<f16>, #blocked>
    }
    tt.return
  }
}
