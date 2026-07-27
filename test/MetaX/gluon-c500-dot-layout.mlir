// RUN: split-file %s %t
// RUN: triton-opt \
// RUN:   --tritonmetaxgpu-gluon-insert-require-layout='compute-capability=80' \
// RUN:   %t/selected.mlir | FileCheck %t/selected.mlir --check-prefix=INSERT \
// RUN:   --implicit-check-not='gluon.require_layout {{.*}} : !ttg.memdesc'
// RUN: triton-opt \
// RUN:   --tritonmetaxgpu-gluon-insert-require-layout='compute-capability=80' \
// RUN:   --tritonmetaxgpu-gluon-propagate-layout \
// RUN:   --tritonmetaxgpu-gluon-resolve-placeholder-layouts \
// RUN:   --tritonmetaxgpu-gluon-verify-layout-contracts \
// RUN:   %t/selected.mlir | FileCheck %t/selected.mlir --check-prefix=FINAL \
// RUN:   --implicit-check-not='#gluon.auto_encoding' \
// RUN:   --implicit-check-not='#gluon.no_verify_encoding' \
// RUN:   --implicit-check-not='gluon.require_layout' \
// RUN:   --implicit-check-not='gluon.release_layout' \
// RUN:   --implicit-check-not='gluon.set_auto_layout'
// RUN: triton-opt \
// RUN:   --verify-each=0 \
// RUN:   --tritonmetaxgpu-gluon-insert-require-layout='compute-capability=80' \
// RUN:   --tritonmetaxgpu-gluon-propagate-layout \
// RUN:   --tritonmetaxgpu-gluon-resolve-placeholder-layouts \
// RUN:   --tritonmetaxgpu-gluon-verify-layout-contracts \
// RUN:   %t/rank1-dot-row-store.mlir | \
// RUN:   FileCheck %t/rank1-dot-row-store.mlir \
// RUN:   --implicit-check-not='#gluon.auto_encoding' \
// RUN:   --implicit-check-not='gluon.require_layout' \
// RUN:   --implicit-check-not='gluon.set_auto_layout'
// RUN: triton-opt \
// RUN:   --tritonmetaxgpu-gluon-insert-require-layout='compute-capability=80' \
// RUN:   --tritonmetaxgpu-gluon-propagate-layout \
// RUN:   --tritonmetaxgpu-gluon-resolve-placeholder-layouts \
// RUN:   --tritonmetaxgpu-gluon-verify-layout-contracts \
// RUN:   %t/direct-global-dot.mlir | \
// RUN:   FileCheck %t/direct-global-dot.mlir \
// RUN:   --implicit-check-not='#gluon.auto_encoding' \
// RUN:   --implicit-check-not='gluon.require_layout' \
// RUN:   --implicit-check-not='gluon.set_auto_layout' \
// RUN:   --implicit-check-not='ttg.convert_layout'
// RUN: triton-opt \
// RUN:   --tritonmetaxgpu-gluon-insert-require-layout='compute-capability=80' \
// RUN:   %t/bsm-carrier.mlir | FileCheck %t/bsm-carrier.mlir \
// RUN:   --check-prefix=BSM
// RUN: triton-opt \
// RUN:   --tritonmetaxgpu-gluon-insert-require-layout='compute-capability=80' \
// RUN:   %t/fixed-view-lowerable.mlir | \
// RUN:   FileCheck %t/fixed-view-lowerable.mlir --check-prefix=FIXED \
// RUN:   --implicit-check-not='gluon.require_layout {{.*}} : !ttg.memdesc'
// RUN: triton-opt \
// RUN:   --tritonmetaxgpu-gluon-insert-require-layout='compute-capability=80' \
// RUN:   %t/reinterpret-fixed-root.mlir | \
// RUN:   FileCheck %t/reinterpret-fixed-root.mlir --check-prefix=REINTERPRET-FIXED \
// RUN:   --implicit-check-not='gluon.require_layout {{.*}} : !ttg.memdesc'
// RUN: triton-opt \
// RUN:   --tritonmetaxgpu-gluon-insert-require-layout='compute-capability=80' \
// RUN:   --tritonmetaxgpu-gluon-propagate-layout \
// RUN:   --tritonmetaxgpu-gluon-resolve-placeholder-layouts \
// RUN:   --tritonmetaxgpu-gluon-legalize-c500-async-copy-layout \
// RUN:   --tritonmetaxgpu-gluon-verify-layout-contracts \
// RUN:   %t/reinterpret-managed-root.mlir | \
// RUN:   FileCheck %t/reinterpret-managed-root.mlir \
// RUN:   --check-prefix=REINTERPRET-MANAGED \
// RUN:   --implicit-check-not='ttg.gluon.default-shared-layout' \
// RUN:   --implicit-check-not='gluon.require_layout' \
// RUN:   --implicit-check-not='#gluon.auto_encoding'

//--- selected.mlir

#a_shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#b_shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1]}>
#output = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 2], order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // INSERT-DAG: #{{.*}} = #ttg.maca_mma<{{.*}}warpsPerCTA = [1, 8]{{.*}}elementsMNK = [2, 1, 8]
  // INSERT-DAG: #[[$A_SHARED:.*]] = #ttg.swizzled_shared<{{.*}}order = [1, 0]
  // INSERT-DAG: #[[$B_SHARED:.*]] = #ttg.swizzled_shared<{{.*}}order = [0, 1]
  // INSERT-DAG: #[[$SMEM:.*]] = #ttg.shared_memory
  // INSERT-LABEL: tt.func public @selected_dot
  // Fixed function-argument shared families keep their source-authored
  // encodings. Only the loaded register values receive selected dot contracts.
  // INSERT: ttg.local_load
  // INSERT: ttg.local_load
  // INSERT: gluon.set_auto_layout
  // INSERT: gluon.require_layout
  // INSERT: gluon.require_layout
  // INSERT: tt.dot
  // INSERT: gluon.set_auto_layout
  // FINAL-LABEL: tt.func public @selected_dot
  // FINAL: ttg.local_load
  // FINAL: tt.dot
  // FINAL: ttg.convert_layout
  tt.func public @selected_dot(
      %a_smem: !ttg.memdesc<32x32xf16, #a_shared, #smem, mutable>,
      %b_smem: !ttg.memdesc<32x128xf16, #b_shared, #smem, mutable>,
      %output_ptr: !tt.ptr<f32>) {
    %a = ttg.local_load %a_smem :
        !ttg.memdesc<32x32xf16, #a_shared, #smem, mutable>
        -> tensor<32x32xf16, #gluon.auto_encoding>
    %b = ttg.local_load %b_smem :
        !ttg.memdesc<32x128xf16, #b_shared, #smem, mutable>
        -> tensor<32x128xf16, #gluon.auto_encoding>
    %c = arith.constant dense<0.000000e+00>
        : tensor<32x128xf32, #gluon.auto_encoding>
    %d = tt.dot %a, %b, %c, inputPrecision = tf32 :
        tensor<32x32xf16, #gluon.auto_encoding> *
        tensor<32x128xf16, #gluon.auto_encoding>
        -> tensor<32x128xf32, #gluon.auto_encoding>
    %pointers = tt.splat %output_ptr :
        !tt.ptr<f32> -> tensor<32x128x!tt.ptr<f32>, #output>
    %stored = ttg.convert_layout %d :
        tensor<32x128xf32, #gluon.auto_encoding>
        -> tensor<32x128xf32, #output>
    tt.store %pointers, %stored : tensor<32x128x!tt.ptr<f32>, #output>
    tt.return
  }
}

//--- reinterpret-fixed-root.mlir

#fixed_a = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#fixed_b = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 4, order = [0, 1]}>
#fixed_smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // A concrete reinterpret is a source-authored shared-layout boundary even
  // when its encoding happens to equal the allocation encoding.
  // REINTERPRET-FIXED-LABEL: tt.func public @reinterpret_fixed_root
  // REINTERPRET-FIXED: %[[A_VIEW:.*]] = ttg.memdesc_reinterpret
  // REINTERPRET-FIXED: %[[B_VIEW:.*]] = ttg.memdesc_reinterpret
  // REINTERPRET-FIXED: %[[A:.*]] = ttg.local_load %[[A_VIEW]]
  // REINTERPRET-FIXED: %[[B:.*]] = ttg.local_load %[[B_VIEW]]
  // REINTERPRET-FIXED: gluon.require_layout %[[A]]
  // REINTERPRET-FIXED: gluon.require_layout %[[B]]
  tt.func public @reinterpret_fixed_root() {
    %a_storage = ttg.local_alloc {"ttg.gluon.default-shared-layout"} : () ->
        !ttg.memdesc<32x128xf16, #fixed_a, #fixed_smem, mutable>
    %b_storage = ttg.local_alloc {"ttg.gluon.default-shared-layout"} : () ->
        !ttg.memdesc<128x128xf16, #fixed_b, #fixed_smem, mutable>
    %a_smem = ttg.memdesc_reinterpret %a_storage :
        !ttg.memdesc<32x128xf16, #fixed_a, #fixed_smem, mutable> ->
        !ttg.memdesc<32x128xf16, #fixed_a, #fixed_smem, mutable>
    %b_smem = ttg.memdesc_reinterpret %b_storage :
        !ttg.memdesc<128x128xf16, #fixed_b, #fixed_smem, mutable> ->
        !ttg.memdesc<128x128xf16, #fixed_b, #fixed_smem, mutable>
    %a = ttg.local_load %a_smem {intrinsic = true, isConstantOffs = true} :
        !ttg.memdesc<32x128xf16, #fixed_a, #fixed_smem, mutable>
        -> tensor<32x128xf16, #gluon.auto_encoding>
    %b = ttg.local_load %b_smem {intrinsic = true, isConstantOffs = true} :
        !ttg.memdesc<128x128xf16, #fixed_b, #fixed_smem, mutable>
        -> tensor<128x128xf16, #gluon.auto_encoding>
    %c = arith.constant dense<0.000000e+00>
        : tensor<32x128xf32, #gluon.auto_encoding>
    %d = tt.dot %a, %b, %c, inputPrecision = tf32 :
        tensor<32x128xf16, #gluon.auto_encoding> *
        tensor<128x128xf16, #gluon.auto_encoding>
        -> tensor<32x128xf32, #gluon.auto_encoding>
    tt.return
  }
}

//--- reinterpret-managed-root.mlir

#storage_a = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#storage_b = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#placeholder_a = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#placeholder_b = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#managed_smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // The marker makes each reinterpret result an independent compiler-managed
  // family root. DotOperand requirements retag those results but cannot cross
  // the reinterpret boundary and rewrite either source allocation.
  // REINTERPRET-MANAGED-DAG: #[[$STORAGE_A:.*]] = #ttg.swizzled_shared<{{.*}}vec = 1{{.*}}order = [1, 0]
  // REINTERPRET-MANAGED-DAG: #[[$STORAGE_B:.*]] = #ttg.swizzled_shared<{{.*}}vec = 1{{.*}}order = [0, 1]
  // REINTERPRET-MANAGED-DAG: #[[$A_SHARED:.*]] = #ttg.swizzled_shared<{{.*}}order = [1, 0]
  // REINTERPRET-MANAGED-DAG: #[[$B_SHARED:.*]] = #ttg.swizzled_shared<{{.*}}order = [0, 1]
  // REINTERPRET-MANAGED-LABEL: tt.func public @reinterpret_managed_root
  // REINTERPRET-MANAGED: %[[A_STORAGE:.*]] = ttg.local_alloc : () -> !ttg.memdesc<32x128xf16, #[[$STORAGE_A]], #[[$SMEM:.*]], mutable>
  // REINTERPRET-MANAGED: %[[B_STORAGE:.*]] = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #[[$STORAGE_B]], #[[$SMEM]], mutable>
  // REINTERPRET-MANAGED: %[[A_VIEW:.*]] = ttg.memdesc_reinterpret %[[A_STORAGE]] : !ttg.memdesc<32x128xf16, #[[$STORAGE_A]], #[[$SMEM]], mutable> -> !ttg.memdesc<32x128xf16, #[[$A_SHARED]], #[[$SMEM]], mutable>
  // REINTERPRET-MANAGED: %[[B_VIEW:.*]] = ttg.memdesc_reinterpret %[[B_STORAGE]] : !ttg.memdesc<128x128xf16, #[[$STORAGE_B]], #[[$SMEM]], mutable> -> !ttg.memdesc<128x128xf16, #[[$B_SHARED]], #[[$SMEM]], mutable>
  // REINTERPRET-MANAGED: %[[A:.*]] = ttg.local_load %[[A_VIEW]]
  // REINTERPRET-MANAGED: %[[B:.*]] = ttg.local_load %[[B_VIEW]]
  // REINTERPRET-MANAGED: tt.dot %[[A]], %[[B]]
  tt.func public @reinterpret_managed_root() {
    %a_storage = ttg.local_alloc : () ->
        !ttg.memdesc<32x128xf16, #storage_a, #managed_smem, mutable>
    %b_storage = ttg.local_alloc : () ->
        !ttg.memdesc<128x128xf16, #storage_b, #managed_smem, mutable>
    %a_smem = ttg.memdesc_reinterpret %a_storage {"ttg.gluon.default-shared-layout"} :
        !ttg.memdesc<32x128xf16, #storage_a, #managed_smem, mutable> ->
        !ttg.memdesc<32x128xf16, #placeholder_a, #managed_smem, mutable>
    %b_smem = ttg.memdesc_reinterpret %b_storage {"ttg.gluon.default-shared-layout"} :
        !ttg.memdesc<128x128xf16, #storage_b, #managed_smem, mutable> ->
        !ttg.memdesc<128x128xf16, #placeholder_b, #managed_smem, mutable>
    %a = ttg.local_load %a_smem {intrinsic = true, isConstantOffs = true} :
        !ttg.memdesc<32x128xf16, #placeholder_a, #managed_smem, mutable>
        -> tensor<32x128xf16, #gluon.auto_encoding>
    %b = ttg.local_load %b_smem {intrinsic = true, isConstantOffs = true} :
        !ttg.memdesc<128x128xf16, #placeholder_b, #managed_smem, mutable>
        -> tensor<128x128xf16, #gluon.auto_encoding>
    %c = arith.constant dense<0.000000e+00>
        : tensor<32x128xf32, #gluon.auto_encoding>
    %d = tt.dot %a, %b, %c, inputPrecision = tf32 :
        tensor<32x128xf16, #gluon.auto_encoding> *
        tensor<128x128xf16, #gluon.auto_encoding>
        -> tensor<32x128xf32, #gluon.auto_encoding>
    tt.return
  }
}

//--- rank1-dot-row-store.mlir

#a_shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#b_shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // A rank-1 store derived from a selected dot retains the accumulator row
  // ownership across pointer, value, and mask instead of using shared scratch.
  // CHECK-LABEL: tt.func public @rank1_dot_row_store
  // CHECK: %[[ROW:.*]] = "tt.reduce"
  // CHECK: }) : {{.*}} -> tensor<32xf32, #[[$ROW_LAYOUT:.*]]>
  // CHECK-NOT: ttg.convert_layout {{.*}} tensor<32xf32
  // CHECK: tt.store %{{.*}}, %[[ROW]] : tensor<32x!tt.ptr<f32>, #[[$ROW_LAYOUT]]>
  tt.func public @rank1_dot_row_store(
      %a_smem: !ttg.memdesc<32x32xf16, #a_shared, #smem, mutable>,
      %b_smem: !ttg.memdesc<32x128xf16, #b_shared, #smem, mutable>,
      %row_ptr: !tt.ptr<f32>) {
    %a = ttg.local_load %a_smem :
        !ttg.memdesc<32x32xf16, #a_shared, #smem, mutable>
        -> tensor<32x32xf16, #gluon.auto_encoding>
    %b = ttg.local_load %b_smem :
        !ttg.memdesc<32x128xf16, #b_shared, #smem, mutable>
        -> tensor<32x128xf16, #gluon.auto_encoding>
    %c = arith.constant dense<0.000000e+00>
        : tensor<32x128xf32, #gluon.auto_encoding>
    %d = tt.dot %a, %b, %c, inputPrecision = tf32 :
        tensor<32x32xf16, #gluon.auto_encoding> *
        tensor<32x128xf16, #gluon.auto_encoding>
        -> tensor<32x128xf32, #gluon.auto_encoding>
    %row = "tt.reduce"(%d) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }) : (tensor<32x128xf32, #gluon.auto_encoding>)
        -> tensor<32xf32, #gluon.auto_encoding>
    %row_offsets = tt.make_range {start = 0 : i32, end = 32 : i32}
        : tensor<32xi32, #gluon.auto_encoding>
    %row_base = tt.splat %row_ptr
        : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #gluon.auto_encoding>
    %row_pointers = tt.addptr %row_base, %row_offsets
        : tensor<32x!tt.ptr<f32>, #gluon.auto_encoding>,
          tensor<32xi32, #gluon.auto_encoding>
    tt.store %row_pointers, %row
        : tensor<32x!tt.ptr<f32>, #gluon.auto_encoding>
    tt.return
  }
}

//--- direct-global-dot.mlir

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // A selected DotOperand may own the complete global-load component only
  // when its order and per-thread vector width lie inside the address-proven
  // coalesced domain. No separate global layout candidate is introduced.
  // CHECK-DAG: #[[$MMA:.*]] = #ttg.maca_mma
  // CHECK-LABEL: tt.func public @direct_global_dot_closes_selected_plan
  // CHECK: %[[A_PTRS:.*]] = tt.splat %{{.*}} {{.*}} -> tensor<32x32x!tt.ptr<f16>, #[[$A:.*]]>
  // CHECK: %[[B_PTRS:.*]] = tt.splat %{{.*}} {{.*}} -> tensor<32x128x!tt.ptr<f16>, #[[$B:.*]]>
  // CHECK: %[[A_VALUE:.*]] = tt.load %[[A_PTRS]] : tensor<32x32x!tt.ptr<f16>, #[[$A]]>
  // CHECK: %[[B_VALUE:.*]] = tt.load %[[B_PTRS]] : tensor<32x128x!tt.ptr<f16>, #[[$B]]>
  // CHECK: %[[C:.*]] = arith.constant
  // CHECK: tt.dot %[[A_VALUE]], %[[B_VALUE]], %[[C]]
  // CHECK-SAME: tensor<32x32xf16, #[[$A]]> * tensor<32x128xf16, #[[$B]]>
  // CHECK-SAME: -> tensor<32x128xf32, #[[$MMA]]>
  tt.func public @direct_global_dot_closes_selected_plan(
      %a_base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %b_base: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %a_ptrs = tt.splat %a_base
        {tt.contiguity = dense<[1, 16]> : tensor<2xi32>}
        : !tt.ptr<f16>
          -> tensor<32x32x!tt.ptr<f16>, #gluon.auto_encoding>
    %b_ptrs = tt.splat %b_base
        {tt.contiguity = dense<[16, 1]> : tensor<2xi32>}
        : !tt.ptr<f16>
          -> tensor<32x128x!tt.ptr<f16>, #gluon.auto_encoding>
    %a = tt.load %a_ptrs :
        tensor<32x32x!tt.ptr<f16>, #gluon.auto_encoding>
    %b = tt.load %b_ptrs :
        tensor<32x128x!tt.ptr<f16>, #gluon.auto_encoding>
    %c = arith.constant dense<0.000000e+00>
        : tensor<32x128xf32, #gluon.auto_encoding>
    %d = tt.dot %a, %b, %c, inputPrecision = tf32 :
        tensor<32x32xf16, #gluon.auto_encoding> *
        tensor<32x128xf16, #gluon.auto_encoding>
        -> tensor<32x128xf32, #gluon.auto_encoding>
    tt.return
  }
}

//--- bsm-carrier.mlir

#a_bsm_shared = #ttg.swizzled_shared<{vec = 4, perPhase = 8, maxPhase = 1, order = [1, 0]}>
#b_bsm_shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#bsm_smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // BSM-DAG: #[[$MMA:.*]] = #ttg.maca_mma
  // BSM-LABEL: tt.func public @bsm_carrier
  // BSM: %[[RAW_REQ:.*]] = gluon.require_layout %{{.*}} : tensor<32x128xi32, #gluon.auto_encoding> -> tensor<32x128xi32, #ttg.dot_op<{opIdx = 1, parent = #[[$MMA]]}>>
  // BSM: %[[PERM:.*]] = "ttg.bsm_perm"(%[[RAW_REQ]]) : (tensor<32x128xi32, #ttg.dot_op<{opIdx = 1, parent = #[[$MMA]]}>>) -> tensor<32x128xf16, #gluon.auto_encoding>
  // BSM: %[[B_REQ:.*]] = gluon.require_layout %[[PERM]] : tensor<32x128xf16, #gluon.auto_encoding> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$MMA]]}>>
  // BSM: tt.dot %{{.*}}, %[[B_REQ]]
  tt.func public @bsm_carrier() {
    %a_smem = ttg.local_alloc {"ttg.gluon.default-shared-layout"} : () ->
        !ttg.memdesc<32x32xf16, #a_bsm_shared, #bsm_smem, mutable>
    %b_smem = ttg.local_alloc {"ttg.gluon.default-shared-layout"} : () ->
        !ttg.memdesc<32x128xf16, #b_bsm_shared, #bsm_smem, mutable>
    %a = ttg.local_load %a_smem {intrinsic = true} :
        !ttg.memdesc<32x32xf16, #a_bsm_shared, #bsm_smem, mutable>
        -> tensor<32x32xf16, #gluon.auto_encoding>
    %b_raw = ttg.local_load %b_smem
        {intrinsic = true, isConstantOffs = true, mmaMode = 2 : i32} :
        !ttg.memdesc<32x128xf16, #b_bsm_shared, #bsm_smem, mutable>
        -> tensor<32x128xi32, #gluon.auto_encoding>
    %b = "ttg.bsm_perm"(%b_raw) :
        (tensor<32x128xi32, #gluon.auto_encoding>)
        -> tensor<32x128xf16, #gluon.auto_encoding>
    %c = arith.constant dense<0.000000e+00> :
        tensor<32x128xf32, #gluon.auto_encoding>
    %d = tt.dot %a, %b, %c, inputPrecision = tf32 :
        tensor<32x32xf16, #gluon.auto_encoding> *
        tensor<32x128xf16, #gluon.auto_encoding>
        -> tensor<32x128xf32, #gluon.auto_encoding>
    tt.return
  }
}

//--- fixed-view-lowerable.mlir

// These are the non-canonical fixed shared views used by the fused Flash
// backward score dot. They differ from the compiler-composed default family
// but their register-to-shared LinearLayout contracts are lowerable.
#flash_a = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#flash_b = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 4, order = [0, 1]}>
#flash_smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // FIXED-LABEL: tt.func public @fixed_view_lowerable
  // FIXED: ttg.local_load
  // FIXED: ttg.local_load
  // FIXED: gluon.set_auto_layout
  // FIXED: gluon.require_layout
  // FIXED: gluon.require_layout
  // FIXED: tt.dot
  tt.func public @fixed_view_lowerable(
      %a_smem: !ttg.memdesc<32x128xf16, #flash_a, #flash_smem, mutable>,
      %b_smem: !ttg.memdesc<128x128xf16, #flash_b, #flash_smem, mutable>) {
    %a = ttg.local_load %a_smem {intrinsic = true, isConstantOffs = true} :
        !ttg.memdesc<32x128xf16, #flash_a, #flash_smem, mutable>
        -> tensor<32x128xf16, #gluon.auto_encoding>
    %b = ttg.local_load %b_smem {intrinsic = true, isConstantOffs = true} :
        !ttg.memdesc<128x128xf16, #flash_b, #flash_smem, mutable>
        -> tensor<128x128xf16, #gluon.auto_encoding>
    %c = arith.constant dense<0.000000e+00>
        : tensor<32x128xf32, #gluon.auto_encoding>
    %d = tt.dot %a, %b, %c, inputPrecision = tf32 :
        tensor<32x128xf16, #gluon.auto_encoding> *
        tensor<128x128xf16, #gluon.auto_encoding>
        -> tensor<32x128xf32, #gluon.auto_encoding>
    tt.return
  }
}
