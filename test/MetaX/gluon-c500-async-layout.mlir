// RUN: split-file %s %t
// RUN: triton-opt --split-input-file \
// RUN:   --tritonmetaxgpu-gluon-legalize-c500-async-copy-layout \
// RUN:   %t/mask.mlir | FileCheck %t/mask.mlir
// RUN: not triton-opt \
// RUN:   --tritonmetaxgpu-gluon-legalize-c500-async-copy-layout \
// RUN:   %t/mask-invalid.mlir 2>&1 | FileCheck %t/mask-invalid.mlir
// RUN: not triton-opt \
// RUN:   --tritonmetaxgpu-gluon-legalize-c500-async-copy-layout \
// RUN:   %t/invalid.mlir 2>&1 | FileCheck %t/invalid.mlir

// These tests start from a selected candidate with concrete tensor and shared
// layouts. Candidate discovery is covered by gluon-layout-candidate-bundle.

//--- mask.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], elementsMNK = [1, 1, 8], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func public @uniform_mask_does_not_swizzle_i1
  // CHECK: %[[PTR:.*]] = ttg.convert_layout {{.*}} : tensor<32x128x!tt.ptr<f16>, #mma> -> tensor<32x128x!tt.ptr<f16>, #blocked>
  // CHECK: %[[MASK:.*]] = ttg.convert_layout {{.*}} : tensor<32x128xi1, #mma> -> tensor<32x128xi1, #blocked>
  // CHECK-NOT: ttg.swizzle_tensor
  // CHECK: ttg.async_copy_global_to_local %[[PTR]], {{.*}} mask %[[MASK]]
  tt.func public @uniform_mask_does_not_swizzle_i1(
      %ptr: tensor<32x128x!tt.ptr<f16>, #mma> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>},
      %predicate: i1) {
    %mask = tt.splat %predicate : i1 -> tensor<32x128xi1, #mma>
    %zero = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #mma>
    %buffer = ttg.local_alloc : () -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    %token = ttg.async_copy_global_to_local %ptr, %buffer mask %mask other %zero {intrinsic = true} : tensor<32x128x!tt.ptr<f16>, #mma> -> <32x128xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#column = #ttg.slice<{dim = 0, parent = #blocked}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func public @contiguous_cmp_rebuilt_from_source
  // CHECK: %[[SWIZZLED:.*]] = "ttg.swizzle_tensor"(%{{.*}}) <{InVec = 8 : i32, OutVec = 8 : i32, maxPhase = 8 : i32, perPhase = 1 : i32}>
  // CHECK-SAME: tensor<32x128xi32, #blocked>
  // CHECK: %[[MASK:.*]] = arith.cmpi slt, %[[SWIZZLED]], %{{.*}} : tensor<32x128xi32, #blocked>
  // CHECK-NOT: "ttg.swizzle_tensor"({{.*}}tensor<32x128xi1
  // CHECK: ttg.async_copy_global_to_local {{.*}} mask %[[MASK]]
  tt.func public @contiguous_cmp_rebuilt_from_source(
      %ptr: tensor<32x128x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>}) {
    %c120 = arith.constant 120 : i32
    %columns = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #column>
    %columns_2d = tt.expand_dims %columns {axis = 0 : i32} : tensor<128xi32, #column> -> tensor<1x128xi32, #blocked>
    %coordinates = tt.broadcast %columns_2d : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %boundary = tt.splat %c120 : i32 -> tensor<32x128xi32, #blocked>
    %mask = arith.cmpi slt, %coordinates, %boundary : tensor<32x128xi32, #blocked>
    %zero = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked>
    %buffer = ttg.local_alloc : () -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    %token = ttg.async_copy_global_to_local %ptr, %buffer mask %mask other %zero {intrinsic = true} : tensor<32x128x!tt.ptr<f16>, #blocked> -> <32x128xf16, #shared, #smem, mutable>
    tt.return
  }
}

//--- mask-invalid.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: error: C500 async_copy has a non-uniform mask for a swizzled shared destination that cannot be rebuilt as a pure elementwise i1 expression with source-side swizzle_tensor
  tt.func public @reject_indirect_dynamic_mask(
      %ptr: tensor<32x128x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>},
      %mask_ptr: tensor<32x128x!tt.ptr<i1>, #blocked>) {
    %mask = tt.load %mask_ptr : tensor<32x128x!tt.ptr<i1>, #blocked>
    %zero = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked>
    %buffer = ttg.local_alloc : () -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    %token = ttg.async_copy_global_to_local %ptr, %buffer mask %mask other %zero {intrinsic = true} : tensor<32x128x!tt.ptr<f16>, #blocked> -> <32x128xf16, #shared, #smem, mutable>
    tt.return
  }
}

//--- invalid.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#converted = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: error: C500 async_copy legalization requires proven global address contiguity; register ownership encoding is insufficient
  tt.func public @concrete_ownership_is_not_address_contiguity(
      %ptr: tensor<32x32x!tt.ptr<f16>, #blocked>,
      %mask: tensor<32x32xi1, #converted>) {
    %converted_ptr = ttg.convert_layout %ptr : tensor<32x32x!tt.ptr<f16>, #blocked> -> tensor<32x32x!tt.ptr<f16>, #converted>
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #converted>
    %buffer = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %token = ttg.async_copy_global_to_local %converted_ptr, %buffer mask %mask other %zero {intrinsic = true} : tensor<32x32x!tt.ptr<f16>, #converted> -> <32x32xf16, #shared, #smem, mutable>
    tt.return
  }
}
