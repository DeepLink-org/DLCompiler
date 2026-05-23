// RUN: %dicp_opt %s --de-linalgize | %FileCheck %s
//
// CHECK-LABEL: func.func @_matmul_relu_fwd
// CHECK: memref.copy %{{.+}}, %{{.+}} {dicp.tmp.stage.anchor_op_to_tile_tag.stage_{{[0-9]+}}.sub_0.u_0
// CHECK: arith.maxnumf %{{.+}}, %{{.+}} : tensor<128x128xf32>
// CHECK: bufferization.materialize_in_destination %{{.+}} in writable %{{.+}}
// CHECK-NOT: linalg.copy
// CHECK-NOT: linalg.generic

#map = affine_map<(d0) -> (d0 * 128)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dicp.backend = "ascend"} {
  func.func @_matmul_relu_fwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<*xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<*xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<*xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "mix"} {
    %c32_i32 = arith.constant {dicp.tmp.stage.id = 2 : i32} 32 : i32
    %c256_i32 = arith.constant {dicp.tmp.stage.id = 2 : i32} 256 : i32
    %c128_i32 = arith.constant {dicp.tmp.stage.id = 2 : i32} 128 : i32
    %cst = arith.constant {dicp.tmp.stage.id = 2 : i32} 0.000000e+00 : f32
    %c512_i32 = arith.constant {dicp.tmp.stage.id = 2 : i32} 512 : i32
    %c24_i32 = arith.constant {dicp.tmp.stage.id = 2 : i32} 24 : i32
    %c2048_i32 = arith.constant {dicp.tmp.stage.id = 2 : i32} 2048 : i32
    %c0_i32 = arith.constant {dicp.tmp.stage.id = 2 : i32} 0 : i32
    %c1048576 = arith.constant {dicp.tmp.stage.id = 2 : i32} 1048576 : index
    %c256 = arith.constant {dicp.tmp.stage.id = 2 : i32} 256 : index
    %c4096 = arith.constant {dicp.tmp.stage.id = 2 : i32} 4096 : index
    %c0 = arith.constant {dicp.tmp.stage.id = 2 : i32} 0 : index
    %c2048 = arith.constant {dicp.tmp.stage.id = 2 : i32} 2048 : index
    %0 = tensor.empty() {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 2 : i32} : tensor<256x128xf32>
    %1 = scf.forall (%arg11) in (2) shared_outs(%arg12 = %0) -> (tensor<256x128xf32>) {
      %2 = affine.apply #map(%arg11)
      %extracted_slice = tensor.extract_slice %arg12[%2, 0] [128, 128] [1, 1] : tensor<256x128xf32> to tensor<128x128xf32>
      %3 = linalg.fill {dicp.tmp.stage.anchor_op_to_tile_tag.stage_2.sub_0.u_0, dicp.tmp.stage.id = 2 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_2.sub_0.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_2.sub_0.u_0", tile_sizes = array<i64: 128, 0>}} ins(%cst : f32) outs(%extracted_slice : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %3 into %arg12[%2, 0] [128, 128] [1, 1] : tensor<128x128xf32> into tensor<256x128xf32>
      }
    } {dicp.tmp.stage.id = 2 : i32}
    scf.for %arg11 = %arg8 to %c512_i32 step %c24_i32  : i32 {
      %2 = arith.divsi %arg11, %c32_i32 {dicp.tmp.stage.id = 0 : i32} : i32
      %3 = arith.remsi %arg11, %c32_i32 {dicp.tmp.stage.id = 0 : i32} : i32
      %4 = arith.muli %2, %c256_i32 {dicp.tmp.stage.id = 0 : i32} : i32
      %5 = arith.index_cast %4 {dicp.tmp.stage.id = 0 : i32} : i32 to index
      %6 = arith.muli %3, %c128_i32 {dicp.tmp.stage.id = 0 : i32} : i32
      %7 = arith.index_cast %6 {dicp.tmp.stage.id = 0 : i32} : i32 to index
      %8 = arith.muli %5, %c2048 {dicp.tmp.stage.id = 0 : i32} : index
      %9:3 = scf.for %arg12 = %c0_i32 to %c2048_i32 step %c256_i32 iter_args(%arg13 = %8, %arg14 = %c0, %arg15 = %1) -> (index, index, tensor<256x128xf32>)  : i32 {
        %13 = arith.addi %arg14, %7 {dicp.tmp.stage.id = 1 : i32} : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%13], sizes: [256, 128], strides: [4096, 1] {dicp.tmp.stage.id = 1 : i32} : memref<*xf32> to memref<256x128xf32, strided<[4096, 1], offset: ?>>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%arg13], sizes: [256, 256], strides: [2048, 1] {dicp.tmp.stage.id = 1 : i32} : memref<*xf32> to memref<256x256xf32, strided<[2048, 1], offset: ?>>
        %alloc = memref.alloc() {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.till_unit_has_cross_user} : memref<256x256xf32>
        %14 = bufferization.to_tensor %alloc restrict writable {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32} : memref<256x256xf32> to tensor<256x256xf32>
        scf.forall (%arg16) in (2) {
          %19 = affine.apply #map(%arg16)
          %subview = memref.subview %reinterpret_cast_1[%19, 0] [128, 256] [1, 1] : memref<256x256xf32, strided<[2048, 1], offset: ?>> to memref<128x256xf32, strided<[2048, 1], offset: ?>>
          %subview_3 = memref.subview %alloc[%19, 0] [128, 256] [1, 1] : memref<256x256xf32> to memref<128x256xf32, strided<[256, 1], offset: ?>>
          linalg.copy {dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_0, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.original_op_name = "memref.copy", dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_1.sub_0.u_0", tile_sizes = array<i64: 128, 0>}} ins(%subview : memref<128x256xf32, strided<[2048, 1], offset: ?>>) outs(%subview_3 : memref<128x256xf32, strided<[256, 1], offset: ?>>)
        } {dicp.tmp.stage.id = 1 : i32}
        %alloc_2 = memref.alloc() {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.no_tile, dicp.tmp.stage.till_unit_has_cross_user} : memref<256x128xf32>
        %15 = bufferization.to_tensor %alloc_2 restrict writable {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.no_tile} : memref<256x128xf32> to tensor<256x128xf32>
        linalg.copy {dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.no_tile, dicp.tmp.stage.original_op_name = "memref.copy"} ins(%reinterpret_cast_0 : memref<256x128xf32, strided<[4096, 1], offset: ?>>) outs(%alloc_2 : memref<256x128xf32>)
        %16 = scf.forall (%arg16) in (2) shared_outs(%arg17 = %arg15) -> (tensor<256x128xf32>) {
          %19 = affine.apply #map(%arg16)
          %extracted_slice = tensor.extract_slice %14[%19, 0] [128, 256] [1, 1] : tensor<256x256xf32> to tensor<128x256xf32>
          %extracted_slice_3 = tensor.extract_slice %15[0, 0] [256, 128] [1, 1] : tensor<256x128xf32> to tensor<256x128xf32>
          %extracted_slice_4 = tensor.extract_slice %arg17[%19, 0] [128, 128] [1, 1] : tensor<256x128xf32> to tensor<128x128xf32>
          %20 = linalg.matmul {dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_1, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_1", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_1.sub_0.u_1", tile_sizes = array<i64: 128, 0, 0>}, input_precison = "ieee"} ins(%extracted_slice, %extracted_slice_3 : tensor<128x256xf32>, tensor<256x128xf32>) outs(%extracted_slice_4 : tensor<128x128xf32>) -> tensor<128x128xf32>
          scf.forall.in_parallel {
            tensor.parallel_insert_slice %20 into %arg17[%19, 0] [128, 128] [1, 1] : tensor<128x128xf32> into tensor<256x128xf32>
          }
        } {dicp.tmp.stage.id = 1 : i32}
        %17 = arith.addi %arg13, %c256 {dicp.tmp.stage.id = 1 : i32} : index
        %18 = arith.addi %arg14, %c1048576 {dicp.tmp.stage.id = 1 : i32} : index
        scf.yield %17, %18, %16 : index, index, tensor<256x128xf32>
      }
      %10 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%9#2, %1 : tensor<256x128xf32>, tensor<256x128xf32>) outs(%9#2 : tensor<256x128xf32>) attrs =  {dicp.tmp.stage.id = 0 : i32, dicp.tmp.stage.op_had_fused, dicp.tmp.stage.producer_to_fuse_tag.stage_0.sub_1.u_0} {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %13 = arith.maxnumf %in, %in_0 : f32
        linalg.yield %13 : f32
      } -> tensor<256x128xf32>
      %11 = arith.muli %5, %c4096 {dicp.tmp.stage.id = 0 : i32} : index
      %12 = arith.addi %11, %7 {dicp.tmp.stage.id = 0 : i32} : index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%12], sizes: [256, 128], strides: [4096, 1] {dicp.tmp.stage.id = 0 : i32} : memref<*xf32> to memref<256x128xf32, strided<[4096, 1], offset: ?>>
      scf.forall (%arg12) in (2) {
        %13 = affine.apply #map(%arg12)
        %extracted_slice = tensor.extract_slice %9#2[%13, 0] [128, 128] [1, 1] : tensor<256x128xf32> to tensor<128x128xf32>
        %extracted_slice_0 = tensor.extract_slice %1[%13, 0] [128, 128] [1, 1] : tensor<256x128xf32> to tensor<128x128xf32>
        %14 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_0 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%extracted_slice : tensor<128x128xf32>) attrs =  {dicp.tmp.stage.id = 0 : i32, dicp.tmp.stage.producer_to_fuse_tag.stage_0.sub_1.u_0} {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %16 = arith.maxnumf %in, %in_1 : f32
          linalg.yield %16 : f32
        } -> tensor<128x128xf32>
        %15 = bufferization.to_buffer %14 : tensor<128x128xf32> to memref<128x128xf32, strided<[128, 1], offset: ?>>
        %subview = memref.subview %reinterpret_cast[%13, 0] [128, 128] [1, 1] : memref<256x128xf32, strided<[4096, 1], offset: ?>> to memref<128x128xf32, strided<[4096, 1], offset: ?>>
        linalg.copy {dicp.tmp.stage.anchor_op_to_tile_tag.stage_0.sub_1.u_0, dicp.tmp.stage.id = 0 : i32, dicp.tmp.stage.original_op_name = "bufferization.materialize_in_destination", dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_0.sub_1.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_0.sub_1.u_0", tile_sizes = array<i64: 128, 0>}} ins(%15 : memref<128x128xf32, strided<[128, 1], offset: ?>>) outs(%subview : memref<128x128xf32, strided<[4096, 1], offset: ?>>)
      } {dicp.tmp.stage.id = 0 : i32}
    }
    return
  }
}
