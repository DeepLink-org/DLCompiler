// RUN: %dicp_opt %s --canonicalize-tileable-ops | %FileCheck %s
//
// CHECK-LABEL: func.func @_matmul_relu_fwd
// CHECK: linalg.copy {dicp.tmp.stage.anchor_op_to_tile_tag.stage_{{[0-9]+}}.sub_0.u_0
// CHECK: dicp.tmp.stage.original_op_name = "memref.copy"
// CHECK: linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
// CHECK: bufferization.to_buffer %{{.+}} {dicp.tmp.stage.id = {{[0-9]+}} : i32, dicp.tmp.stage.producer_to_fuse_tag.stage_{{[0-9]+}}.sub_1.u_0}
// CHECK: linalg.copy {dicp.tmp.stage.anchor_op_to_tile_tag.stage_{{[0-9]+}}.sub_1.u_0
// CHECK-NOT: bufferization.materialize_in_destination %

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
    %1 = linalg.fill {dicp.tmp.stage.anchor_op_to_tile_tag.stage_2.sub_0.u_0, dicp.tmp.stage.id = 2 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_2.sub_0.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_2.sub_0.u_0", tile_sizes = array<i64: 128, 0>}} ins(%cst : f32) outs(%0 : tensor<256x128xf32>) -> tensor<256x128xf32>
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
        memref.copy %reinterpret_cast_1, %alloc {dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_0, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_1.sub_0.u_0", tile_sizes = array<i64: 128, 0>}} : memref<256x256xf32, strided<[2048, 1], offset: ?>> to memref<256x256xf32>
        %14 = bufferization.to_tensor %alloc restrict writable {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32} : memref<256x256xf32> to tensor<256x256xf32>
        %alloc_2 = memref.alloc() {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.no_tile, dicp.tmp.stage.till_unit_has_cross_user} : memref<256x128xf32>
        memref.copy %reinterpret_cast_0, %alloc_2 {dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.no_tile} : memref<256x128xf32, strided<[4096, 1], offset: ?>> to memref<256x128xf32>
        %15 = bufferization.to_tensor %alloc_2 restrict writable {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.no_tile} : memref<256x128xf32> to tensor<256x128xf32>
        %16 = linalg.matmul {dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_1, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_1", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_1.sub_0.u_1", tile_sizes = array<i64: 128, 0, 0>}, input_precison = "ieee"} ins(%14, %15 : tensor<256x256xf32>, tensor<256x128xf32>) outs(%arg15 : tensor<256x128xf32>) -> tensor<256x128xf32>
        %17 = arith.addi %arg13, %c256 {dicp.tmp.stage.id = 1 : i32} : index
        %18 = arith.addi %arg14, %c1048576 {dicp.tmp.stage.id = 1 : i32} : index
        scf.yield %17, %18, %16 : index, index, tensor<256x128xf32>
      }
      %10 = arith.maxnumf %9#2, %1 {dicp.tmp.stage.id = 0 : i32, dicp.tmp.stage.producer_to_fuse_tag.stage_0.sub_1.u_0} : tensor<256x128xf32>
      %11 = arith.muli %5, %c4096 {dicp.tmp.stage.id = 0 : i32} : index
      %12 = arith.addi %11, %7 {dicp.tmp.stage.id = 0 : i32} : index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%12], sizes: [256, 128], strides: [4096, 1] {dicp.tmp.stage.id = 0 : i32} : memref<*xf32> to memref<256x128xf32, strided<[4096, 1], offset: ?>>
      bufferization.materialize_in_destination %10 in writable %reinterpret_cast {dicp.tmp.stage.anchor_op_to_tile_tag.stage_0.sub_1.u_0, dicp.tmp.stage.id = 0 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_0.sub_1.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_0.sub_1.u_0", tile_sizes = array<i64: 128, 0>}} : (tensor<256x128xf32>, memref<256x128xf32, strided<[4096, 1], offset: ?>>) -> ()
    }
    return
  }
}
