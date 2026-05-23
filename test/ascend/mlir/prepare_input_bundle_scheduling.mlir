// RUN: %dicp_opt %s --prepare-input-bundle-scheduling | %FileCheck %s
//
// CHECK-LABEL: func.func @_matmul_relu_fwd
// CHECK: %c524288 = arith.constant 524288 : index
// CHECK: %c262144 = arith.constant 262144 : index
// CHECK: memref.reinterpret_cast %arg2 to offset: [%arg13], sizes: [128, 256], strides: [2048, 1]
// CHECK: memref.reinterpret_cast %arg4 to offset: [%13], sizes: [128, 128], strides: [4096, 1]
// CHECK-NOT: memref.subview

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
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = tensor.empty() : tensor<128x128xf32>
    %2 = linalg.fill {dicp.tmp.stage.anchor_op_to_tile_tag.stage_2.sub_0.u_0, dicp.tmp.stage.id = 2 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_2.sub_0.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_2.sub_0.u_0", tile_sizes = array<i64: 128, 0>}} ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %3 = linalg.fill {dicp.tmp.stage.anchor_op_to_tile_tag.stage_2.sub_0.u_0, dicp.tmp.stage.id = 2 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_2.sub_0.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_2.sub_0.u_0", tile_sizes = array<i64: 128, 0>}} ins(%cst : f32) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.for %arg11 = %arg8 to %c512_i32 step %c24_i32  : i32 {
      %4 = arith.divsi %arg11, %c32_i32 {dicp.tmp.stage.id = 0 : i32} : i32
      %5 = arith.remsi %arg11, %c32_i32 {dicp.tmp.stage.id = 0 : i32} : i32
      %6 = arith.muli %4, %c256_i32 {dicp.tmp.stage.id = 0 : i32} : i32
      %7 = arith.index_cast %6 {dicp.tmp.stage.id = 0 : i32} : i32 to index
      %8 = arith.muli %5, %c128_i32 {dicp.tmp.stage.id = 0 : i32} : i32
      %9 = arith.index_cast %8 {dicp.tmp.stage.id = 0 : i32} : i32 to index
      %10 = arith.muli %7, %c2048 {dicp.tmp.stage.id = 0 : i32} : index
      %11:4 = scf.for %arg12 = %c0_i32 to %c2048_i32 step %c256_i32 iter_args(%arg13 = %10, %arg14 = %c0, %arg15 = %2, %arg16 = %3) -> (index, index, tensor<128x128xf32>, tensor<128x128xf32>)  : i32 {
        %16 = arith.addi %arg14, %9 {dicp.tmp.stage.id = 1 : i32} : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%16], sizes: [256, 128], strides: [4096, 1] {dicp.tmp.stage.id = 1 : i32} : memref<*xf32> to memref<256x128xf32, strided<[4096, 1], offset: ?>>
        %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%arg13], sizes: [256, 256], strides: [2048, 1] {dicp.tmp.stage.id = 1 : i32} : memref<*xf32> to memref<256x256xf32, strided<[2048, 1], offset: ?>>
        %alloc = memref.alloc() {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.till_unit_has_cross_user} : memref<128x256xf32>
        %alloc_3 = memref.alloc() {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.till_unit_has_cross_user} : memref<128x256xf32>
        %subview_4 = memref.subview %reinterpret_cast_2[0, 0] [128, 256] [1, 1] : memref<256x256xf32, strided<[2048, 1], offset: ?>> to memref<128x256xf32, strided<[2048, 1], offset: ?>>
        memref.copy %subview_4, %alloc_3 {dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_0, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_1.sub_0.u_0", tile_sizes = array<i64: 128, 0>}, operandSegmentSizes = array<i32: 1, 1>} : memref<128x256xf32, strided<[2048, 1], offset: ?>> to memref<128x256xf32>
        %subview_5 = memref.subview %reinterpret_cast_2[128, 0] [128, 256] [1, 1] : memref<256x256xf32, strided<[2048, 1], offset: ?>> to memref<128x256xf32, strided<[2048, 1], offset: ?>>
        memref.copy %subview_5, %alloc {dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_0, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_1.sub_0.u_0", tile_sizes = array<i64: 128, 0>}, operandSegmentSizes = array<i32: 1, 1>} : memref<128x256xf32, strided<[2048, 1], offset: ?>> to memref<128x256xf32>
        %alloc_6 = memref.alloc() {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.no_tile, dicp.tmp.stage.till_unit_has_cross_user} : memref<256x128xf32>
        %17 = bufferization.to_tensor %alloc_6 restrict writable {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.no_tile} : memref<256x128xf32> to tensor<256x128xf32>
        memref.copy %reinterpret_cast_1, %alloc_6 {dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.no_tile, operandSegmentSizes = array<i32: 1, 1>} : memref<256x128xf32, strided<[4096, 1], offset: ?>> to memref<256x128xf32>
        %18 = bufferization.to_tensor %alloc_3 restrict writable : memref<128x256xf32> to tensor<128x256xf32>
        %19 = linalg.matmul {dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_1, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_1", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_1.sub_0.u_1", tile_sizes = array<i64: 128, 0, 0>}, input_precison = "ieee"} ins(%18, %17 : tensor<128x256xf32>, tensor<256x128xf32>) outs(%arg15 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %20 = bufferization.to_tensor %alloc restrict writable : memref<128x256xf32> to tensor<128x256xf32>
        %21 = linalg.matmul {dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_1, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_1", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_1.sub_0.u_1", tile_sizes = array<i64: 128, 0, 0>}, input_precison = "ieee"} ins(%20, %17 : tensor<128x256xf32>, tensor<256x128xf32>) outs(%arg16 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %22 = arith.addi %arg13, %c256 {dicp.tmp.stage.id = 1 : i32} : index
        %23 = arith.addi %arg14, %c1048576 {dicp.tmp.stage.id = 1 : i32} : index
        scf.yield %22, %23, %19, %21 : index, index, tensor<128x128xf32>, tensor<128x128xf32>
      }
      %12 = arith.muli %7, %c4096 {dicp.tmp.stage.id = 0 : i32} : index
      %13 = arith.addi %12, %9 {dicp.tmp.stage.id = 0 : i32} : index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%13], sizes: [256, 128], strides: [4096, 1] {dicp.tmp.stage.id = 0 : i32} : memref<*xf32> to memref<256x128xf32, strided<[4096, 1], offset: ?>>
      %14 = arith.maxnumf %11#2, %2 : tensor<128x128xf32>
      %subview = memref.subview %reinterpret_cast[0, 0] [128, 128] [1, 1] : memref<256x128xf32, strided<[4096, 1], offset: ?>> to memref<128x128xf32, strided<[4096, 1], offset: ?>>
      bufferization.materialize_in_destination %14 in writable %subview {dicp.tmp.stage.anchor_op_to_tile_tag.stage_0.sub_1.u_0, dicp.tmp.stage.id = 0 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_0.sub_1.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_0.sub_1.u_0", tile_sizes = array<i64: 128, 0>}, operandSegmentSizes = array<i32: 1, 1>} : (tensor<128x128xf32>, memref<128x128xf32, strided<[4096, 1], offset: ?>>) -> ()
      %15 = arith.maxnumf %11#3, %3 : tensor<128x128xf32>
      %subview_0 = memref.subview %reinterpret_cast[128, 0] [128, 128] [1, 1] : memref<256x128xf32, strided<[4096, 1], offset: ?>> to memref<128x128xf32, strided<[4096, 1], offset: ?>>
      bufferization.materialize_in_destination %15 in writable %subview_0 {dicp.tmp.stage.anchor_op_to_tile_tag.stage_0.sub_1.u_0, dicp.tmp.stage.id = 0 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_0.sub_1.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_0.sub_1.u_0", tile_sizes = array<i64: 128, 0>}, operandSegmentSizes = array<i32: 1, 1>} : (tensor<128x128xf32>, memref<128x128xf32, strided<[4096, 1], offset: ?>>) -> ()
    }
    return
  }
}
