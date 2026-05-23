// RUN: %dicp_opt %s --schedule-input-bundles | %FileCheck %s
//
// CHECK-LABEL: func.func @_matmul_relu_fwd
// CHECK: %[[BVIEW:.+]] = memref.reinterpret_cast %arg3 to offset: [%{{.+}}], sizes: [256, 128], strides: [4096, 1]
// CHECK: %[[A1OFF:.+]] = arith.addi %arg13, %c262144 : index
// CHECK: %[[ALLOC_B:.+]] = memref.alloc() {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.no_tile, dicp.tmp.stage.till_unit_has_cross_user} : memref<256x128xf32>
// CHECK: %[[TENSOR_B:.+]] = bufferization.to_tensor %[[ALLOC_B]] restrict writable {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.no_tile} : memref<256x128xf32> to tensor<256x128xf32>
// CHECK: memref.copy %[[BVIEW]], %[[ALLOC_B]] {dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.no_tile
// CHECK: %[[ALLOC_A0:.+]] = memref.alloc() {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.till_unit_has_cross_user} : memref<128x256xf32>
// CHECK: %[[AVIEW0:.+]] = memref.reinterpret_cast %arg2 to offset: [%arg13], sizes: [128, 256], strides: [2048, 1]
// CHECK: memref.copy %[[AVIEW0]], %[[ALLOC_A0]] {dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_0
// CHECK: %[[TENSOR_A0:.+]] = bufferization.to_tensor %[[ALLOC_A0]] restrict writable : memref<128x256xf32> to tensor<128x256xf32>
// CHECK: %[[MATMUL0:.+]] = linalg.matmul
// CHECK-SAME: ins(%[[TENSOR_A0]], %[[TENSOR_B]] : tensor<128x256xf32>, tensor<256x128xf32>)
// CHECK: %[[ALLOC_A1:.+]] = memref.alloc() {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.till_unit_has_cross_user} : memref<128x256xf32>
// CHECK: %[[AVIEW1:.+]] = memref.reinterpret_cast %arg2 to offset: [%[[A1OFF]]], sizes: [128, 256], strides: [2048, 1]
// CHECK: memref.copy %[[AVIEW1]], %[[ALLOC_A1]] {dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_0
// CHECK: %[[TENSOR_A1:.+]] = bufferization.to_tensor %[[ALLOC_A1]] restrict writable : memref<128x256xf32> to tensor<128x256xf32>
// CHECK: %[[MATMUL1:.+]] = linalg.matmul
// CHECK-SAME: ins(%[[TENSOR_A1]], %[[TENSOR_B]] : tensor<128x256xf32>, tensor<256x128xf32>)
// CHECK: %[[OUT1OFF:.+]] = arith.addi %{{.+}}, %c524288 : index
// CHECK: %{{.+}} = memref.reinterpret_cast %arg4 to offset: [%[[OUT1OFF]]], sizes: [128, 128], strides: [4096, 1]

module attributes {dicp.backend = "ascend"} {
  func.func @_matmul_relu_fwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<*xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<*xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<*xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "mix"} {
    %c524288 = arith.constant 524288 : index
    %c262144 = arith.constant 262144 : index
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
        %17 = arith.addi %arg14, %9 {dicp.tmp.stage.id = 1 : i32} : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%17], sizes: [256, 128], strides: [4096, 1] {dicp.tmp.stage.id = 1 : i32} : memref<*xf32> to memref<256x128xf32, strided<[4096, 1], offset: ?>>
        %alloc = memref.alloc() {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.till_unit_has_cross_user} : memref<128x256xf32>
        %alloc_2 = memref.alloc() {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.till_unit_has_cross_user} : memref<128x256xf32>
        %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%arg13], sizes: [128, 256], strides: [2048, 1] : memref<*xf32> to memref<128x256xf32, strided<[2048, 1], offset: ?>>
        memref.copy %reinterpret_cast_3, %alloc_2 {dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_0, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_1.sub_0.u_0", tile_sizes = array<i64: 128, 0>}, operandSegmentSizes = array<i32: 1, 1>} : memref<128x256xf32, strided<[2048, 1], offset: ?>> to memref<128x256xf32>
        %18 = arith.addi %arg13, %c262144 : index
        %reinterpret_cast_4 = memref.reinterpret_cast %arg2 to offset: [%18], sizes: [128, 256], strides: [2048, 1] : memref<*xf32> to memref<128x256xf32, strided<[2048, 1], offset: ?>>
        memref.copy %reinterpret_cast_4, %alloc {dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_0, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_1.sub_0.u_0", tile_sizes = array<i64: 128, 0>}, operandSegmentSizes = array<i32: 1, 1>} : memref<128x256xf32, strided<[2048, 1], offset: ?>> to memref<128x256xf32>
        %alloc_5 = memref.alloc() {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.no_tile, dicp.tmp.stage.till_unit_has_cross_user} : memref<256x128xf32>
        %19 = bufferization.to_tensor %alloc_5 restrict writable {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.no_tile} : memref<256x128xf32> to tensor<256x128xf32>
        memref.copy %reinterpret_cast_1, %alloc_5 {dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.no_tile, operandSegmentSizes = array<i32: 1, 1>} : memref<256x128xf32, strided<[4096, 1], offset: ?>> to memref<256x128xf32>
        %20 = bufferization.to_tensor %alloc_2 restrict writable : memref<128x256xf32> to tensor<128x256xf32>
        %21 = linalg.matmul {dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_1, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_1", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_1.sub_0.u_1", tile_sizes = array<i64: 128, 0, 0>}, input_precison = "ieee"} ins(%20, %19 : tensor<128x256xf32>, tensor<256x128xf32>) outs(%arg15 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %22 = bufferization.to_tensor %alloc restrict writable : memref<128x256xf32> to tensor<128x256xf32>
        %23 = linalg.matmul {dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_1, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_1", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_1.sub_0.u_1", tile_sizes = array<i64: 128, 0, 0>}, input_precison = "ieee"} ins(%22, %19 : tensor<128x256xf32>, tensor<256x128xf32>) outs(%arg16 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %24 = arith.addi %arg13, %c256 {dicp.tmp.stage.id = 1 : i32} : index
        %25 = arith.addi %arg14, %c1048576 {dicp.tmp.stage.id = 1 : i32} : index
        scf.yield %24, %25, %21, %23 : index, index, tensor<128x128xf32>, tensor<128x128xf32>
      }
      %12 = arith.muli %7, %c4096 {dicp.tmp.stage.id = 0 : i32} : index
      %13 = arith.addi %12, %9 {dicp.tmp.stage.id = 0 : i32} : index
      %14 = arith.maxnumf %11#2, %2 : tensor<128x128xf32>
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%13], sizes: [128, 128], strides: [4096, 1] : memref<*xf32> to memref<128x128xf32, strided<[4096, 1], offset: ?>>
      bufferization.materialize_in_destination %14 in writable %reinterpret_cast {dicp.tmp.stage.anchor_op_to_tile_tag.stage_0.sub_1.u_0, dicp.tmp.stage.id = 0 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_0.sub_1.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_0.sub_1.u_0", tile_sizes = array<i64: 128, 0>}, operandSegmentSizes = array<i32: 1, 1>} : (tensor<128x128xf32>, memref<128x128xf32, strided<[4096, 1], offset: ?>>) -> ()
      %15 = arith.maxnumf %11#3, %3 : tensor<128x128xf32>
      %16 = arith.addi %13, %c524288 : index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%16], sizes: [128, 128], strides: [4096, 1] : memref<*xf32> to memref<128x128xf32, strided<[4096, 1], offset: ?>>
      bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 {dicp.tmp.stage.anchor_op_to_tile_tag.stage_0.sub_1.u_0, dicp.tmp.stage.id = 0 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_0.sub_1.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_0.sub_1.u_0", tile_sizes = array<i64: 128, 0>}, operandSegmentSizes = array<i32: 1, 1>} : (tensor<128x128xf32>, memref<128x128xf32, strided<[4096, 1], offset: ?>>) -> ()
    }
    return
  }
}
