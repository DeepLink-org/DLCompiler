// RUN: %dicp_opt %s --shrink-buffers | %FileCheck %s
//
// CHECK-LABEL: func.func @_matmul_relu_fwd
// CHECK-COUNT-2: tensor.empty() : tensor<128x128xf32>
// CHECK-COUNT-2: memref.alloc() {{.*}}: memref<128x256xf32>
// CHECK: memref.reinterpret_cast %arg4 to offset: [%{{.+}}], sizes: [256, 128], strides: [4096, 1]
// CHECK-NOT: tensor.empty() : tensor<256x128xf32>
// CHECK-NOT: memref.alloc() {{.*}}: memref<256x256xf32>
func.func @_matmul_relu_fwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<*xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<*xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<*xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "mix"} {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
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
  %c2_0 = arith.constant 2 : index
  %1 = affine.apply affine_map<(d0) -> (d0 * 128)>(%c0)
  %extracted_slice = tensor.extract_slice %0[%1, 0] [128, 128] [1, 1] : tensor<256x128xf32> to tensor<128x128xf32>
  %2 = linalg.fill {dicp.tmp.stage.anchor_op_to_tile_tag.stage_2.sub_0.u_0, dicp.tmp.stage.id = 2 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_2.sub_0.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_2.sub_0.u_0", tile_sizes = array<i64: 128, 0>}} ins(%cst : f32) outs(%extracted_slice : tensor<128x128xf32>) -> tensor<128x128xf32>
  %inserted_slice = tensor.insert_slice %2 into %0[%1, 0] [128, 128] [1, 1] : tensor<128x128xf32> into tensor<256x128xf32>
  %c1_1 = arith.constant 1 : index
  %3 = arith.muli %c1, %c1_1 : index
  %4 = arith.addi %c0, %3 : index
  %5 = affine.apply affine_map<(d0) -> (d0 * 128)>(%4)
  %extracted_slice_2 = tensor.extract_slice %inserted_slice[%5, 0] [128, 128] [1, 1] : tensor<256x128xf32> to tensor<128x128xf32>
  %6 = linalg.fill {dicp.tmp.stage.anchor_op_to_tile_tag.stage_2.sub_0.u_0, dicp.tmp.stage.id = 2 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_2.sub_0.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_2.sub_0.u_0", tile_sizes = array<i64: 128, 0>}} ins(%cst : f32) outs(%extracted_slice_2 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %inserted_slice_3 = tensor.insert_slice %6 into %inserted_slice[%5, 0] [128, 128] [1, 1] : tensor<128x128xf32> into tensor<256x128xf32>
  scf.for %arg11 = %arg8 to %c512_i32 step %c24_i32  : i32 {
    %7 = arith.divsi %arg11, %c32_i32 {dicp.tmp.stage.id = 0 : i32} : i32
    %8 = arith.remsi %arg11, %c32_i32 {dicp.tmp.stage.id = 0 : i32} : i32
    %9 = arith.muli %7, %c256_i32 {dicp.tmp.stage.id = 0 : i32} : i32
    %10 = arith.index_cast %9 {dicp.tmp.stage.id = 0 : i32} : i32 to index
    %11 = arith.muli %8, %c128_i32 {dicp.tmp.stage.id = 0 : i32} : i32
    %12 = arith.index_cast %11 {dicp.tmp.stage.id = 0 : i32} : i32 to index
    %13 = arith.muli %10, %c2048 {dicp.tmp.stage.id = 0 : i32} : index
    %14:3 = scf.for %arg12 = %c0_i32 to %c2048_i32 step %c256_i32 iter_args(%arg13 = %13, %arg14 = %c0, %arg15 = %inserted_slice_3) -> (index, index, tensor<256x128xf32>)  : i32 {
      %23 = arith.addi %arg14, %12 {dicp.tmp.stage.id = 1 : i32} : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%23], sizes: [256, 128], strides: [4096, 1] {dicp.tmp.stage.id = 1 : i32} : memref<*xf32> to memref<256x128xf32, strided<[4096, 1], offset: ?>>
      %reinterpret_cast_12 = memref.reinterpret_cast %arg2 to offset: [%arg13], sizes: [256, 256], strides: [2048, 1] {dicp.tmp.stage.id = 1 : i32} : memref<*xf32> to memref<256x256xf32, strided<[2048, 1], offset: ?>>
      %alloc = memref.alloc() {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.till_unit_has_cross_user} : memref<256x256xf32>
      %24 = bufferization.to_tensor %alloc restrict writable {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32} : memref<256x256xf32> to tensor<256x256xf32>
      %c2_13 = arith.constant 2 : index
      %25 = affine.apply affine_map<(d0) -> (d0 * 128)>(%c0)
      %subview_14 = memref.subview %reinterpret_cast_12[%25, 0] [128, 256] [1, 1] : memref<256x256xf32, strided<[2048, 1], offset: ?>> to memref<128x256xf32, strided<[2048, 1], offset: ?>>
      %subview_15 = memref.subview %alloc[%25, 0] [128, 256] [1, 1] : memref<256x256xf32> to memref<128x256xf32, strided<[256, 1], offset: ?>>
      memref.copy %subview_14, %subview_15 {dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_0, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_1.sub_0.u_0", tile_sizes = array<i64: 128, 0>}, operandSegmentSizes = array<i32: 1, 1>} : memref<128x256xf32, strided<[2048, 1], offset: ?>> to memref<128x256xf32, strided<[256, 1], offset: ?>>
      %c1_16 = arith.constant 1 : index
      %26 = arith.muli %c1, %c1_16 : index
      %27 = arith.addi %c0, %26 : index
      %28 = affine.apply affine_map<(d0) -> (d0 * 128)>(%27)
      %subview_17 = memref.subview %reinterpret_cast_12[%28, 0] [128, 256] [1, 1] : memref<256x256xf32, strided<[2048, 1], offset: ?>> to memref<128x256xf32, strided<[2048, 1], offset: ?>>
      %subview_18 = memref.subview %alloc[%28, 0] [128, 256] [1, 1] : memref<256x256xf32> to memref<128x256xf32, strided<[256, 1], offset: ?>>
      memref.copy %subview_17, %subview_18 {dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_0, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_1.sub_0.u_0", tile_sizes = array<i64: 128, 0>}, operandSegmentSizes = array<i32: 1, 1>} : memref<128x256xf32, strided<[2048, 1], offset: ?>> to memref<128x256xf32, strided<[256, 1], offset: ?>>
      %alloc_19 = memref.alloc() {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.no_tile, dicp.tmp.stage.till_unit_has_cross_user} : memref<256x128xf32>
      %29 = bufferization.to_tensor %alloc_19 restrict writable {dicp.tmp.stage.alloc_producer, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.no_tile} : memref<256x128xf32> to tensor<256x128xf32>
      memref.copy %reinterpret_cast_11, %alloc_19 {dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.no_tile, operandSegmentSizes = array<i32: 1, 1>} : memref<256x128xf32, strided<[4096, 1], offset: ?>> to memref<256x128xf32>
      %c2_20 = arith.constant 2 : index
      %30 = affine.apply affine_map<(d0) -> (d0 * 128)>(%c0)
      %extracted_slice_21 = tensor.extract_slice %24[%30, 0] [128, 256] [1, 1] : tensor<256x256xf32> to tensor<128x256xf32>
      %extracted_slice_22 = tensor.extract_slice %arg15[%30, 0] [128, 128] [1, 1] : tensor<256x128xf32> to tensor<128x128xf32>
      %31 = linalg.matmul {dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_1, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_1", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_1.sub_0.u_1", tile_sizes = array<i64: 128, 0, 0>}, input_precison = "ieee"} ins(%extracted_slice_21, %29 : tensor<128x256xf32>, tensor<256x128xf32>) outs(%extracted_slice_22 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %inserted_slice_23 = tensor.insert_slice %31 into %arg15[%30, 0] [128, 128] [1, 1] : tensor<128x128xf32> into tensor<256x128xf32>
      %c1_24 = arith.constant 1 : index
      %32 = arith.muli %c1, %c1_24 : index
      %33 = arith.addi %c0, %32 : index
      %34 = affine.apply affine_map<(d0) -> (d0 * 128)>(%33)
      %extracted_slice_25 = tensor.extract_slice %24[%34, 0] [128, 256] [1, 1] : tensor<256x256xf32> to tensor<128x256xf32>
      %extracted_slice_26 = tensor.extract_slice %inserted_slice_23[%34, 0] [128, 128] [1, 1] : tensor<256x128xf32> to tensor<128x128xf32>
      %35 = linalg.matmul {dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_1, dicp.tmp.stage.id = 1 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_1.sub_0.u_1", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_1.sub_0.u_1", tile_sizes = array<i64: 128, 0, 0>}, input_precison = "ieee"} ins(%extracted_slice_25, %29 : tensor<128x256xf32>, tensor<256x128xf32>) outs(%extracted_slice_26 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %inserted_slice_27 = tensor.insert_slice %35 into %inserted_slice_23[%34, 0] [128, 128] [1, 1] : tensor<128x128xf32> into tensor<256x128xf32>
      %36 = arith.addi %arg13, %c256 {dicp.tmp.stage.id = 1 : i32} : index
      %37 = arith.addi %arg14, %c1048576 {dicp.tmp.stage.id = 1 : i32} : index
      scf.yield %36, %37, %inserted_slice_27 : index, index, tensor<256x128xf32>
    }
    %15 = arith.muli %10, %c4096 {dicp.tmp.stage.id = 0 : i32} : index
    %16 = arith.addi %15, %12 {dicp.tmp.stage.id = 0 : i32} : index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%16], sizes: [256, 128], strides: [4096, 1] {dicp.tmp.stage.id = 0 : i32} : memref<*xf32> to memref<256x128xf32, strided<[4096, 1], offset: ?>>
    %c2_4 = arith.constant 2 : index
    %17 = affine.apply affine_map<(d0) -> (d0 * 128)>(%c0)
    %extracted_slice_5 = tensor.extract_slice %14#2[%17, 0] [128, 128] [1, 1] : tensor<256x128xf32> to tensor<128x128xf32>
    %extracted_slice_6 = tensor.extract_slice %inserted_slice_3[%17, 0] [128, 128] [1, 1] : tensor<256x128xf32> to tensor<128x128xf32>
    %18 = arith.maxnumf %extracted_slice_5, %extracted_slice_6 : tensor<128x128xf32>
    %subview = memref.subview %reinterpret_cast[%17, 0] [128, 128] [1, 1] : memref<256x128xf32, strided<[4096, 1], offset: ?>> to memref<128x128xf32, strided<[4096, 1], offset: ?>>
    bufferization.materialize_in_destination %18 in writable %subview {dicp.tmp.stage.anchor_op_to_tile_tag.stage_0.sub_1.u_0, dicp.tmp.stage.id = 0 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_0.sub_1.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_0.sub_1.u_0", tile_sizes = array<i64: 128, 0>}, operandSegmentSizes = array<i32: 1, 1>} : (tensor<128x128xf32>, memref<128x128xf32, strided<[4096, 1], offset: ?>>) -> ()
    %c1_7 = arith.constant 1 : index
    %19 = arith.muli %c1, %c1_7 : index
    %20 = arith.addi %c0, %19 : index
    %21 = affine.apply affine_map<(d0) -> (d0 * 128)>(%20)
    %extracted_slice_8 = tensor.extract_slice %14#2[%21, 0] [128, 128] [1, 1] : tensor<256x128xf32> to tensor<128x128xf32>
    %extracted_slice_9 = tensor.extract_slice %inserted_slice_3[%21, 0] [128, 128] [1, 1] : tensor<256x128xf32> to tensor<128x128xf32>
    %22 = arith.maxnumf %extracted_slice_8, %extracted_slice_9 : tensor<128x128xf32>
    %subview_10 = memref.subview %reinterpret_cast[%21, 0] [128, 128] [1, 1] : memref<256x128xf32, strided<[4096, 1], offset: ?>> to memref<128x128xf32, strided<[4096, 1], offset: ?>>
    bufferization.materialize_in_destination %22 in writable %subview_10 {dicp.tmp.stage.anchor_op_to_tile_tag.stage_0.sub_1.u_0, dicp.tmp.stage.id = 0 : i32, dicp.tmp.stage.tile_meta = {dicp.tmp.stage.anchor_op_to_tile_tag = "dicp.tmp.stage.anchor_op_to_tile_tag.stage_0.sub_1.u_0", dicp.tmp.stage.producer_to_fuse_tag = "dicp.tmp.stage.producer_to_fuse_tag.stage_0.sub_1.u_0", tile_sizes = array<i64: 128, 0>}, operandSegmentSizes = array<i32: 1, 1>} : (tensor<128x128xf32>, memref<128x128xf32, strided<[4096, 1], offset: ?>>) -> ()
  }
  return
}
