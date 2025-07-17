// RUN: %triton_shared_opt %s --triton-to-linalg | %FileCheck %s

module {
  tt.func public @fn_broadcast(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.shape_1 = 0 : i32} , %arg2: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> 
    %1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> 
    %2 = tt.addptr %1, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %3 = tt.load %2 : tensor<1024x!tt.ptr<f32>>
    %4 = tt.reshape %3 : tensor<1024xf32> -> tensor<128x1x8xf32>
    %5 = tt.broadcast %4 : tensor<128x1x8xf32> -> tensor<128x4x8xf32>
    %6 = tt.reshape %5 : tensor<128x4x8xf32> -> tensor<4096xf32>
    %7 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>
    tt.store %9, %6 : tensor<4096x!tt.ptr<f32>>
    tt.return
  } 
}



// CHECK: module {
// CHECK:   func.func @fn_broadcast(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32, tt.shape_1 = 0 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
// CHECK:     %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1]>>
// CHECK:     %alloc = memref.alloc() : memref<1024xf32>
// CHECK:     memref.copy %reinterpret_cast, %alloc : memref<1024xf32, strided<[1]>> to memref<1024xf32>
// CHECK:     %0 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32>
// CHECK:     %1 = tensor.empty() : tensor<128x4x8xf32>
// CHECK:     %expanded = tensor.expand_shape %0 
// CHECK-SAME{literal}: [[0,1]] output_shape [128, 8] : tensor<1024xf32> into tensor<128x8xf32>
// CHECK:     %broadcasted = linalg.broadcast ins(%expanded : tensor<128x8xf32>) outs(%1 : tensor<128x4x8xf32>) dimensions = [1] 
// CHECK:     %collapsed = tensor.collapse_shape %broadcasted 
// CHECK-SAME{literal}: [[0, 1, 2]] : tensor<128x4x8xf32> into tensor<4096xf32>
// CHECK:     %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096], strides: [1] : memref<*xf32> to memref<4096xf32, strided<[1]>>
// CHECK:     bufferization.materialize_in_destination %collapsed in writable %reinterpret_cast_0 : (tensor<4096xf32>, memref<4096xf32, strided<[1]>>) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }

