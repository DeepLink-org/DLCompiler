// RUN: %dicp_opt %s --vectorize-parallel-loop | %FileCheck %s

module {
  func.func @main(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %c1024_i32 = arith.constant 1024 : i32
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = tensor.empty() : tensor<1xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1048576], strides: [1] : memref<?xf32> to memref<1048576xf32, strided<[1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1048576], strides: [1] : memref<?xf32> to memref<1048576xf32, strided<[1]>>
    %1 = arith.muli %arg8, %c1024_i32 : i32
    scf.parallel (%arg11) = (%c0) to (%c1024) step (%c1) {
      %2 = arith.index_cast %arg11 : index to i32
      %3 = arith.addi %1, %2 : i32
      %4 = arith.index_cast %3 : i32 to index
      %5 = memref.load %reinterpret_cast[%4] : memref<1048576xf32, strided<[1]>>
      %6 = arith.mulf %5, %5 : f32
      %7 = memref.load %reinterpret_cast_0[%4] : memref<1048576xf32, strided<[1]>>
      %8 = arith.addf %6, %7 : f32
      %inserted = tensor.insert %8 into %0[%c0] : tensor<1xf32>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%4], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %inserted in writable %reinterpret_cast_1 : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
      scf.reduce 
    }
    return
  }
}

// CHECK-NOT: scf.parallel
// CHECK: %alloc = memref.alloc() : memref<1024xf32>
// CHECK: %subview = memref.subview %reinterpret_cast[%1] [1024] [1] : memref<1048576xf32, strided<[1]>> to memref<1024xf32, strided<[1], offset: ?>>
// CHECK: memref.copy %subview, %alloc : memref<1024xf32, strided<[1], offset: ?>> to memref<1024xf32>
// CHECK: %[[TENSOR_VAL:.+]] = bufferization.to_tensor %alloc restrict : memref<1024xf32> to tensor<1024xf32>
// CHECK: %[[SQUARE_RESULT:.+]] = arith.mulf %[[TENSOR_VAL]], %[[TENSOR_VAL]] : tensor<1024xf32>
// CHECK: %alloc_1 = memref.alloc() : memref<1024xf32>
// CHECK: %subview_2 = memref.subview %reinterpret_cast_0[%1] [1024] [1] : memref<1048576xf32, strided<[1]>> to memref<1024xf32, strided<[1], offset: ?>>
// CHECK: memref.copy %subview_2, %alloc_1 : memref<1024xf32, strided<[1], offset: ?>> to memref<1024xf32>
// CHECK: %[[TENSOR_VAL2:.+]] = bufferization.to_tensor %alloc_1 restrict : memref<1024xf32> to tensor<1024xf32>
// CHECK: %[[ADD_RESULT:.+]] = arith.addf %[[SQUARE_RESULT]], %[[TENSOR_VAL2]] : tensor<1024xf32>