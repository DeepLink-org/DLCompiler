// RUN: %dicp_opt %s --vectorize-parallel-loop | %FileCheck %s
// /opt/conda/envs/commonir/lib/python3.10/site-packages/triton/_C/dicp_opt

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
      %6 = memref.load %reinterpret_cast_0[%4] : memref<1048576xf32, strided<[1]>>
      %7 = arith.addf %5, %6 : f32
      %inserted = tensor.insert %7 into %0[%c0] : tensor<1xf32>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%4], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %inserted in writable %reinterpret_cast_1 : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
      scf.reduce 
    }
    return
  }
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: scf.parallel
// CHECK: %[[ALLOC0:.+]] = memref.alloc() : memref<1024xf32>
// CHECK: %[[SUBVIEW0:.+]] = memref.subview %{{.+}}[%{{.+}}] [1024] [1] : memref<1048576xf32, strided<[1]>> to memref<1024xf32, strided<[1], offset: ?>>
// CHECK: memref.copy %[[SUBVIEW0]], %[[ALLOC0]] : memref<1024xf32, strided<[1], offset: ?>> to memref<1024xf32>
// CHECK: %[[TENSOR0:.+]] = bufferization.to_tensor %[[ALLOC0]] restrict : memref<1024xf32> to tensor<1024xf32>
// CHECK: %[[ALLOC1:.+]] = memref.alloc() : memref<1024xf32>
// CHECK: %[[SUBVIEW1:.+]] = memref.subview %{{.+}}[%{{.+}}] [1024] [1] : memref<1048576xf32, strided<[1]>> to memref<1024xf32, strided<[1], offset: ?>>
// CHECK: memref.copy %[[SUBVIEW1]], %[[ALLOC1]] : memref<1024xf32, strided<[1], offset: ?>> to memref<1024xf32>
// CHECK: %[[TENSOR1:.+]] = bufferization.to_tensor %[[ALLOC1]] restrict : memref<1024xf32> to tensor<1024xf32>
// CHECK: %[[RESULT:.+]] = arith.addf %[[TENSOR0]], %[[TENSOR1]] : tensor<1024xf32>
// CHECK: %[[ALLOC_RESULT:.+]] = memref.alloc() : memref<1024xf32>
// CHECK: bufferization.materialize_in_destination %[[RESULT]] in writable %[[ALLOC_RESULT]] : (tensor<1024xf32>, memref<1024xf32>) -> ()
// CHECK: %[[SUBVIEW_OUT:.+]] = memref.subview %{{.+}}[%{{.+}}] [1024] [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
// CHECK: memref.copy %[[ALLOC_RESULT]], %[[SUBVIEW_OUT]] : memref<1024xf32> to memref<1024xf32, strided<[1], offset: ?>>
// CHECK: return
