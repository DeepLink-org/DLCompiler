// RUN: %triton-shared-opt-v3_4 %s --triton-to-linalg | %FileCheck %s

module {
  tt.func public @gcd_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.splat %1 : i32 -> tensor<128xi32>
    %4 = arith.addi %3, %2 : tensor<128xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<128xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32>
    %7 = tt.addptr %arg0, %c0_i32 : !tt.ptr<i32>, i32
    %8 = tt.load %7 : !tt.ptr<i32>
    %9 = tt.addptr %arg1, %c0_i32 : !tt.ptr<i32>, i32
    %10 = tt.load %9 : !tt.ptr<i32> 
    %11 = math.absi %8 : i32
    %12 = math.absi %10 : i32
    %13:2 = scf.while (%arg4 = %11, %arg5 = %12) : (i32, i32) -> (i32, i32) {
      %17 = arith.cmpi ne, %arg5, %c0_i32 : i32
      scf.condition(%17) %arg4, %arg5 : i32, i32
    } do {
    ^bb0(%arg4: i32 , %arg5: i32):
      %17 = arith.remsi %arg4, %arg5 : i32
      scf.yield %arg5, %17 : i32, i32
    }
    %14 = tt.splat %13#0 : i32 -> tensor<128xi32>
    %15 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %16 = tt.addptr %15, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    tt.store %16, %14, %6 : tensor<128x!tt.ptr<i32>>
    tt.return
  }
}


// CHECK: module {
// CHECK:   func.func @gcd_kernel(%arg0: memref<*xi32> {tt.divisibility = 16 : i32}, %arg1: memref<*xi32> {tt.divisibility = 16 : i32}, %arg2: memref<*xi32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %c128 = arith.constant 128 : index
// CHECK:     %c0_i32 = arith.constant 0 : i32
// CHECK:     %c128_i32 = arith.constant 128 : i32
// CHECK:     %0 = arith.muli %arg7, %c128_i32 : i32
// CHECK:     %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK:     %1 = affine.load %reinterpret_cast[0] : memref<1xi32, strided<[1], offset: ?>>
// CHECK:     %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK:     %2 = affine.load %reinterpret_cast_0[0] : memref<1xi32, strided<[1], offset: ?>>
// CHECK:     %3 = math.absi %1 : i32
// CHECK:     %4 = math.absi %2 : i32
// CHECK:     %5:2 = scf.while (%arg10 = %3, %arg11 = %4) : (i32, i32) -> (i32, i32) {
// CHECK:       %15 = arith.cmpi ne, %arg11, %c0_i32 : i32
// CHECK:       scf.condition(%15) %arg10, %arg11 : i32, i32
// CHECK:     } do {
// CHECK:     ^bb0(%arg10: i32, %arg11: i32):
// CHECK:       %15 = arith.remsi %arg10, %arg11 : i32
// CHECK:       scf.yield %arg11, %15 : i32, i32
// CHECK:     }
// CHECK:     %6 = tensor.empty() : tensor<128xi32>
// CHECK:     %7 = linalg.fill ins(%5#0 : i32) outs(%6 : tensor<128xi32>) -> tensor<128xi32>
// CHECK:     %8 = arith.index_cast %0 : i32 to index
// CHECK:     %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [128], strides: [1] : memref<*xi32> to memref<128xi32, strided<[1], offset: ?>>
// CHECK:     %9 = arith.index_cast %0 : i32 to index
// CHECK:     %10 = arith.addi %9, %c128 : index
// CHECK:     %11 = arith.index_cast %arg3 : i32 to index
// CHECK:     %12 = arith.minsi %10, %11 : index
// CHECK:     %13 = arith.maxsi %12, %9 : index
// CHECK:     %14 = arith.subi %13, %9 : index
// CHECK:     %extracted_slice = tensor.extract_slice %7[0] [%14] [1] : tensor<128xi32> to tensor<?xi32>
// CHECK:     %subview = memref.subview %reinterpret_cast_1[0] [%14] [1] : memref<128xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
// CHECK:     bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?xi32>, memref<?xi32, strided<[1], offset: ?>>) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }
