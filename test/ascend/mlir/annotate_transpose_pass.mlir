// Test for AnnotateTransposePass - checks that the pass adds MayImplicitTransposeWithLastAxis annotations appropriately

// RUN: bishengir-opt %s -annotate-transpose-pass | FileCheck %s

func.func @test_linalg_copy_with_permuted_memref() {
  // Original memref with permuted layout
  %0 = memref.alloc() : memref<128x5xf32, strided<[8, 1]>>
  %1 = memref.alloc() : memref<128x5xf32, strided<[1, 128], offset: ?>>
  // linalg.copy should get annotated since target has permuted memref
  linalg.copy %1, %0 : memref<128x5xf32, strided<[1, 128], offset: ?>> to memref<128x5xf32, strided<[8, 1]>>
  // CHECK: linalg.copy
  // CHECK: annotation.mark
  // CHECK: "MayImplicitTransposeWithLastAxis"
  return
}

func.func @test_memref_copy_with_permuted_memref() {
  // Original memref with permuted layout
  %0 = memref.alloc() : memref<128x8xf32, strided<[8, 1]>>
  %1 = memref.alloc() : memref<128x8xf32, strided<[1, 128], offset: ?>>
  // memref.copy should get annotated since target has permuted memref
  memref.copy %1, %0 : memref<128x8xf32, strided<[1, 128], offset: ?>> to memref<128x8xf32, strided<[8, 1]>>
  // CHECK: memref.copy
  // CHECK: annotation.mark
  // CHECK: "MayImplicitTransposeWithLastAxis"
  return
}

func.func @test_bufferization_to_tensor_with_permuted_source() {
  %0 = memref.alloc() : memref<128x8xf32, strided<[8, 1]>>
  // bufferization.to_tensor should get annotated since source has permuted memref
  %1 = bufferization.to_tensor %0 : memref<128x8xf32, strided<[8, 1]>>
  // CHECK: bufferization.to_tensor
  // CHECK: annotation.mark
  // CHECK: "MayImplicitTransposeWithLastAxis"
  return
}

func.func @test_memref_subview_with_permuted_source() {
  %0 = memref.alloc() : memref<128x8xf32, strided<[8, 1]>>
  // memref.subview should get annotated since source has permuted memref
  %1 = memref.subview %0[0, 0] to [64, 4] : memref<128x8xf32, strided<[8, 1]>>
  // CHECK: memref.subview
  // CHECK: annotation.mark
  // CHECK: "MayImplicitTransposeWithLastAxis"
  return
}

func.func @test_non_permuted_memref_no_annotation() {
  // Non-permuted memref should not get annotated
  %0 = memref.alloc() : memref<128x5xf32>
  %1 = memref.alloc() : memref<128x5xf32>
  linalg.copy %1, %0 : memref<128x5xf32> to memref<128x5xf32>
  // CHECK-NOT: annotation.mark
  // CHECK-NOT: "MayImplicitTransposeWithLastAxis"
  return
}