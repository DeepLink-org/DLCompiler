module {
  tt.func public @flip_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<8> : tensor<64xi32>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %2 = arith.muli %1, %cst : tensor<64xi32>
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %4 = tt.expand_dims %2 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %5 = tt.broadcast %3 : tensor<1x8xi32> -> tensor<64x8xi32>
    %6 = tt.broadcast %4 : tensor<64x1xi32> -> tensor<64x8xi32>
    %7 = arith.addi %5, %6 : tensor<64x8xi32>
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x8x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<64x8x!tt.ptr<f32>>, tensor<64x8xi32>
    %10 = tt.load %9 : tensor<64x8x!tt.ptr<f32>>
    %11 = tt.bitcast %10 : tensor<64x8xf32> -> tensor<64x8xi32>
    %12 = tt.reshape %11 : tensor<64x8xi32> -> tensor<64x2x2x2xi32>
    %13 = "tt.reduce"(%12) <{axis = 1 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %29 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %29 : i32
    }) : (tensor<64x2x2x2xi32>) -> tensor<64x2x2xi32>
    %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<64x2x2xi32> -> tensor<64x1x2x2xi32>
    %15 = tt.broadcast %14 : tensor<64x1x2x2xi32> -> tensor<64x2x2x2xi32>
    %16 = arith.xori %12, %15 : tensor<64x2x2x2xi32>
    %17 = "tt.reduce"(%16) <{axis = 2 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %29 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %29 : i32
    }) : (tensor<64x2x2x2xi32>) -> tensor<64x2x2xi32>
    %18 = tt.expand_dims %17 {axis = 2 : i32} : tensor<64x2x2xi32> -> tensor<64x2x1x2xi32>
    %19 = tt.broadcast %18 : tensor<64x2x1x2xi32> -> tensor<64x2x2x2xi32>
    %20 = arith.xori %16, %19 : tensor<64x2x2x2xi32>
    %21 = "tt.reduce"(%20) <{axis = 3 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %29 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %29 : i32
    }) : (tensor<64x2x2x2xi32>) -> tensor<64x2x2xi32>
    %22 = tt.expand_dims %21 {axis = 3 : i32} : tensor<64x2x2xi32> -> tensor<64x2x2x1xi32>
    %23 = tt.broadcast %22 : tensor<64x2x2x1xi32> -> tensor<64x2x2x2xi32>
    %24 = arith.xori %20, %23 : tensor<64x2x2x2xi32>
    %25 = tt.reshape %24 : tensor<64x2x2x2xi32> -> tensor<64x8xi32>
    %26 = tt.bitcast %25 : tensor<64x8xi32> -> tensor<64x8xf32>
    %27 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x8x!tt.ptr<f32>>
    %28 = tt.addptr %27, %7 : tensor<64x8x!tt.ptr<f32>>, tensor<64x8xi32>
    tt.store %28, %26 : tensor<64x8x!tt.ptr<f32>>
    tt.return
  }
}
