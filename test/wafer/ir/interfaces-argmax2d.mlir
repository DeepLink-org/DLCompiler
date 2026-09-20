module {
  tt.func public @argmax_kernel_2d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<16x1xi32>
    %c16_i32 = arith.constant 16 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c16_i32 : i32
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %3 = tt.splat %1 : i32 -> tensor<16xi32>
    %4 = arith.addi %3, %2 : tensor<16xi32>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %6 = arith.muli %5, %cst : tensor<16x1xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %8 = tt.addptr %7, %6 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %9 = tt.expand_dims %2 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %10 = tt.broadcast %8 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %11 = tt.broadcast %9 : tensor<1x16xi32> -> tensor<16x16xi32>
    %12 = tt.addptr %10, %11 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %13 = tt.load %12 : tensor<16x16x!tt.ptr<f32>>
    %14:2 = "tt.reduce"(%13, %11) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: i32, %arg4: f32, %arg5: i32):
      %17 = arith.cmpf oeq, %arg2, %arg4 : f32
      %18 = arith.cmpi slt, %arg3, %arg5 : i32
      %19 = arith.andi %17, %18 : i1
      %20 = arith.cmpf ogt, %arg2, %arg4 : f32
      %21 = arith.ori %20, %19 : i1
      %22 = arith.select %21, %arg2, %arg4 : f32
      %23 = arith.select %21, %arg3, %arg5 : i32
      tt.reduce.return %22, %23 : f32, i32
    }) : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16xf32>, tensor<16xi32>)
    %15 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %16 = tt.addptr %15, %4 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    tt.store %16, %14#1 : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}
