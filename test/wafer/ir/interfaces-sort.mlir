module {
  tt.func public @sort_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
    %11 = tt.reshape %10 : tensor<64x8xf32> -> tensor<2x2x2x2x2x2x2x2x2xf32>
    %12 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %13 = tt.reshape %12 : tensor<2xi32> -> tensor<1x1x1x1x1x1x1x2x1xi32>
    %14 = tt.bitcast %11 : tensor<2x2x2x2x2x2x2x2x2xf32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %15 = "tt.reduce"(%14) <{axis = 8 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %94 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %94 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2x2x2xi32>
    %16 = tt.expand_dims %15 {axis = 8 : i32} : tensor<2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x1xi32>
    %17 = tt.broadcast %16 : tensor<2x2x2x2x2x2x2x2x1xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %18 = arith.xori %14, %17 : tensor<2x2x2x2x2x2x2x2x2xi32>
    %19 = tt.bitcast %18 : tensor<2x2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2xf32>
    %20 = tt.reshape %12 : tensor<2xi32> -> tensor<1x1x1x1x1x1x1x1x2xi32>
    %21 = arith.cmpf ogt, %11, %19 : tensor<2x2x2x2x2x2x2x2x2xf32>
    %22 = tt.broadcast %13 : tensor<1x1x1x1x1x1x1x2x1xi32> -> tensor<1x1x1x1x1x1x1x2x2xi32>
    %23 = tt.broadcast %20 : tensor<1x1x1x1x1x1x1x1x2xi32> -> tensor<1x1x1x1x1x1x1x2x2xi32>
    %24 = arith.xori %22, %23 : tensor<1x1x1x1x1x1x1x2x2xi32>
    %25 = arith.extui %21 : tensor<2x2x2x2x2x2x2x2x2xi1> to tensor<2x2x2x2x2x2x2x2x2xi32>
    %26 = tt.broadcast %24 : tensor<1x1x1x1x1x1x1x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %27 = arith.cmpi ne, %25, %26 : tensor<2x2x2x2x2x2x2x2x2xi32>
    %28 = arith.select %27, %19, %11 : tensor<2x2x2x2x2x2x2x2x2xi1>, tensor<2x2x2x2x2x2x2x2x2xf32>
    %29 = tt.reshape %12 : tensor<2xi32> -> tensor<1x1x1x1x1x1x2x1x1xi32>
    %30 = tt.bitcast %28 : tensor<2x2x2x2x2x2x2x2x2xf32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %31 = "tt.reduce"(%30) <{axis = 7 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %94 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %94 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2x2x2xi32>
    %32 = tt.expand_dims %31 {axis = 7 : i32} : tensor<2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x1x2xi32>
    %33 = tt.broadcast %32 : tensor<2x2x2x2x2x2x2x1x2xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %34 = arith.xori %30, %33 : tensor<2x2x2x2x2x2x2x2x2xi32>
    %35 = tt.bitcast %34 : tensor<2x2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2xf32>
    %36 = arith.cmpf ogt, %28, %35 : tensor<2x2x2x2x2x2x2x2x2xf32>
    %37 = tt.broadcast %29 : tensor<1x1x1x1x1x1x2x1x1xi32> -> tensor<1x1x1x1x1x1x2x2x1xi32>
    %38 = tt.broadcast %13 : tensor<1x1x1x1x1x1x1x2x1xi32> -> tensor<1x1x1x1x1x1x2x2x1xi32>
    %39 = arith.xori %37, %38 : tensor<1x1x1x1x1x1x2x2x1xi32>
    %40 = arith.extui %36 : tensor<2x2x2x2x2x2x2x2x2xi1> to tensor<2x2x2x2x2x2x2x2x2xi32>
    %41 = tt.broadcast %39 : tensor<1x1x1x1x1x1x2x2x1xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %42 = arith.cmpi ne, %40, %41 : tensor<2x2x2x2x2x2x2x2x2xi32>
    %43 = arith.select %42, %35, %28 : tensor<2x2x2x2x2x2x2x2x2xi1>, tensor<2x2x2x2x2x2x2x2x2xf32>
    %44 = tt.bitcast %43 : tensor<2x2x2x2x2x2x2x2x2xf32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %45 = "tt.reduce"(%44) <{axis = 8 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %94 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %94 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2x2x2xi32>
    %46 = tt.expand_dims %45 {axis = 8 : i32} : tensor<2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x1xi32>
    %47 = tt.broadcast %46 : tensor<2x2x2x2x2x2x2x2x1xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %48 = arith.xori %44, %47 : tensor<2x2x2x2x2x2x2x2x2xi32>
    %49 = tt.bitcast %48 : tensor<2x2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2xf32>
    %50 = arith.cmpf ogt, %43, %49 : tensor<2x2x2x2x2x2x2x2x2xf32>
    %51 = tt.broadcast %29 : tensor<1x1x1x1x1x1x2x1x1xi32> -> tensor<1x1x1x1x1x1x2x1x2xi32>
    %52 = tt.broadcast %20 : tensor<1x1x1x1x1x1x1x1x2xi32> -> tensor<1x1x1x1x1x1x2x1x2xi32>
    %53 = arith.xori %51, %52 : tensor<1x1x1x1x1x1x2x1x2xi32>
    %54 = arith.extui %50 : tensor<2x2x2x2x2x2x2x2x2xi1> to tensor<2x2x2x2x2x2x2x2x2xi32>
    %55 = tt.broadcast %53 : tensor<1x1x1x1x1x1x2x1x2xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %56 = arith.cmpi ne, %54, %55 : tensor<2x2x2x2x2x2x2x2x2xi32>
    %57 = arith.select %56, %49, %43 : tensor<2x2x2x2x2x2x2x2x2xi1>, tensor<2x2x2x2x2x2x2x2x2xf32>
    %58 = tt.bitcast %57 : tensor<2x2x2x2x2x2x2x2x2xf32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %59 = "tt.reduce"(%58) <{axis = 6 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %94 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %94 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2x2x2xi32>
    %60 = tt.expand_dims %59 {axis = 6 : i32} : tensor<2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x1x2x2xi32>
    %61 = tt.broadcast %60 : tensor<2x2x2x2x2x2x1x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %62 = arith.xori %58, %61 : tensor<2x2x2x2x2x2x2x2x2xi32>
    %63 = tt.bitcast %62 : tensor<2x2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2xf32>
    %64 = arith.cmpf ogt, %57, %63 : tensor<2x2x2x2x2x2x2x2x2xf32>
    %65 = arith.extui %64 : tensor<2x2x2x2x2x2x2x2x2xi1> to tensor<2x2x2x2x2x2x2x2x2xi32>
    %66 = tt.broadcast %29 : tensor<1x1x1x1x1x1x2x1x1xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %67 = arith.cmpi ne, %65, %66 : tensor<2x2x2x2x2x2x2x2x2xi32>
    %68 = arith.select %67, %63, %57 : tensor<2x2x2x2x2x2x2x2x2xi1>, tensor<2x2x2x2x2x2x2x2x2xf32>
    %69 = tt.bitcast %68 : tensor<2x2x2x2x2x2x2x2x2xf32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %70 = "tt.reduce"(%69) <{axis = 7 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %94 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %94 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2x2x2xi32>
    %71 = tt.expand_dims %70 {axis = 7 : i32} : tensor<2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x1x2xi32>
    %72 = tt.broadcast %71 : tensor<2x2x2x2x2x2x2x1x2xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %73 = arith.xori %69, %72 : tensor<2x2x2x2x2x2x2x2x2xi32>
    %74 = tt.bitcast %73 : tensor<2x2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2xf32>
    %75 = arith.cmpf ogt, %68, %74 : tensor<2x2x2x2x2x2x2x2x2xf32>
    %76 = arith.extui %75 : tensor<2x2x2x2x2x2x2x2x2xi1> to tensor<2x2x2x2x2x2x2x2x2xi32>
    %77 = tt.broadcast %13 : tensor<1x1x1x1x1x1x1x2x1xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %78 = arith.cmpi ne, %76, %77 : tensor<2x2x2x2x2x2x2x2x2xi32>
    %79 = arith.select %78, %74, %68 : tensor<2x2x2x2x2x2x2x2x2xi1>, tensor<2x2x2x2x2x2x2x2x2xf32>
    %80 = tt.bitcast %79 : tensor<2x2x2x2x2x2x2x2x2xf32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %81 = "tt.reduce"(%80) <{axis = 8 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %94 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %94 : i32
    }) : (tensor<2x2x2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2x2x2xi32>
    %82 = tt.expand_dims %81 {axis = 8 : i32} : tensor<2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x1xi32>
    %83 = tt.broadcast %82 : tensor<2x2x2x2x2x2x2x2x1xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %84 = arith.xori %80, %83 : tensor<2x2x2x2x2x2x2x2x2xi32>
    %85 = tt.bitcast %84 : tensor<2x2x2x2x2x2x2x2x2xi32> -> tensor<2x2x2x2x2x2x2x2x2xf32>
    %86 = arith.cmpf ogt, %79, %85 : tensor<2x2x2x2x2x2x2x2x2xf32>
    %87 = arith.extui %86 : tensor<2x2x2x2x2x2x2x2x2xi1> to tensor<2x2x2x2x2x2x2x2x2xi32>
    %88 = tt.broadcast %20 : tensor<1x1x1x1x1x1x1x1x2xi32> -> tensor<2x2x2x2x2x2x2x2x2xi32>
    %89 = arith.cmpi ne, %87, %88 : tensor<2x2x2x2x2x2x2x2x2xi32>
    %90 = arith.select %89, %85, %79 : tensor<2x2x2x2x2x2x2x2x2xi1>, tensor<2x2x2x2x2x2x2x2x2xf32>
    %91 = tt.reshape %90 : tensor<2x2x2x2x2x2x2x2x2xf32> -> tensor<64x8xf32>
    %92 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x8x!tt.ptr<f32>>
    %93 = tt.addptr %92, %7 : tensor<64x8x!tt.ptr<f32>>, tensor<64x8xi32>
    tt.store %93, %91 : tensor<64x8x!tt.ptr<f32>>
    tt.return
  }
}
