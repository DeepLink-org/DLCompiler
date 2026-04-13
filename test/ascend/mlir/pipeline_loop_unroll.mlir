// RUN: %dicp_opt %s --pipeline-loop-unroll > /dev/null
//
// This regression covers the critical recurrence that cannot be scheduled with
// a naive stage-major traversal:
//   stage_2(iter_k) -> stage_0(iter_k+1)
// Stage 2 also contains two substages, so flattening only substage 0 would
// miss the true loop-carried producer.

module attributes {dicp.backend = "ascend"} {
  func.func @pipeline_recurrence(%arg0: i32) -> i32 {
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %0 = scf.for %iv = %c0 to %c2 step %c1 iter_args(%iter = %arg0) -> (i32) {
      %1 = arith.addi %iter, %c1_i32 {dicp.tmp.stage.id = 0 : i32} : i32
      hivm.hir.sync_block_set {dicp.tmp.stage.id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0

      hivm.hir.sync_block_wait {dicp.tmp.stage.id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
      %2 = arith.addi %1, %c1_i32 {dicp.tmp.stage.id = 1 : i32} : i32
      hivm.hir.sync_block_set {dicp.tmp.stage.id = 1 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE2>] flag = 1

      hivm.hir.sync_block_wait {dicp.tmp.stage.id = 2 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE2>] flag = 1
      %3 = arith.addi %2, %c1_i32 {dicp.tmp.stage.id = 2 : i32} : i32
      hivm.hir.sync_block_set {dicp.tmp.stage.id = 2 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 2
      %4 = arith.addi %3, %c1_i32 {dicp.tmp.stage.id = 2 : i32} : i32

      hivm.hir.sync_block_wait {dicp.tmp.stage.id = 3 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 2
      %5 = arith.addi %4, %c1_i32 {dicp.tmp.stage.id = 3 : i32} : i32
      scf.yield %4 : i32
    } {tt.num_stages = 2 : i32}

    return %0 : i32
  }
}
