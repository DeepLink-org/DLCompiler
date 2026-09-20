include "mlir/Pass/PassBase.td"
def InsertBarrier : Pass<"wafer-insert-barrier", "mlir::ModuleOp"> {
  let summary = "Insert Wafer SPM/DDR producer-consumer barriers";
  let constructor = "mlir::triton::createInsertBarrierPass()";
}
