#ifndef LinalgEXT_PASSES
#define LinalgEXT_PASSES

include "mlir/Pass/PassBase.td"

def LinalgIfToSelect : Pass<"linalg-if-to-select", "mlir::ModuleOp"> {
  let summary = "Convert scf.if inside parallel linalg.generic to arith.select and hoist selects.";
  let description = [{
    This pass converts conditional logic (`scf.if`) inside a parallel `linalg.generic`
    operation into data-flow operations (`arith.select`).

    The conversion process involves two main stages:
    1.  **LinalgIfToSelectPattern**: Converts `scf.if` statements (where the else
        block yields a constant or is empty) into an equivalent sequence of
        `arith.select` operations. For arguments used inside the `then` block,
        it introduces an `arith.select` using a zero constant for the false case.
        It also attempts to fix zero-value assignment for `memref.store` operations
        that result from the conversion (FixLinalgSelectZeroForStorePattern).
    2.  **LinalgLiftSelectPattern**: Lifts scalar `arith.select` operations that use
        loop-invariant operands or block arguments outside the `linalg.generic`
        operation, converting them into tensor-based `arith.select`s which are
        then added as new inputs to the `linalg.generic`. This process helps to
        simplify the body of the `linalg.generic` and enables further fusion.
  }];
  let constructor = "mlir::dicp::LinalgExt::createLinalgIfToSelectPass()";
  let dependentDialects = ["mlir::linalg::LinalgDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::tensor::TensorDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::memref::MemRefDialect"]; 
}


def LinalgGenericToSCF : Pass<"linalg-generic-to-scf", "mlir::ModuleOp"> {
  let summary = "Lower linalg.generic ops to SCF loops";
  let description = [{
    Converts linalg.generic operations into explicit scf.for loops,
    generating structured loop nests that preserve the original
    indexing maps and iterator types.
  }];
  let constructor = "mlir::dicp::LinalgExt::createLinalgGenericToSCFPass()";
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}


def ScalarTo1DTensor : Pass<"scalar-to-1d-tensor", "mlir::func::FuncOp"> {
  let summary = "Convert scalar computations and memref load/store to tensor<1 x T> form";
  let description = [{
    The ScalarTo1DTensor pass targets scalar computations and
    memory access within a function and rewrites them into an explicit
    tensor<1 x T>-based form. This enables uniform handling of scalar
    values in subsequent bufferization and lowering passes.
  }];
  
  let constructor = "mlir::dicp::LinalgExt::createScalarTo1DTensorPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::func::FuncDialect"
  ];
}

#endif
