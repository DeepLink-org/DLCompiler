#ifndef DIALECT_LINALGEXT_IR_LINALG_EXT_OPS
#define DIALECT_LINALGEXT_IR_LINALG_EXT_OPS

include "LinalgExtOpBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/TilingInterface.td"

class VectorTwoOperands<string mnemonic, list<Trait> traits = []>
    : LinalgExt_Op<mnemonic, traits> {
  let arguments = (ins
    Arg<AnyMemRef, "Input operand0", [MemRead]>:$lhs,
    Arg<AnyMemRef, "Input operand1", [MemRead]>:$rhs
  );
  let results = (outs AnyMemRef : $outs);
  let assemblyFormat =
      [{operands attr - dict `:` functional - type(operands, results)}];
}

class VectorOneOperands<string mnemonic, list<Trait> traits = []>
    : LinalgExt_Op<mnemonic, traits> {
  let arguments = (ins Arg<AnyMemRef, "Input operand0", [MemRead]> : $lhs);
  let results = (outs AnyMemRef : $outs);
}

def LinalgExt_ReduceSumOp : VectorTwoOperands<"reduce_sum", [LinalgExt_ReduceTrait]> {
  let summary = "Reduce sum operator";
  let description = [{
      Reduce a tensor along the given axis by computing the sum of the axis.}];

  let arguments = (ins 
    AnyRankedTensorOrMemRefType:$input,
    AnyRankedTensorOrMemRefType:$output,
    DefaultValuedAttr<I64Attr, "0">:$axis
  );

  let results = (outs Variadic<AnyRankedTensorOrMemRefType> : $result);
  let assemblyFormat = [{
    attr-dict
    `reduce_axis` `(` $axis `)`
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    (`->` type($result)^)?
  }];
}

#endif // DIALECT_LINALGEXT_IR_LINALG_EXT_OPS