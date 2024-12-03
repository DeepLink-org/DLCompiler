#ifndef NPU_DIALECT
#define NPU_DIALECT

include "mlir/IR/OpBase.td"
// include "mlir/Interfaces/FunctionInterfaces.td"
// include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the 'NPU' dialect in the ODS framework so that we
// can define our operations.
def NPU_Dialect : Dialect {
  let name = "npu";
  let cppNamespace = "::mlir::dicp::npu";
  let usePropertiesForAttributes = 1;
}

// Base class for NPU dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class NPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<NPU_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// NPU Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : NPU_Op<"add"> {
  let summary = "element-wise addition operation";
  let description =
      [{The "add" operation performs element -
        wise addition between two
            tensors.The shapes of the tensor operands are expected to match.}];

  let arguments = (ins F16Tensor : $lhs, F16Tensor : $rhs);
  let results = (outs F16Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an AddOp with from the two input operands.
  let builders = [OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>];
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

def CopyOp : NPU_Op<"copy"> {

  let description =
      [{Copies the data from the source to the destination
            .

        Source and destination are expected to have the same element type and
            shape.Otherwise,
        the result is undefined.They may have different layouts.}];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "the memref to copy from"
                           >:$source,
                       Arg<AnyRankedOrUnrankedMemRef, "the memref to copy to"
                           >:$target
                    );

  let assemblyFormat = [{
    $source `,` $target attr-dict `:` type($source) `to` type($target)
  }];

  // let builders = [
  //  Build a SubViewOp with static entries and inferred result type.
  //  OpBuilder<(ins "MemRefType":$resultType, "Value":$source,
  //    "ArrayRef<int64_t>":$offsets, "ArrayRef<int64_t>":$sizes,
  //    "ArrayRef<int64_t>":$strides)>
  //  ];

  let hasCanonicalizer = 0;
  let hasFolder = 0;
}

#endif // NPU_DIALECT