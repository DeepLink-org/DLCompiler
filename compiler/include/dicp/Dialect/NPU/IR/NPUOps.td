#ifndef NPU_DIALECT_OPS
#define NPU_DIALECT_OPS

include "dicp/Dialect/NPU/IR/NPUDialect.td"
include "dicp/Dialect/NPU/IR/NPUTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td" // SymbolUserOpInterface
include "mlir/IR/OpAsmInterface.td" // OpAsmOpInterface
include "mlir/Interfaces/CallInterfaces.td" // CallOpInterface
include "mlir/Interfaces/CastInterfaces.td" // CastOpInterface
include "mlir/Interfaces/FunctionInterfaces.td" // FunctionOpInterface
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/ControlFlowInterfaces.td" // BranchOpInterface
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/CastInterfaces.td" // CastOpInterface
include "mlir/Interfaces/CallInterfaces.td" // CallOpInterface
include "mlir/IR/BuiltinAttributeInterfaces.td"

// Provide a definition of the 'NPU' dialect in the ODS framework so that we
// can define our operations.
// def NPU_Dialect : Dialect {
//   let name = "npu";
//   let cppNamespace = "::mlir::dicp::npu";
//   let usePropertiesForAttributes = 1;
// }

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
// CreateTPipOp
//===----------------------------------------------------------------------===//
def CreateTPipOp : NPU_Op<"create_tpip"> {
  let summary = "create npu tpip";
  let description =
      [{Create npu tpip.}];
  let results = (outs NPU_TPipType:$result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
  let hasCanonicalizer = 0;
  let hasFolder = 0;
}
def CreateTQueueOp : NPU_Op<"create_tqueue"> {
  let summary = "create npu tqueue";
  let description =
      [{Create npu tqueue.}];
  let results = (outs NPU_TQueueType : $result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
  let hasCanonicalizer = 0;
  let hasFolder = 0;
}

def CreateGlobalTensorOp : NPU_Op<"create_global_tensor"> {
  let summary = "create npu global tensor";
  let description =
      [{Create npu global tensor.}];
  let results = (outs NPU_GlobalTensorType : $result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
  let hasCanonicalizer = 0;
  let hasFolder = 0;
}

def SetGlobalBufferOp : NPU_Op<"set_global_buffer"> {
  let summary = "set npu global buffer";
  let description =
      [{set npu global buffer.}];
  let arguments = (ins Arg<NPU_GlobalTensorType, "global tensor"
                           >:$globalTensor
                  );
  let results = (outs NPU_GlobalTensorType : $result);
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $globalTensor `:` type($globalTensor)`)`
    `:` type($result)
  }];
  let hasCanonicalizer = 0;
  let hasFolder = 0;
}

def InitlBufferOp : NPU_Op<"init_buffer"> {
  let summary = "init npu buffer";
  let description =
      [{init npu buffer.}];
  let arguments = (ins Arg<NPU_TPipType, "npu tpip"
                           >:$tpip,
                       Arg<NPU_TQueueType, "npu queue"
                           >:$queue
                  );
  let results = (outs NPU_TQueueType : $result);
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $tpip `:` type($tpip) `,` $queue `:` type($queue)`)`
    `:` type($result)
  }];
  let hasCanonicalizer = 0;
  let hasFolder = 0;
}

def AllocLocalOp : NPU_Op<"alloc_local"> {
  let summary = "alloc npu local tensor";
  let description =
      [{alloc npu local tensor.}];
  let arguments = (ins Arg<NPU_TQueueType, "npu queue"
                           >:$queue
                  );
  let results = (outs AnyRankedOrUnrankedMemRef : $result);
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $queue `:` type($queue)`)`
    `:` type($result)
  }];
  let hasCanonicalizer = 0;
  let hasFolder = 0;
}

def FreeLocalOp : NPU_Op<"free_local"> {
  let summary = "free npu local tensor";
  let description =
      [{free npu local tensor.}];
  let arguments = (ins Arg<NPU_TQueueType, "npu queue"
                           >:$queue,
                       AnyRankedOrUnrankedMemRef : $local   
                  );
  let results = (outs NPU_TQueueType : $result);
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $queue `:` type($queue) `,` $local `:` type($local) `)`
    `:` type($result)
  }];
  let hasCanonicalizer = 0;
  let hasFolder = 0;
}

def EnQueOp : NPU_Op<"en_queue"> {
  let summary = "npu local tensor en queue";
  let description =
      [{npu local tensor en queue.}];
  let arguments = (ins Arg<NPU_TQueueType, "npu queue"
                           >:$queue,
                       AnyRankedOrUnrankedMemRef : $local
                  );
  let results = (outs NPU_TQueueType : $result);
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $queue `:` type($queue) `,` $local `:` type($local)`)`
    `:` type($result)
  }];
  let hasCanonicalizer = 0;
  let hasFolder = 0;
}

def DeQueOp : NPU_Op<"de_queue"> {
  let summary = "npu local tensor de queue";
  let description =
      [{npu local tensor de queue.}];
  let arguments = (ins Arg<NPU_TQueueType, "npu queue"
                           >:$queue
                  );
  let results = (outs AnyRankedOrUnrankedMemRef : $result);
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $queue `:` type($queue)`)`
    `:` type($result)
  }];
  let hasCanonicalizer = 0;
  let hasFolder = 0;
}
//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

def AddFOp : NPU_Op<"addf"> {
  let summary = "element-wise addition operation";
  let description =
      [{The "add" operation performs element -
        wise addition between two
            tensors.The shapes of the tensor operands are expected to match.}];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "the memref to copy from"
                           > : $lhs, 
                       Arg<AnyRankedOrUnrankedMemRef, "the memref to copy from"
                           > : $rhs,
                       AnyRankedOrUnrankedMemRef: $out);
  // let results = (outs AnyRankedOrUnrankedMemRef);

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;

  // Allow building an AddOp with from the two input operands.
  // let builders = [OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>];
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `)`
    `outs` `(` $out `:` type($out) `)`
  }];
  let hasCanonicalizer = 0;
  let hasFolder = 0;
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

def CopyOp : NPU_Op<"copy"> {

  let description =
      [{Copies the data from the source to the destination.
        Source and destination are expected to have the same element type and
            shape.Otherwise,
        the result is undefined.They may have different layouts.}];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "the memref to copy from"
                           >:$source,
                       Arg<AnyRankedOrUnrankedMemRef, "the memref to copy to"
                           >:$target
                    );

  let assemblyFormat = [{
    attr-dict
    `ins` `(` $source `:` type($source)`)`
    `outs` `(` $target `:` type($target) `)`
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

#endif // NPU_DIALECT_OPS