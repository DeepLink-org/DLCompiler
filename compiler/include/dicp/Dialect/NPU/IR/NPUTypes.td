#ifndef NPU_DIALECT_TYPES
#define NPU_DIALECT_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "dicp/Dialect/NPU/IR/NPUDialect.td"

class NPUTypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<NPU_Dialect, name, traits> {
    // Used by printer/parser
    let mnemonic = _mnemonic;
}


def NPU_TPipType : NPUTypeDef<"TPip", "tpip", []> {
  let summary = "queue type (`::mlir::dicp::npu::TPipType`) in NPU IR type system";
  let description = [{tpip}];
  let hasCustomAssemblyFormat = 1;
}

def NPU_GlobalTensorType : NPUTypeDef<"GlobalTensor", "globaltensor", []> {
  let summary = "GlobalTensor type (`::mlir::dicp::npu::GlobalTensorType`) in NPU IR type system";
  let description = [{GlobalTensor}];
  let parameters = (ins
    "uint32_t":$type
  );
  let builders = [
      TypeBuilder<(ins "unsigned":$type), [{
        return $_get($_ctxt, type);
      }]>
    ];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

def NPU_TQueueType : NPUTypeDef<"TQueue", "tqueue", []> {
  let summary = "queue type (`::mlir::dicp::npu::TQueueType`) in NPU IR type system";
  let description = [{queue}];
  let parameters = (ins
    "uint32_t":$position,
    "uint32_t":$bufferNumber
  );
  let builders = [
      TypeBuilder<(ins "unsigned":$position,  "unsigned":$bufferNumber), [{
        return $_get($_ctxt, position, bufferNumber);
      }]>
    ];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

#endif
