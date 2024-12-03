#ifndef NPU_TYPES
#define NPU_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "compiler/include/Dialect/NPU/IR/NPUOps.td"

class NPUTypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<NPU_Dialect, name, traits> {
    // Used by printer/parser
    let mnemonic = _mnemonic;
}

def NPU_TPipType : NPUTypeDef<"TPip", "tpip", []> {
    let summary = "tpip in NPU IR type system";

    let description = [{
        tpip for npu.
    }];

  //let hasCustomAssemblyFormat = 1;
}

#endif