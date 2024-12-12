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
  let useDefaultTypePrinterParser = 1;
  
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}


// def NPU_TPipType : I<4>;
// def NPU_TQueType : I<4>;

// def NPU_TQueueType : DialectType<
//     NPU_Dialect,
//     CPred<"isa<mlir::dicp::npu::TQueueType>($_self)">,
//     "tqueue"> {
//   let description = [{
//     Allocates buffers for a particular device memory space.
//   }];
//   let builderCall = "$_builder.getType<mlir::dicp::npu::TQueueType>()";
// }

include "dicp/Dialect/NPU/IR/NPUTypes.td"

#endif // NPU_DIALECT