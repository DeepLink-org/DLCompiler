#ifndef GLUON_DIALECT
#define GLUON_DIALECT

include "mlir/IR/OpBase.td"

def Gluon_Dialect : Dialect {
  let name = "gluon";
  let cppNamespace = "::mlir::triton::gluon";
  let description = [{
    Gluon dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "triton::gpu::TritonGPUDialect",
    "mlir::gpu::GPUDialect",
  ];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let usePropertiesForAttributes = 1;
  let extraClassDeclaration = [{
    // MetaX extends the generic Gluon dialect with storage-alias types.  The
    // implementation stays under Gluon/metax; this is only its registration
    // hook on the owning dialect.
    void registerMetaXTypes();
  }];
}

#endif
