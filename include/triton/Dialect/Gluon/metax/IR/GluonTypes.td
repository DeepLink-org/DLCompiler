#ifndef GLUON_METAX_TYPES
#define GLUON_METAX_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "triton/Dialect/Gluon/IR/GluonDialect.td"

def Gluon_StorageKind_SMEM : I32EnumAttrCase<"smem", 0, "smem">;

def Gluon_StorageKindAttr : I32EnumAttr<
    "StorageKind", "Gluon storage kind for explicitly aliased buffers",
    [Gluon_StorageKind_SMEM]> {
  let cppNamespace = "::mlir::triton::gluon";
}

def Gluon_ReuseGroupKind_Shared : I32EnumAttrCase<"shared", 0, "shared">;
def Gluon_ReuseGroupKind_Distinct : I32EnumAttrCase<"distinct", 1, "distinct">;

def Gluon_ReuseGroupKindAttr : I32EnumAttr<
    "ReuseGroupKind", "Gluon Shared-buffer overlap relationship",
    [Gluon_ReuseGroupKind_Shared, Gluon_ReuseGroupKind_Distinct]> {
  let cppNamespace = "::mlir::triton::gluon";
}

class GluonTypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<Gluon_Dialect, name, traits> {
  let mnemonic = _mnemonic;
}

def Gluon_StorageAliasSpecType
    : GluonTypeDef<"StorageAliasSpec", "storage_alias_spec"> {
  let summary = "An explicit Shared-storage alias specification";
  let description = [{
    Groups multiple logical Shared allocations onto one physical backing.
    The optional size is validated by gluon-storage-alias-lowering. Without an
    overlap definition the pass uses the largest referencing allocation; with
    one it computes the recursive reuse-group footprint. Stage one supports
    only C500 Shared memory.
  }];

  let parameters = (ins
    EnumParameter<Gluon_StorageKindAttr>:$storage,
    OptionalParameter<"std::optional<int64_t>">:$bufferSizeBytes
  );
  let assemblyFormat = "`<` $storage (`,` $bufferSizeBytes^)? `>`";
  let genVerifyDecl = 1;
}

def Gluon_ReuseGroupType : GluonTypeDef<"ReuseGroup", "reuse_group"> {
  let summary = "A tree node describing Shared-buffer overlap";
  let description = [{
    Represents either a shared group, whose children occupy the same logical
    region, or a distinct group, whose children occupy non-overlapping
    regions. Leaves are gluon.storage_alias_local_alloc results.
  }];
  let parameters = (ins
    EnumParameter<Gluon_ReuseGroupKindAttr>:$groupKind
  );
  let assemblyFormat = "`<` $groupKind `>`";
  let genVerifyDecl = 1;
}

#endif // GLUON_METAX_TYPES
