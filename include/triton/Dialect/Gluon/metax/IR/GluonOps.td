#ifndef GLUON_METAX_OPS
#define GLUON_METAX_OPS

include "triton/Dialect/Gluon/metax/IR/GluonTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"

def Gluon_SameOperandAndResultMemorySpace
    : NativeOpTrait<"SameOperandAndResultMemorySpace">;

def Gluon_StorageAliasSpecOp
    : Gluon_Op<"storage_alias_spec", [Pure]> {
  let summary = "Define an explicit Shared-storage alias class";
  let description = [{
    Defines one physical Shared backing used by multiple logical allocations.
    The backing size is either supplied explicitly or computed from the
    referencing allocations and their optional reuse-group overlap tree.
  }];

  let arguments = (ins
    Gluon_StorageKindAttr:$storage,
    OptionalAttr<I64Attr>:$buffer_size_bytes,
    OptionalAttr<DenseI64ArrayAttr>:$buffer_shape
  );
  let results = (outs Gluon_StorageAliasSpecType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `storage` `=` $storage
    (`,` `size` `=` $buffer_size_bytes^)?
    attr-dict `:` qualified(type($result))
  }];
}

def Gluon_StorageAliasLocalAllocOp
    : Gluon_Op<"storage_alias_local_alloc", [Pure]> {
  let summary = "Allocate a logical Shared view from a storage alias class";
  let description = [{
    Frontend operation created by local_alloc(reuse=storage_alias_spec). The
    storage-alias lowering replaces it with gluon.local_alias over one standard
    ttg.local_alloc backing.
  }];

  let arguments = (ins Gluon_StorageAliasSpecType:$storage_alias);
  let results = (outs TTG_MemDescType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $storage_alias attr-dict `:` qualified(type($storage_alias))
    `->` qualified(type($result))
  }];
}

def Gluon_ReuseGroupElement
    : AnyTypeOf<[TTG_MemDescType, Gluon_ReuseGroupType],
                "Shared memdesc or nested Gluon reuse group">;

def Gluon_ReuseGroupOp : Gluon_Op<"reuse_group", [Pure]> {
  let summary = "Define Shared-buffer overlap relationships";
  let description = [{
    Builds a tree whose leaves are storage_alias_local_alloc results. A shared
    node overlays its children, while a distinct node places its children in
    non-overlapping regions. group_size groups consecutive buffer indices for
    subtiling.
  }];
  let arguments = (ins
    Variadic<Gluon_ReuseGroupElement>:$elements,
    Gluon_ReuseGroupKindAttr:$group_kind,
    DefaultValuedAttr<I64Attr, "1">:$group_size
  );
  let results = (outs Gluon_ReuseGroupType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $elements `)` `group_kind` `=` $group_kind
    (`,` `group_size` `=` $group_size^)? attr-dict `:`
    `(` qualified(type($elements)) `)` `->` qualified(type($result))
  }];
}

def Gluon_SetBufferOverlapOp : Gluon_Op<"set_buffer_overlap"> {
  let summary = "Attach an overlap tree to a storage alias specification";
  let arguments = (ins
    Gluon_StorageAliasSpecType:$storage_alias_spec,
    Gluon_ReuseGroupType:$overlap_def
  );
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $storage_alias_spec `,` $overlap_def `)` `:`
    `(` qualified(type($storage_alias_spec)) `,`
        qualified(type($overlap_def)) `)` `->` `(` `)` attr-dict
  }];
}

def Gluon_RequireLayoutOp : Gluon_Op<"require_layout",
                                 [SameOperandsAndResultShape,
                                  SameOperandsAndResultElementType, Pure]> {
  let summary = "explicit internal tensor or memdesc layout requirement";
  let description = [{
    Internal layout contract used by backend-specific Gluon layout pipelines.
    The result must carry a concrete distributed encoding for a register tensor
    or a concrete Shared encoding for a memdesc. Layout propagation may retag
    the source producer/carrier to satisfy the contract. Tensor cleanup may
    materialize an explicit convert_layout when the source is already concrete;
    memdesc cleanup requires the contract to fold to an identity after
    retagging.
  }];

  let arguments = (ins AnyTypeOf<[TT_Tensor, TTG_MemDescType]>:$src);
  let results = (outs AnyTypeOf<[TT_Tensor, TTG_MemDescType]>:$result);
  let hasFolder = 1;
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def Gluon_LocalAliasOp : Gluon_Op<"local_alias",
    [Gluon_SameOperandAndResultMemorySpace, Pure]> {
  let summary = "zero-copy storage-reuse alias of a Shared descriptor";
  let description = [{
    Materializes an explicit storage-alias or reuse allocation as a new
    logical Shared-memory root over the same physical backing. Ordinary
    descriptor reinterpretation uses ttg.memdesc_reinterpret instead. This op
    may change shape, element type, or Shared encoding. Its contents and
    lifetime are an independent compiler-analysis boundary: it does not copy,
    initialize, synchronize, infer contents, or validate backing reuse. Layout
    propagation treats it as a hard Shared-layout boundary and
    gluon-rewrite-local-alias later materializes standard
    ttg.memdesc_reinterpret views.
  }];

  let arguments = (ins TTG_MemDescType:$src);
  let results = (outs TTG_MemDescType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def Gluon_ExtractSliceOp
    : Gluon_Op<"extract_slice", [Pure, SameOperandsAndResultEncoding]> {
  let summary = "layout-parametric register tensor slice glue op";
  let description = [{
    A machine-independent register subview. It preserves its existing
    same-encoding contract and does not participate in Shared-offset,
    async-copy, or physical ttg.extract_tensor policy in this migration.
  }];

  let arguments = (ins
    TT_Tensor:$source,
    DenseI64ArrayAttr:$offsets
  );
  let results = (outs TT_Tensor:$result);
  let hasVerifier = 1;
}

def Gluon_InsertSliceOp
    : Gluon_Op<"insert_slice", [Pure, SameOperandsAndResultEncoding]> {
  let summary = "layout-parametric register tensor slice update glue op";
  let description = [{
    A machine-independent register subview update. It preserves its existing
    same-encoding contract and does not participate in Shared-offset,
    async-copy, or physical ttg.insert_tensor policy in this migration.
  }];

  let arguments = (ins
    TT_Tensor:$base,
    TT_Tensor:$update,
    DenseI64ArrayAttr:$offsets
  );
  let results = (outs TT_Tensor:$result);
  let hasVerifier = 1;
}

#endif // GLUON_METAX_OPS
