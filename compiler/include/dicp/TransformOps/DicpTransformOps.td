#ifndef DICP_DICPTRANSFORMOPS
#define DICP_DICPTRANSFORMOPS

include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Dialect/Linalg/TransformOps/LinalgTransformEnums.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"



//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

def ReverseOp : Op<Transform_Dialect, "reverse",
  [FunctionalStyleTransformOpTrait,
   DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
   TransformOpInterface]> {
  let description = [{
    This transform op gets and reverses the list of operations held by the
    input `target` handle.

    This transform reads the `target` handle and produces the `result` handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$result);

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];
  let assemblyFormat = [{
    $target attr-dict `:` functional-type($target, $result)
  }];
}

def ForwardInitToIterArgOp : Op<Transform_Dialect, "forward_init_to_iter_arg",
    [FunctionalStyleTransformOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformOpInterface]> {
  let description = [{
    Target must be a handle to `scf.for` operations.
    
    This transform analyzes the `scf.for` loop body. If it finds an `extract_slice`
    using the loop's init_arg that is structurally equivalent to the `insert_slice`
    yielded into the corresponding iter_arg, it replaces the `extract_slice` source
    with the iter_arg.

    This enables In-Place updates and removes dependencies on the init_arg.
  }];

  // 输入是一个指向 scf.for 的 Handle
  let arguments = (ins TransformHandleTypeInterface:$target);
  // 输出是处理后的 scf.for Handle (通常还是原来的 Op，但内容变了)
  let results = (outs TransformHandleTypeInterface:$result);

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];

  let assemblyFormat = [{
    $target attr-dict `:` functional-type($target, $result)
  }];
}

//===----------------------------------------------------------------------===//
// ExtendedFuseIntoContainingOp
//===----------------------------------------------------------------------===//
def ExtendedFuseIntoContainingOp :
  Op<Transform_Dialect, "structured.extended_fuse_into_containing_op",
     [DeclareOpInterfaceMethods<TransformOpInterface,
          ["allowsRepeatedHandleOperands"]>,
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      ReportTrackingListenerFailuresOpTrait,
      AttrSizedResultSegments,
      FunctionalStyleTransformOpTrait]> {
  let summary = "Fuse a producer into a containing operation.";

  let description = [{
    Fuses the `producer_op` into the `containing_op`.
    Returns a handle to the fused ops and the `new_containing_op`.

    The producer is typically a slice of a tileable op (i.e., implements
    TilingInterface). In that case, this transform computes the accessed
    producer slice inside of the containing op ("tile and fuse") and if required,
    creates a new containing op with outputs from the fused producer. Otherwise,
    the entire producer is cloned inside the containing op ("clone and fuse").

    Each containing op handle must be associated with exactly one payload op. The
    producer op handle may be associated with multiple payload ops. This
    transform fuses producers one-by-one, always picking an unspecified producer
    that has at least one use inside the containing op among the
    producers. A producer can be listed multiple times in the handle.

    If the `producer_op` has uses that are post-dominated by the `containing_op`,
    then it is fused into `containing_op` completely to avoid recomputation.
    This behavior can be disabled by setting `duplicate_producer` to true.

    Note: If a producer has multiple uses inside the containing op, a union
    of the requested regions is computed, and each consumer will only access the
    region it needs via slicing.

    #### Return modes

    If at least one producer could not be fused, this operation produces a
    silenceable failure.  This is the case when tiling fails or when no
    producer op could be found among the remaining producers that has at least
    one use within the containing op. I.e., "producers" that are not consumed
    within the containing op are rejected by this operation.

    This operation consumes the producer handle.
    This operation only reads the containing op handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$producer_op,
                       Variadic<TransformHandleTypeInterface>:$containing_op,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$duplicate_producer);

  let results = (outs Variadic<TransformHandleTypeInterface>:$fused_op,
                      Variadic<TransformHandleTypeInterface>:$new_containing_op);

  let builders = [
    OpBuilder<(ins "Value":$producerOp, "Value":$containingOp)>
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure fuseIntoOneContaining(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &results,
        ::mlir::transform::TransformState &state,
        size_t index,
        ::mlir::Operation* containingOp);
  }];

  let hasCustomAssemblyFormat = 1;
}

def ExtendedLoopFuseSiblingOp : Op<Transform_Dialect, "loop.extended_fuse_sibling",
  [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
   DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Fuse a loop into another loop, assuming the fusion is legal.";

  let description = [{
    Fuses the `target` loop into the `source` loop assuming they are
    independent of each other. In the fused loop, the arguments, body and
    results of `target` are placed _before_ those of `source`.

    For fusion of two `scf.for` loops, the bounds and step size must match. For
    fusion of two `scf.forall` loops, the bounds and the mapping must match.
    Otherwise a silencable failure is produced.

    The `target` and `source` handles must refer to exactly one operation,
    otherwise a definite failure is produced. It is the responsibility of the
    user to ensure that the `target` and `source` loops are independent of each
    other -- this op will only perform rudimentary legality checks.

    #### Return modes

    This operation consumes the `target` and `source` handles and produces the
    `fused_loop` handle, which points to the fused loop.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       TransformHandleTypeInterface:$source);
  let results = (outs TransformHandleTypeInterface:$fused_loop);
  let assemblyFormat = "$target `into` $source attr-dict "
                       " `:` functional-type(operands, results)";
}

#endif // DICP_DicpTransformOps
