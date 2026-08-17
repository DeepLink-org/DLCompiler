#ifndef GLUON_METAX_PASSES
#define GLUON_METAX_PASSES

include "mlir/Pass/PassBase.td"

def GluonAccelerateMatmulPass
    : Pass<"gluon-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "accelerate C500 Gluon matmul with aggregate-leaf aware MMA layout";
  let description = [{
    This pass is the Gluon C500 M0 authority. Its Blocked-to-MACA rewrite is
    kept aligned with the existing MetaX AccelerateMETAXMatmul implementation:
    it creates the accumulator and A/B conversions into one concrete
    MACAMmaEncodingAttr contract, rebuilds the dot, and converts the result
    back to the pre-rewrite result type so ordinary users keep a legal layout.

    The only Gluon-specific selection rule recognizes a direct
    extract_slice -> dot -> insert_slice accumulator leaf and prevents the
    generic multi-dot shortcut from forcing an unrepresentable one-dimensional
    warp topology. All remaining MACA fields keep the existing C500 baseline;
    Shared layout requirements remain the responsibility of
    gluon-insert-require-layout.
  }];
  let constructor =
      "mlir::triton::gluon::createGluonAccelerateMatmulPass()";
  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
  ];
  let options = [
    Option<"numStages", "num-stages", "int32_t", /*default*/"2",
           "number of pipeline stages">,
    Option<"disablePrefetch", "disable-prefetch", "bool", /*default*/"false",
           "disable prefetch">,
    Option<"storeCoalesce", "store-coalesce", "bool", /*default*/"false",
           "enable store coalescing">,
    Option<"computeCapability", "compute-capability", "int32_t", /*default*/"80",
           "C500 compute capability">,
  ];
}

def GluonAlignMmaConsumersPass
    : Pass<"gluon-align-mma-consumers", "mlir::ModuleOp"> {
  let summary =
      "align proven register-only consumers with an upstream C500 MMA layout";
  let description = [{
    Recognizes two bounded C500 ownership peepholes after MMA acceleration:
    a register-only value flowing into a dot operand, and a register-only
    value flowing into tensor atomic_rmw. If the backward conversion slice has
    one upstream MACA MMA encoding and existing layout inference proves that
    the consumer can use it, the pass retargets only that consumer. Existing
    InsertRequireLayout, PropagateLayout, and RemoveLayoutConversions passes
    remain responsible for Shared constraints and rematerialization.

    The pass does not form global MMA families, enumerate layout candidates,
    cross Shared-memory boundaries, or modify synchronization.
  }];
  let constructor =
      "mlir::triton::gluon::createGluonAlignMmaConsumersPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
  ];
}

def GluonStorageAliasLoweringPass
    : Pass<"gluon-storage-alias-lowering", "mlir::ModuleOp"> {
  let summary = "lower explicit Shared storage aliases";
  let description = [{
    Computes or validates each storage_alias_spec size, applies an optional
    TLX-style reuse-group overlap tree to calculate buffer offsets, creates one
    standard Shared backing allocation, and replaces every
    storage_alias_local_alloc with a gluon.local_alias root. It does not copy
    data or infer synchronization and lifetime.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::triton::gluon::GluonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
  ];
}

def GluonPropagateLayoutPass : Pass<"gluon-propagate-layout", "mlir::ModuleOp"> {
  let summary = "propagate explicit Shared and register layout requirements";
  let description = [{
    This pass directly follows TLX's analyze-then-rewrite structure. One
    DataFlowSolver runs memdesc backward propagation, memdesc forward
    propagation, and conservative tensor backward propagation to a fixed
    point before mutating types.

    The pass retags concrete memdesc producers and native memdesc views,
    supported tensor producers (`ttg.local_load`), and consensus-backed
    RegionBranchOpInterface carriers. `gluon.local_alias` is a hard logical
    Shared-layout boundary. Warp-specialize capture and partition argument
    types are updated only when their lattice values have one concrete
    consensus.

    Residual tensor requirements become `ttg.convert_layout`; residual
    memdesc requirements are errors because memdesc layout conversion is not
    a register conversion. This is the only layout solver in the C500 Gluon
    pipeline.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::triton::gluon::GluonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
  ];
}

def GluonInsertRequireLayoutPass
    : Pass<"gluon-insert-require-layout", "mlir::ModuleOp"> {
  let summary = "synthesize baseline C500 Gluon layout requirements";
  let description = [{
    This pass follows TLX's two-phase DotRewriteBackward structure. Analysis
    first walks existing C500 Accelerate convert_layout-to-dot A/B chains and
    transparent RegionBranch carriers without changing IR. Materialization
    then replaces each direct M0 dot-operand conversion with a tensor
    `gluon.require_layout`; for a unique legal `ttg.local_load` source it also
    derives S0 with the existing MACA Shared composer and inserts a memdesc
    requirement immediately before that load.

    Unsupported or conflicting paths keep their real register conversion and
    do not synthesize a Shared requirement. This pass does not retag values,
    inspect global accesses or async-copy endpoints, or schedule
    synchronization; those are outside this phase.
  }];
  let constructor =
      "mlir::triton::gluon::createGluonInsertRequireLayoutPass()";
  let dependentDialects = [
    "mlir::triton::gluon::GluonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
  ];
}

def GluonToTritonGPUConversionPass
    : Pass<"gluon-to-tritongpu-conversion", "mlir::ModuleOp"> {
  let summary = "lower late Gluon register views and C500 logical BSM";
  let description = [{
    Lowers rank-2 Gluon extract_slice and insert_slice through direct
    ttg.extract_tensor and ttg.insert_tensor patterns when their existing
    same-encoding logical view is a complete physical CTA/register partition.
    Layout analysis treats logical bsm_perm as a transparent same-encoding
    register carrier. This pass consumes its direct local_load source and
    materializes the existing i32 C500 BSM ABI without changing that encoding.
    It never creates a register layout conversion as a slice fallback.
  }];
  let constructor =
      "mlir::triton::gluon::createGluonToTritonGPUConversionPass()";
  let dependentDialects = [
    "mlir::triton::gluon::GluonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
  ];
}

def GluonRewriteLocalAliasPass : Pass<"gluon-rewrite-local-alias", "mlir::ModuleOp"> {
  let summary = "materialize gluon.local_alias as zero-copy memdesc reinterprets";
  let description = [{
    This pass is the final half of the TLX storage-alias contract. It groups
    aliases by their standard `ttg.local_alloc` backing, grows the backing only
    when a larger view requires it, and replaces every `gluon.local_alias` with
    a zero-copy `ttg.memdesc_reinterpret`. It does not copy data or infer
    synchronization, ownership, or lifetime.
  }];
  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
  ];
}

def GluonMmaLayoutCandidatePass
    : Pass<"gluon-mma-layout-candidates", "mlir::triton::FuncOp"> {
  let summary = "enumerate complete C500 MMA layout assignments";
  let description = [{
    Reads final concrete TTGIR and attaches at most five non-baseline C500 MMA
    component evolutions.  The pass does not clone IR or modify the baseline.
  }];
  let constructor =
      "mlir::triton::gluon::createGluonMmaLayoutCandidatePass()";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
  let options = [
    Option<"computeCapability", "compute-capability", "int32_t",
           /*default*/"80", "C500 compute capability">,
  ];
}

def GluonSharedLayoutCandidatePass
    : Pass<"gluon-shared-layout-candidates", "mlir::triton::FuncOp"> {
  let summary = "enumerate the bounded C500 Shared layout portfolio";
  let description = [{
    Evolves every concrete SwizzledShared owner in four correlated whole-
    function lanes while preserving order, CTA layout and memdesc contracts.
  }];
  let constructor =
      "mlir::triton::gluon::createGluonSharedLayoutCandidatePass()";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}

def GluonBlockedLayoutCandidatePass
    : Pass<"gluon-blocked-layout-candidates", "mlir::triton::FuncOp"> {
  let summary = "enumerate B1 and B3 for every concrete Blocked layout";
  let description = [{
    Uses each concrete Blocked encoding as its own owner and emits one complete
    function assignment per legal B1 or B3 neighbor.  No anchor discovery or
    cost model participates in candidate generation.
  }];
  let constructor =
      "mlir::triton::gluon::createGluonBlockedLayoutCandidatePass()";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}

def GluonExpandLayoutCandidatesPass
    : Pass<"gluon-expand-layout-candidates", "mlir::ModuleOp"> {
  let summary = "clone and verify one stage of complete layout assignments";
  let description = [{
    This single generic Expand implementation is invoked after MMA, Shared and
    Blocked candidate generation.  It contains no layout evolution formulas.
  }];
  let constructor =
      "mlir::triton::gluon::createGluonExpandLayoutCandidatesPass()";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
  let options = [
    Option<"stage", "stage", "int32_t", /*default*/"0",
           "candidate stage: 0=MMA, 1=Shared, 2=Blocked">,
  ];
}

#endif // GLUON_METAX_PASSES
