#ifndef LinalgEXT_PASSES
#define LinalgEXT_PASSES

include "mlir/Pass/PassBase.td"

def LinalgIfToSelect : Pass<"linalg-if-to-select", "mlir::ModuleOp"> {
  let summary = "Convert scf.if inside parallel linalg.generic to arith.select "
                "and hoist selects.";
  let description = [{
    This pass converts conditional logic (`scf.if`) inside a parallel `linalg.generic`
    operation into data-flow operations (`arith.select`).

    The conversion process involves two main stages:
    1.  **LinalgIfToSelectPattern**: Converts `scf.if` statements (where the else
        block yields a constant or is empty) into an equivalent sequence of
        `arith.select` operations. For arguments used inside the `then` block,
        it introduces an `arith.select` using a zero constant for the false case.
        It also attempts to fix zero-value assignment for `memref.store` operations
        that result from the conversion (FixLinalgSelectZeroForStorePattern).
    2.  **LinalgLiftSelectPattern**: Lifts scalar `arith.select` operations that use
        loop-invariant operands or block arguments outside the `linalg.generic`
        operation, converting them into tensor-based `arith.select`s which are
        then added as new inputs to the `linalg.generic`. This process helps to
        simplify the body of the `linalg.generic` and enables further fusion.
  }];
  let constructor = "mlir::dicp::LinalgExt::createLinalgIfToSelectPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect", "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect", "mlir::scf::SCFDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def LinalgGenericToSCF : Pass<"linalg-generic-to-scf", "mlir::ModuleOp"> {
  let summary = "Lower linalg.generic ops to SCF loops";
  let description = [{
    Converts linalg.generic operations into explicit scf.for loops,
    generating structured loop nests that preserve the original
    indexing maps and iterator types.
  }];
  let constructor = "mlir::dicp::LinalgExt::createLinalgGenericToSCFPass()";
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}

def ScalarTo1DTensor : Pass<"scalar-to-1d-tensor", "mlir::func::FuncOp"> {
  let summary =
      "Convert scalar computations and memref load/store to tensor<1 x T> form";
  let description =
      [{The ScalarTo1DTensor pass targets scalar computations and memory access
            within a function and rewrites them into an explicit tensor<1 x T> -
        based form
            .This enables uniform handling of scalar values in subsequent
                bufferization and lowering passes.}];

  let constructor = "mlir::dicp::LinalgExt::createScalarTo1DTensorPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect", "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect", "mlir::bufferization::BufferizationDialect",
    "mlir::func::FuncDialect"
  ];
}

def NormalizeSliceOps : Pass<"normalize-slice-ops", "func::FuncOp"> {
  let summary = "Normalize Slice Ops.";
  let constructor = "mlir::dicp::LinalgExt::createNormalizeSliceOpsPass()";
  let dependentDialects = ["mlir::tensor::TensorDialect"];
}

def VectorizeParallelLoop : Pass<"vectorize-parallel-loop", "func::FuncOp"> {
  let summary = "Convert single-element parallel loops to vectorized batch "
                "processing with step=size.";
  let description = [{This pass transforms parallel loops that process single
                          elements into vectorized operations that can process
                              multiple elements simultaneously,
                      potentially increasing computational throughput.}];
  let constructor = "mlir::dicp::LinalgExt::createVectorizeParallelLoopPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect", "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect", "mlir::scf::SCFDialect",
    "mlir::bufferization::BufferizationDialect", "mlir::func::FuncDialect"
  ];
}

def PipelineLoopUnroll : Pass<"pipeline-loop-unroll", "func::FuncOp"> {
  let summary = "Unroll a software-pipelined loop by its stage count.";
  let description = [{
    Finds the target software-pipelined `scf.for`, reorders stage-local
    operations according to dependency analysis, marks yielded loop-carried
    values, and expands the loop body across pipeline stages.
  }];
  let constructor = "mlir::dicp::LinalgExt::createPipelineLoopUnrollPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::func::FuncDialect"
    ];
}

def StageTilingPlan : Pass<"stage-tiling-plan", "func::FuncOp"> {
  let summary = "Analyze staged ops and attach tiling and fusion metadata";
  let description = [{
    This pass analyzes staged operations to identify tiling anchors and fusion
    candidates.
    It marks operations with attributes to guide the subsequent transform pass.
  }];
  let constructor = "mlir::dicp::LinalgExt::createStageTilingPlanPass()";
  let options = [
    Option<"tiledMixVectorLoopNumber", "vector-tile", "unsigned",
           /*default=*/"2", "Trip count for vector loop tiling">,
    Option<"rejectStrideInconsistentStage", "reject-stride-inconsistent-stage",
           "bool", /*default=*/"true",
           "When true, skip tiling for an entire stage if any memref.copy "
           "within it has mismatched src/dst strides (layout-changing copy)">,
    Option<"tileAllBlocks", "tile-all-blocks", "bool", /*default=*/"true",
           "When true, tile all eligible blocks and fail if any unit violates "
           "eligibility rules. When false, tile only the densest block.">,
  ];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::memref::MemRefDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::scf::SCFDialect"
  ];
}

def CanonicalizeTileableOps : Pass<"canonicalize-tileable-ops", "func::FuncOp"> {
  let summary = "Canonicalize IR into tile-friendly forms before tiling";
  let description = [{
    This pass canonicalizes pre-tagging IR into tile-friendly forms using
    greedy rewrites:
    1. Sinks `bufferization.to_tensor` directly after `memref.alloc`.
    2. Converts stage-marked elementwise ops to `linalg.generic`.
    3. Lowers copy-like ops (`bufferization.materialize_in_destination` and
       `memref.copy`) to `linalg.copy`.
    The pass is intentionally separated from tagging to keep normalization and
    analysis pipelines modular and easier to debug.
  }];
  let constructor = "mlir::dicp::LinalgExt::createCanonicalizeTileableOpsPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::memref::MemRefDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::func::FuncDialect"
  ];
}

def DuplicateStageProducers : Pass<"duplicate-stage-producers", "func::FuncOp"> {
  let summary = "Duplicate cross-unit SIMD-like producers per use-site";
  let description = [{
    This pass breaks cross-unit SSA sharing for SIMD-like pure compute ops
    within the same stage/substage. For each direct cross-unit use, it clones a
    private producer closure next to the consumer, retags the clone for the
    consumer's unit, and drops tiling anchor metadata from the duplicate.
  }];
  let constructor =
      "mlir::dicp::LinalgExt::createDuplicateStageProducersPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::func::FuncDialect"
  ];
}

def ApplyTilingPlan : Pass<"apply-tiling-plan", "ModuleOp"> {
  let summary = "Apply tiling and fusion using Transform Dialect based on tags";
  let description = [{
    This pass reads the tags generated by the tagging pass and executes a
    Transform Dialect sequence to perform the actual tiling and fusion.
  }];
  let constructor = "mlir::dicp::LinalgExt::createApplyTilingPlanPass()";
  let dependentDialects = [
    "mlir::transform::TransformDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect"
  ];
}

def DeLinalgize : Pass<"de-linalgize", "mlir::ModuleOp"> {
  let summary = "De-linalgize Linalg operations back to specific dialects.";
  let description = [{
    This pass restores high-level operations from their `linalg` representations:
    1. Converts `linalg.generic` back to elementwise operations (e.g., `arith` or `math`).
    2. Converts `linalg.copy` back to `bufferization.materialize_in_destination`
       or `memref.copy`.
    It is essentially the inverse of normalization/generalization passes.
  }];

  let constructor = "mlir::dicp::LinalgExt::createDeLinalgizePass()";

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::memref::MemRefDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::linalg::LinalgDialect"
  ];
}

def FuseLoop : Pass<"fuse-loop", "mlir::ModuleOp"> {
  let summary = "Fuse cube and vector loops";
  let constructor = "mlir::dicp::LinalgExt::createFuseLoopPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::transform::TransformDialect"
  ];
}

def LowerForallToFor : Pass<"lower-forall-to-for", "mlir::func::FuncOp"> {
  let summary = "Lower scf.forall to scf.for with sequential accumulation.";
  let description = [{
    This pass rewrites every `scf.forall` (parallel loop with shared outputs) into
    a nest of `scf.for` loops that accumulate results sequentially via
    `tensor.insert_slice`.  Attributes on the original `scf.forall` are preserved
    on the outermost generated `scf.for`.
  }];
  let constructor = "mlir::dicp::LinalgExt::createLowerForallToForPass()";
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::func::FuncDialect"
  ];
}

def UnrollStageLoops : Pass<"unroll-stage-loops", "mlir::func::FuncOp"> {
  let summary = "Fully unroll scf.for loops marked with stage attributes.";
  let description = [{
    This pass collects `scf.for` loops carrying the `dicp_stage` attribute and
    fully unrolls them when constant bounds are available.  After unrolling it
    cleans up all internal `dicp.tmp.stage.*` attributes.
  }];
  let constructor = "mlir::dicp::LinalgExt::createUnrollStageLoopsPass()";
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::func::FuncDialect"
  ];
}

def ShrinkBuffers : Pass<"shrink-buffers", "mlir::ModuleOp"> {
  let summary = "Shrink memref.alloc and tensor.empty based on consistent slicing usage.";
  let description = [{
    Runs two rewrite patterns (ShrinkAllocWithSlicing and ShrinkEmptyTensorWithSlicing)
    to reduce allocation/empty sizes when all slices agree on a smaller size.
  }];
  let constructor = "mlir::dicp::LinalgExt::createShrinkBuffersPass()";
  let options = [
    Option<"tileAllBlocks", "tile-all-blocks", "bool", /*default=*/"true",
           "When true, enable DecomposeLoopTensorIterArgsPattern to decompose "
           "loop tensor iter_args into partition-scoped iter_args.">,
  ];
}

def PrepareVectorTiling
    : Pass<"prepare-vector-tiling", "mlir::func::FuncOp"> {
  let summary = "Run early vector-tile legality checks and buffer-to-tensor cleanup";
  let description = [{
    This pass performs early preflight preparation for the vector-tiling
    pipeline.

    First, it enforces global legality constraints that must hold before any
    stage tagging or transform-dialect driven tiling is attempted.

    Second, it canonicalizes selected buffer-semantics `linalg.fill`
    operations into tensor semantics so that downstream tiling and fusion
    passes operate on a cleaner IR.
  }];
  let constructor = "mlir::dicp::LinalgExt::createPrepareVectorTilingPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def VerifyNoLocalBufferCopy
    : Pass<"verify-no-local-buffer-copy", "mlir::ModuleOp"> {
  let summary = "Verify that no memref.copy writes into a locally allocated buffer.";
  let description = [{
    This verification pass walks all `memref.copy` operations and checks whether
    the **destination** operand traces back (through subview / cast chains) to a
    `memref.alloc`.  If such a copy exists, the pass emits an error diagnostic,
    because after the tiling-and-shrink pipeline every local allocation should
    have been eliminated or replaced by precisely-sized independent buffers that
    do not require an intermediate copy.

    The pass also flags `tensor.empty` ops that are still reachable from a copy
    destination via `bufferization.to_tensor` → `tensor.insert_slice` chains.
  }];
  let constructor = "mlir::dicp::LinalgExt::createVerifyNoLocalBufferCopyPass()";
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::bufferization::BufferizationDialect"
  ];
}

def ScheduleInputBundles
    : Pass<"schedule-input-bundles", "mlir::func::FuncOp"> {
  let summary = "Sink local input bundles toward the earliest in-substage consumer.";
  let description = [{
    This pass performs a conservative late-I/O scheduling pass on local buffers.
    It recognizes simple input bundles rooted at `memref.alloc`:

      alloc -> optional local subviews -> memref.copy -> bufferization.to_tensor

    and sinks the entire bundle toward the earliest use of the tensor result,
    while remaining inside the same recovered stage/substage and preserving
    SSA and memory ordering.

    The implementation intentionally favors low complexity over maximum
    aggressiveness:
      * only same-block, same-substage bundles are considered;
      * unsupported or ambiguous users are skipped;
      * sync ops and structured control flow terminate motion;
      * if legality cannot be proven cheaply, the bundle is left unchanged.
  }];
  let constructor = "mlir::dicp::LinalgExt::createScheduleInputBundlesPass()";
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::func::FuncDialect"
  ];
}

def PrepareInputBundleScheduling
    : Pass<"prepare-input-bundle-scheduling", "mlir::ModuleOp"> {
  let summary = "Canonicalize and fold memref alias ops before bundle scheduling.";
  let description = [{
    This pass is a dedicated preprocessing stage for substage-local bundle
    scheduling. It intentionally stays separate from the scheduler itself and
    runs a small canonicalization pipeline:

      1. Canonicalizer
      2. fold-memref-alias-ops

    The goal is to simplify subview / reinterpret_cast chains and expose a
    more regular memref use-def structure before the late-I/O scheduler runs.
  }];
  let constructor =
      "mlir::dicp::LinalgExt::createPrepareInputBundleSchedulingPass()";
  let dependentDialects = [
    "mlir::memref::MemRefDialect"
  ];
}

def VectorTilingPipeline : Pass<"vector-tiling-pipeline", "mlir::ModuleOp"> {
  let summary = "Full vector tiling, fusion, and unrolling pipeline with rollback.";
  let description = [{
    Orchestrates the complete vector-tiling pipeline in order:
      1. prepare-vector-tiling          — early legality checks and cleanup
      2. stage-tiling-plan              — tag tiling anchors and fusion candidates
      3. duplicate-stage-producers      — clone cross-unit pure producers
      4. canonicalize-tileable-ops      — canonicalize IR into tile-friendly forms
      5. apply-tiling-plan              — apply tiling/fusion via Transform Dialect
      6. lower-forall-to-for            — lower scf.forall to scf.for
      7. fuse-loop                      — fuse sibling loops
      8. de-linalgize                   — restore high-level ops from linalg
      9. unroll-stage-loops             — fully unroll staged loops
      10. shrink-buffers                — shrink allocations to actual usage
      11. prepare-input-bundle-scheduling — simplify alias ops before sinking
      12. schedule-input-bundles        — sink local input bundles late

    If any sub-pass fails, the module IR is rolled back to its pre-pipeline
    state so that downstream passes can proceed on the original IR.
  }];
  let constructor = "mlir::dicp::LinalgExt::createVectorTilingPipelinePass()";
  let options = [
    Option<"vectorTile", "vector-tile", "unsigned",
           /*default=*/"2", "Trip count for vector loop tiling">,
    Option<"tileAllBlocks", "tile-all-blocks", "bool", /*default=*/"true",
           "When true, tile all eligible blocks and fail if any unit violates "
           "eligibility rules. When false, tile only the densest block.">,
  ];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::math::MathDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::transform::TransformDialect"
  ];
}

#endif
