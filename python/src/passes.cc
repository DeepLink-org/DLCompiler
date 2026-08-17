#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_triton_analysis(py::module &&m) {
  py::class_<mlir::ModuleAllocation>(m, "allocation", py::module_local())
      .def(py::init<mlir::ModuleOp>());
  py::class_<mlir::ModuleMembarAnalysis>(m, "membar", py::module_local())
      .def(py::init<mlir::ModuleAllocation *>())
      .def("run", &mlir::ModuleMembarAnalysis::run);
}

void init_triton_passes_common(py::module &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_sccp", createSCCPPass);
  ADD_PASS_WRAPPER_0("add_symbol_dce", createSymbolDCEPass);
  ADD_PASS_WRAPPER_0("add_inliner", createInlinerPass);
  ADD_PASS_WRAPPER_0("add_canonicalizer", createCanonicalizerPass);
  ADD_PASS_WRAPPER_0("add_cse", createCSEPass);
  ADD_PASS_WRAPPER_0("add_licm", createLoopInvariantCodeMotionPass);
  ADD_PASS_WRAPPER_0("print_ir", createPrintIRPass);
}

void init_triton_passes_ttir(py::module &&m) {
  using namespace mlir::triton;
  ADD_PASS_WRAPPER_0("add_combine", createTritonCombineOps);
  ADD_PASS_WRAPPER_0("add_reorder_broadcast", createTritonReorderBroadcast);
  ADD_PASS_WRAPPER_0("add_rewrite_tensor_pointer",
                     createTritonRewriteTensorPointer);
  ADD_PASS_WRAPPER_0("add_rewrite_tensor_descriptor_to_pointer",
                     createTritonRewriteTensorDescriptorToPointer);
  ADD_PASS_WRAPPER_0("add_loop_unroll", createTritonLoopUnroll);
  ADD_PASS_WRAPPER_0("add_triton_licm", createTritonLoopInvariantCodeMotion);
  ADD_PASS_WRAPPER_0("add_loop_aware_cse", createTritonLoopAwareCSE);
  ADD_PASS_OPTION_WRAPPER_4("add_convert_to_ttgpuir",
                            createConvertTritonToTritonGPU, const std::string &,
                            int, int, int);
}

void init_triton_passes_ttgpuir(py::module &&m) {
  using namespace mlir;
  using namespace mlir::triton::gpu;
  using namespace mlir::triton::instrument;
  ADD_PASS_WRAPPER_0("add_coalesce", createTritonGPUCoalesce);
  ADD_PASS_WRAPPER_0("add_optimize_thread_locality",
                     createTritonGPUOptimizeThreadLocality);
  ADD_PASS_OPTION_WRAPPER_1("add_hoist_tmem_alloc",
                            createTritonGPUHoistTMEMAlloc, bool);
  ADD_PASS_OPTION_WRAPPER_1("add_assign_latencies",
                            createTritonGPUAssignLatencies, int);
  ADD_PASS_WRAPPER_0("add_schedule_loops", createTritonGPUScheduleLoops);
  ADD_PASS_OPTION_WRAPPER_2("add_pipeline", createTritonGPUPipeline, int, bool);
  ADD_PASS_OPTION_WRAPPER_1("add_warp_specialize",
                            createTritonGPUAutomaticWarpSpecialization, int);
  ADD_PASS_WRAPPER_0("add_prefetch", createTritonGPUPrefetch);
  ADD_PASS_WRAPPER_0("add_accelerate_matmul", createTritonGPUAccelerateMatmul);
  ADD_PASS_WRAPPER_0("add_reorder_instructions",
                     createTritonGPUReorderInstructions);
  ADD_PASS_OPTION_WRAPPER_1("add_f32_dot_tc", createTritonGPUF32DotTC, bool);
  ADD_PASS_OPTION_WRAPPER_1("add_optimize_dot_operands",
                            createTritonGPUOptimizeDotOperands, bool);
  ADD_PASS_WRAPPER_0("add_remove_layout_conversions",
                     createTritonGPURemoveLayoutConversions);
  ADD_PASS_WRAPPER_0("add_reduce_data_duplication",
                     createTritonGPUReduceDataDuplication);
  ADD_PASS_WRAPPER_0("add_allocate_warp_groups",
                     createTritonGPUAllocateWarpGroups);
  ADD_PASS_WRAPPER_0("add_allocate_shared_memory", createAllocateSharedMemory);
  ADD_PASS_WRAPPER_0("add_allocate_global_scratch_memory",
                     createTritonGPUGlobalScratchAllocationPass);
  ADD_PASS_WRAPPER_0("add_combine_tensor_select_and_if",
                     createTritonGPUCombineTensorSelectAndIf);
  ADD_PASS_WRAPPER_0("add_optimize_accumulator_init",
                     createTritonGPUOptimizeAccumulatorInit);
  ADD_PASS_WRAPPER_0("add_fuse_nested_loops", createTritonGPUFuseNestedLoops);
  ADD_PASS_WRAPPER_0("add_coalesce_async_copy",
                     createTritonGPUCoalesceAsyncCopy);
  ADD_PASS_WRAPPER_0("add_concurrency_sanitizer",
                     createTritonInstrumentConcurrencySanitizer);
  ADD_PASS_WRAPPER_0("add_optimize_partition_warps",
                     createTritonGPUOptimizePartitionWarps);
}

void init_triton_passes_convert(py::module &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_scf_to_cf", createSCFToControlFlowPass);
  ADD_PASS_WRAPPER_0("add_cf_to_llvmir", createConvertControlFlowToLLVMPass);
  ADD_PASS_WRAPPER_0("add_index_to_llvmir", createConvertIndexToLLVMPass);
  ADD_PASS_WRAPPER_0("add_arith_to_llvmir", createArithToLLVMConversionPass);
  ADD_PASS_WRAPPER_0("add_nvvm_to_llvm", createConvertNVVMToLLVMPass);
}

void init_triton_passes_llvmir(py::module &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_di_scope", mlir::createLLVMDIScope);
  ADD_PASS_WRAPPER_0("add_di_local_variable", mlir::createLLVMDILocalVariable);
}

void init_gluon_metax_passes(py::module &&m) {
  using namespace mlir;
  namespace gluon = mlir::triton::gluon;
  ADD_PASS_WRAPPER_4("add_accelerate_matmul",
                     gluon::createGluonAccelerateMatmulPass, int, bool, bool,
                     int);
  ADD_PASS_WRAPPER_0("add_align_mma_consumers",
                     gluon::createGluonAlignMmaConsumersPass);
  ADD_PASS_WRAPPER_0("add_storage_alias_lowering",
                     gluon::createGluonStorageAliasLoweringPass);
  ADD_PASS_WRAPPER_0("add_insert_require_layout",
                     gluon::createGluonInsertRequireLayoutPass);
  ADD_PASS_WRAPPER_0("add_propagate_layout",
                     gluon::createGluonPropagateLayoutPass);
  ADD_PASS_WRAPPER_0("add_rewrite_local_alias",
                     gluon::createGluonRewriteLocalAliasPass);
  ADD_PASS_WRAPPER_0("add_gluon_to_tritongpu_conversion",
                     gluon::createGluonToTritonGPUConversionPass);
  m.def("add_mma_layout_candidates", [](PassManager &pm, int capability) {
    pm.addNestedPass<mlir::triton::FuncOp>(
        gluon::createGluonMmaLayoutCandidatePass(capability));
  });
  m.def("add_shared_layout_candidates", [](PassManager &pm) {
    pm.addNestedPass<mlir::triton::FuncOp>(
        gluon::createGluonSharedLayoutCandidatePass());
  });
  m.def("add_blocked_layout_candidates", [](PassManager &pm) {
    pm.addNestedPass<mlir::triton::FuncOp>(
        gluon::createGluonBlockedLayoutCandidatePass());
  });
  ADD_PASS_WRAPPER_1("add_expand_layout_candidates",
                     gluon::createGluonExpandLayoutCandidatesPass, int);
}

void init_gluon_passes(py::module &&m) {
  namespace gluon = mlir::triton::gluon;
  ADD_PASS_WRAPPER_0("add_resolve_auto_encodings",
                     gluon::createGluonResolveAutoEncodingsPass);
  ADD_PASS_WRAPPER_0("add_canonicalizer", gluon::createGluonCanonicalize);
  ADD_PASS_WRAPPER_0("add_inliner", gluon::createGluonInline);
  ADD_PASS_WRAPPER_0("add_infer_coalesced_encodings",
                     gluon::createGluonInferCoalescedEncodingsPass);
  init_gluon_metax_passes(m.def_submodule("metax"));
}

void init_triton_passes(py::module &&m) {
  init_triton_analysis(m.def_submodule("analysis"));
  init_triton_passes_common(m.def_submodule("common"));
  init_triton_passes_convert(m.def_submodule("convert"));
  init_triton_passes_ttir(m.def_submodule("ttir"));
  init_triton_passes_ttgpuir(m.def_submodule("ttgpuir"));
  init_triton_passes_llvmir(m.def_submodule("llvmir"));
  init_gluon_passes(m.def_submodule("gluon"));
}
