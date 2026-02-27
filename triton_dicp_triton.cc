#include "dicp/Conversion/DiscreteMaskAccessConversion/Passes.h"
#include "dicp/Conversion/LinalgToLinked/LinalgToLinked.h"
#include "dicp/Conversion/LinalgToLinked/Passes.h"
#include "dicp/Conversion/LinalgToNPU/Passes.h"
#include "dicp/Conversion/LinkedToHIVM/Passes.h"
#include "dicp/Conversion/TritonToLinalgNPU/TritonToLinalgNPUCoversion/Passes.h"
#include "dicp/Conversion/TritonToUnstructure/BubbleUpOperation.h"
#include "dicp/Conversion/TritonToUnstructure/UnstructureConversionPass.h"
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"
#include "dicp/Dialect/TritonExt/Transforms/Passes.h"
#include "dicp/TransformOps/DicpTransformOps.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Conversion/FuncToEmitC/FuncToEmitC.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/ShardingInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/TransformOps/DLTITransformOps.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/Transforms/AllInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemRefMemorySlot.h"
#include "mlir/Dialect/MemRef/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Extensions/AllExtensions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/DebugExtension/DebugExtension.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IRDLExtension/IRDLExtension.h"
#include "mlir/Dialect/Transform/LoopExtension/LoopExtension.h"
#include "mlir/Dialect/Transform/PDLExtension/PDLExtension.h"
#include "mlir/Dialect/Transform/TuneExtension/TuneExtension.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/Constants.h"

#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "passes.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace mlir;

void init_triton_dicp_triton_pass_triton_shared_ascend(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_canonicalize_cmpi",
                     dicp::trtion_ext::createCanonicalizeCmpiPass);
  ADD_PASS_WRAPPER_0("add_canonicalize_triton_ir_ascend",
                     dicp::trtion_ext::createCanonicalizeTritonIRAscendPass);
  ADD_PASS_WRAPPER_0("add_triton_to_linalg_npu",
                     dicp::linked::createTritonToLinalgNPUCoversionPass);
  ADD_PASS_OPTION_WRAPPER_2("add_discrete_mask_access_conversion",
                            triton::createDiscreteMaskAccessConversionPass,
                            bool, bool);
  ADD_PASS_WRAPPER_0("add_triton_to_unstructure",
                     triton::createTritonToUnstructurePass);
  ADD_PASS_WRAPPER_0("add_bubble_up_operation",
                     triton::createBubbleUpOperationPass);
}

void init_triton_dicp_triton_pass_linked_npu(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_lower_affine", createLowerAffinePass);
  m.def("add_normalize_slice_ops", [](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::func::FuncOp>(
        dicp::LinalgExt::createNormalizeSliceOpsPass());
  });
  ADD_PASS_WRAPPER_0("add_linalg_if_to_select",
                     dicp::LinalgExt::createLinalgIfToSelectPass);
  ADD_PASS_WRAPPER_0("add_linalg_generic_to_scf",
                     dicp::LinalgExt::createLinalgGenericToSCFPass);
  m.def("add_scalar_to_1d_tensor", [](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::func::FuncOp>(
        dicp::LinalgExt::createScalarTo1DTensorPass());
  });
  m.def("add_npu_unroll_pipeline", [](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::func::FuncOp>(
        dicp::LinalgExt::createNPUUnroolPipelinePass());
  });
  m.def(
      "add_npu_vector_tile_tagging", [](mlir::PassManager &pm, int vectorTile) {
        pm.addPass(
            mlir::dicp::LinalgExt::createNPUVectorTileTaggingPass(vectorTile));
      });
  ADD_PASS_WRAPPER_0("add_npu_vector_tile_transform",
                     dicp::LinalgExt::createNPUVectorTileTransformPass);
  ADD_PASS_WRAPPER_0("add_de_linalgize",
                     dicp::LinalgExt::createDeLinalgizePass);
  ADD_PASS_WRAPPER_0("add_fuse_loop", dicp::LinalgExt::createFuseLoopPass);

  m.def("add_loop_unroll_stage", [](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::func::FuncOp>(
        dicp::LinalgExt::createLoopUnrollStagePass());
  });

  m.def("add_linalg_to_linked",
        [](mlir::PassManager &pm, bool globalKernel, bool namedOps) {
          pm.addPass(mlir::dicp::linked::createLinalgToLinkedPass(globalKernel,
                                                                  namedOps));
        });
  ADD_PASS_WRAPPER_0("add_linked_to_hivm",
                     dicp::linked::createLinkedToHIVMPass);
}

void init_triton_dicp_triton(py::module &&m) {
  m.doc() = "Python bindings to the Deeplink Triton backend";
  auto passes = m.def_submodule("passes");
  init_triton_dicp_triton_pass_triton_shared_ascend(
      passes.def_submodule("triton_shared_ascend"));
  init_triton_dicp_triton_pass_linked_npu(passes.def_submodule("linked_npu"));

  // load dialects
  m.def("load_dialects", [](MLIRContext &context) {
    llvm::errs() << ">>> [DEBUG] load_dialects ENTERED\n";
    llvm::errs() << ">>> [DEBUG] MLIRContext ptr: " << &context << "\n";
    DialectRegistry registry;
    registry.insert<tensor::TensorDialect, memref::MemRefDialect,
                    bufferization::BufferizationDialect, arith::ArithDialect,
                    cf::ControlFlowDialect, func::FuncDialect,
                    linalg::LinalgDialect, index::IndexDialect,
                    math::MathDialect, scf::SCFDialect, triton::TritonDialect,
                    affine::AffineDialect, ttx::TritonTilingExtDialect>();

    mlir::dicp::registerTransformDialectExtension(registry);
    mlir::linalg::registerTransformDialectExtension(registry);

    affine::registerValueBoundsOpInterfaceExternalModels(registry);
    tensor::registerInferTypeOpInterfaceExternalModels(registry);
    tensor::registerSubsetOpInterfaceExternalModels(registry);
    tensor::registerTilingInterfaceExternalModels(registry);
    linalg::registerAllDialectInterfaceImplementations(registry);
    scf::registerTransformDialectExtension(registry);
    func::registerAllExtensions(registry);

    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}