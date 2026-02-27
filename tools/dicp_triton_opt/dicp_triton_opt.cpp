#include "dicp/Conversion/DiscreteMaskAccessConversion/Passes.h"
#include "dicp/Conversion/LinalgToLinked/Passes.h"
#include "dicp/Conversion/LinalgToNPU/Passes.h"
#include "dicp/Conversion/LinkedToHIVM/Passes.h"
#include "dicp/Conversion/TritonToLinalgNPU/MemRefCopyGatherToTensorInsert/Passes.h"
#include "dicp/Conversion/TritonToLinalgNPU/TritonToLinalgNPUCoversion/Passes.h"
#include "dicp/Conversion/TritonToUnstructure/Passes.h"
#include "dicp/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"
#include "dicp/Dialect/NPU/IR/NPUDialect.h"
#include "dicp/Dialect/TritonExt/Transforms/Passes.h"
#include "dicp/TransformOps/DicpTransformOps.h"

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
#include "mlir/Dialect/DLTI/TransformOps/DLTITransformOps.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/Transforms/AllInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
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
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

using namespace mlir;

inline void registerDICPDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();
  mlir::registerLinalgPasses();

  mlir::triton::registerDiscreteMaskAccessConversionPass();
  mlir::triton::registerTritonToUnstructurePass();
  mlir::triton::registerBubbleUpOperationPass();

  dicp::npu::registerLinalgToNPUPass();
  dicp::linked::registerLinalgToLinkedPass();
  dicp::trtion_ext::registerCanonicalizeTritonIRAscendPass();
  dicp::trtion_ext::registerCanonicalizeCmpiPass();
  dicp::linked::registerLinkedToHIVMPass();
  dicp::linked::registerTritonToLinalgNPUCoversionPass();
  dicp::linked::registerMemRefCopyGatherToTensorInsertPass();

  dicp::LinalgExt::registerLinalgIfToSelectPass();
  dicp::LinalgExt::registerLinalgGenericToSCFPass();
  dicp::LinalgExt::registerScalarTo1DTensorPass();
  dicp::LinalgExt::registerNormalizeSliceOpsPass();
  dicp::LinalgExt::registerNPUUnroolPipelinePass();
  dicp::LinalgExt::registerNPUVectorTileTaggingPass();
  dicp::LinalgExt::registerNPUVectorTileTransformPass();
  dicp::LinalgExt::registerDeLinalgizePass();
  dicp::LinalgExt::registerFuseLoopPass();
  dicp::LinalgExt::registerLoopUnrollStagePass();

  mlir::dicp::registerTransformDialectExtension(registry);
  mlir::linalg::registerTransformDialectExtension(registry);

  affine::registerValueBoundsOpInterfaceExternalModels(registry);
  tensor::registerInferTypeOpInterfaceExternalModels(registry);
  tensor::registerSubsetOpInterfaceExternalModels(registry);
  tensor::registerTilingInterfaceExternalModels(registry);
  linalg::registerAllDialectInterfaceImplementations(registry);


  scf::registerTransformDialectExtension(registry);

  registry.insert<bufferization::BufferizationDialect, dicp::npu::NPUDialect,
                  dicp::LinalgExt::LinalgExtDialect, arith::ArithDialect,
                  cf::ControlFlowDialect, func::FuncDialect, gpu::GPUDialect,
                  linalg::LinalgDialect, LLVM::LLVMDialect, math::MathDialect,
                  memref::MemRefDialect, pdl::PDLDialect, scf::SCFDialect,
                  tensor::TensorDialect, transform::TransformDialect,
                  vector::VectorDialect, ub::UBDialect, triton::TritonDialect,
                  affine::AffineDialect, ttx::TritonTilingExtDialect,
                  mlir::hivm::HIVMDialect>();
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerDICPDialects(registry);

  auto r = mlir::MlirOptMain(argc, argv, "dicp optimizer\n", registry);
  if (!r.succeeded()) {
    llvm::errs() << "MlirOptMain failed\n";
  }

  return mlir::asMainReturnCode(r);
}
