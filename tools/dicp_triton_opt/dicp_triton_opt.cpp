#include "dicp/Conversion/LinalgToLinked/Passes.h"
#include "dicp/Conversion/LinalgToNPU/Passes.h"
#include "dicp/Conversion/LinkedToHIVM/Passes.h"
#include "dicp/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "dicp/Dialect/NPU/IR/NPUDialect.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Linalg/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

using namespace mlir;
inline void registerDICPDialects(mlir::DialectRegistry &registry) {
  dicp::npu::registerLinalgToNPUPass();
  dicp::linked::registerLinalgToLinkedPass();
  dicp::linked::registerBoolTritonPtrPromotionPass();
  dicp::linked::registerLinkedToHIVMPass();
  registry.insert<
      bufferization::BufferizationDialect, dicp::npu::NPUDialect,
      mlir::dicp::LinalgExt::LinalgExtDialect, mlir::arith::ArithDialect,
      cf::ControlFlowDialect, func::FuncDialect, gpu::GPUDialect,
      linalg::LinalgDialect, index::IndexDialect, LLVM::LLVMDialect,
      math::MathDialect, memref::MemRefDialect, pdl::PDLDialect,
      scf::SCFDialect, tensor::TensorDialect, transform::TransformDialect,
      vector::VectorDialect, ub::UBDialect, triton::TritonDialect>();

  arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  arith::registerValueBoundsOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  builtin::registerCastOpInterfaceExternalModels(registry);
  cf::registerBufferizableOpInterfaceExternalModels(registry);
  cf::registerBufferDeallocationOpInterfaceExternalModels(registry);

  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerValueBoundsOpInterfaceExternalModels(registry);
  linalg::registerTilingInterfaceExternalModels(registry);
  linalg::registerValueBoundsOpInterfaceExternalModels(registry);
  memref::registerAllocationOpInterfaceExternalModels(registry);
  memref::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  memref::registerValueBoundsOpInterfaceExternalModels(registry);
  memref::registerMemorySlotExternalModels(registry);
  scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerValueBoundsOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerFindPayloadReplacementOpInterfaceExternalModels(registry);
  tensor::registerInferTypeOpInterfaceExternalModels(registry);
  tensor::registerSubsetOpInterfaceExternalModels(registry);
  tensor::registerTilingInterfaceExternalModels(registry);
  tensor::registerValueBoundsOpInterfaceExternalModels(registry);
  vector::registerBufferizableOpInterfaceExternalModels(registry);
  vector::registerSubsetOpInterfaceExternalModels(registry);

  affine::registerTransformDialectExtension(registry);
  bufferization::registerTransformDialectExtension(registry);
  func::registerTransformDialectExtension(registry);
  linalg::registerTransformDialectExtension(registry);
  memref::registerTransformDialectExtension(registry);
  scf::registerTransformDialectExtension(registry);
  tensor::registerTransformDialectExtension(registry);
  vector::registerTransformDialectExtension(registry);

  arith::registerConvertArithToLLVMInterface(registry);
  registerConvertComplexToLLVMInterface(registry);
  cf::registerConvertControlFlowToLLVMInterface(registry);

  registerConvertFuncToLLVMInterface(registry);
  index::registerConvertIndexToLLVMInterface(registry);
  registerConvertMathToLLVMInterface(registry);
  registerConvertMemRefToLLVMInterface(registry);
  registerConvertNVVMToLLVMInterface(registry);
  ub::registerConvertUBToLLVMInterface(registry);
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
