#include "dicp/Conversion/LinalgToNPU/Passes.h"
#include "dicp/Conversion/LinalgToLinked/Passes.h"
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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
// #include "mlir/Dialect/Affine/IR/AffineDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
// #include "mlir/Dialect/Vector/IR/VectorDialect.h"
// #include "mlir/Dialect/Vector/IR/VectorDialect.h"
#include "mlir/InitAllExtensions.h"


#include "mlir/InitAllExtensions.h"

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToEmitC/FuncToEmitC.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MPIToLLVM/MPIToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/XeVMToLLVM/XeVMToLLVM.h"
#include "mlir/Dialect/AMX/Transforms.h"
#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h"
#include "mlir/Dialect/ArmNeon/TransformOps/ArmNeonVectorTransformOps.h"
#include "mlir/Dialect/ArmSVE/TransformOps/ArmSVEVectorTransformOps.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/DLTI/TransformOps/DLTITransformOps.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/SparseTensor/TransformOps/SparseTensorTransformOps.h"
#include "mlir/Dialect/Tensor/Extensions/AllExtensions.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/DebugExtension/DebugExtension.h"
#include "mlir/Dialect/Transform/IRDLExtension/IRDLExtension.h"
#include "mlir/Dialect/Transform/LoopExtension/LoopExtension.h"
#include "mlir/Dialect/Transform/PDLExtension/PDLExtension.h"
#include "mlir/Dialect/Transform/TuneExtension/TuneExtension.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/XeVM/XeVMToLLVMIRTranslation.h"

using namespace mlir;
inline void registerDICPDialects(mlir::DialectRegistry &registry) {
  dicp::npu::registerLinalgToNPUPass();
  dicp::linked::registerLinalgToLinkedPass();
  registry.insert<bufferization::BufferizationDialect, dicp::npu::NPUDialect,
                  mlir::dicp::LinalgExt::LinalgExtDialect,
                  mlir::arith::ArithDialect, cf::ControlFlowDialect,
                  func::FuncDialect, gpu::GPUDialect, linalg::LinalgDialect,
                  index::IndexDialect, LLVM::LLVMDialect, math::MathDialect,
                  memref::MemRefDialect, pdl::PDLDialect, scf::SCFDialect,
                  tensor::TensorDialect, transform::TransformDialect,
                  vector::VectorDialect, ub::UBDialect, triton::TritonDialect>();

  // arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  // arith::registerBufferizableOpInterfaceExternalModels(registry);
  // arith::registerValueBoundsOpInterfaceExternalModels(registry);
  // bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
  //     registry);
  // builtin::registerCastOpInterfaceExternalModels(registry);
  // cf::registerBufferizableOpInterfaceExternalModels(registry);
  // cf::registerBufferDeallocationOpInterfaceExternalModels(registry);

  // linalg::registerBufferizableOpInterfaceExternalModels(registry);
  // linalg::registerValueBoundsOpInterfaceExternalModels(registry);
  // linalg::registerTilingInterfaceExternalModels(registry);
  // linalg::registerValueBoundsOpInterfaceExternalModels(registry);
  // memref::registerAllocationOpInterfaceExternalModels(registry);
  // memref::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  // memref::registerValueBoundsOpInterfaceExternalModels(registry);
  // memref::registerMemorySlotExternalModels(registry);
  // scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  // scf::registerBufferizableOpInterfaceExternalModels(registry);
  // scf::registerValueBoundsOpInterfaceExternalModels(registry);
  // tensor::registerBufferizableOpInterfaceExternalModels(registry);
  // tensor::registerFindPayloadReplacementOpInterfaceExternalModels(registry);
  // tensor::registerInferTypeOpInterfaceExternalModels(registry);
  // tensor::registerSubsetOpInterfaceExternalModels(registry);
  // tensor::registerTilingInterfaceExternalModels(registry);
  // tensor::registerValueBoundsOpInterfaceExternalModels(registry);
  // vector::registerBufferizableOpInterfaceExternalModels(registry);
  // vector::registerSubsetOpInterfaceExternalModels(registry);

  // affine::registerTransformDialectExtension(registry);
  // bufferization::registerTransformDialectExtension(registry);
  // func::registerTransformDialectExtension(registry);
  // linalg::registerTransformDialectExtension(registry);
  // memref::registerTransformDialectExtension(registry);
  // scf::registerTransformDialectExtension(registry);
  // tensor::registerTransformDialectExtension(registry);
  // vector::registerTransformDialectExtension(registry);

  // arith::registerConvertArithToLLVMInterface(registry);
  // registerConvertComplexToLLVMInterface(registry);
  // cf::registerConvertControlFlowToLLVMInterface(registry);

  // registerConvertFuncToLLVMInterface(registry);
  // index::registerConvertIndexToLLVMInterface(registry);
  // registerConvertMathToLLVMInterface(registry);
  // registerConvertMemRefToLLVMInterface(registry);
  // registerConvertNVVMToLLVMInterface(registry);
  // ub::registerConvertUBToLLVMInterface(registry);
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerDICPDialects(registry);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "dicp optimizer\n", registry));
}