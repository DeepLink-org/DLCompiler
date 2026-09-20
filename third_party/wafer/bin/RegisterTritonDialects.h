#pragma once
#include "Address/Dialect/IR/AddressDialect.h"
#include "Address/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/AllInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"

#include "magic-kernel/Conversion/TLEToMK/Passes.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "third_party/tle/include/tle-dsa/Conversion/DsaToCore/DsaToCore.h"
#include "third_party/tle/include/tle-dsa/Dialect/IR/DsaDialect.h"
#include "triton-shared/Conversion/ConvertTritonPtr/Passes.h"
#include "triton-shared/Conversion/ReconcilePtrCasts/Passes.h"
#include "triton-shared/Conversion/StructuredToMemref/Passes.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonPtrToMemref/Passes.h"
#include "triton-shared/Conversion/TritonToCoreDialects/Passes.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToStructured/Passes.h"
#include "triton-shared/Conversion/TritonToUnstructured/Passes.h"
#include "triton-shared/Conversion/UnstructuredToMemref/Passes.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "wafer/Conversion/LinalgFusion/Passes.h"
#include "wafer/Conversion/LinalgTiling/Passes.h"
#include "wafer/Dialect/IR/WaferDialect.h"

#include "magic-kernel/Conversion/CoreDialectsToMK/Passes.h"
#include "magic-kernel/Conversion/LegalizeTensorFormLoops/Passes.h"
#include "magic-kernel/Conversion/LinalgToMK/Passes.h"
#include "magic-kernel/Transforms/Passes.h"
#include "magic-kernel/Conversion/MKPipeline/Passes.h"
#include "wafer/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "wafer/Conversion/AllocateSharedMemory/Passes.h"
#include "wafer/Conversion/ExportKernelSymbols/Passes.h"
#include "wafer/Conversion/MKToWafer/Passes.h"
#include "wafer/Conversion/WaferMemrefToLLVM/Passes.h"
#include "wafer/Conversion/WaferToLLVM/KernelArgBufferPass.h"
#include "wafer/Conversion/WaferToLLVM/Passes.h"

#include "magic-kernel/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

inline void registerTritonDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();
    mlir::triton::registerTritonPasses();
  mlir::registerLinalgPasses();
  mlir::dsa::registerDsaMemoryToCorePass();
  mlir::triton::registerTLEToMKPass();
    mlir::triton::nvidia_gpu::registerTritonNvidiaGPUPasses();
  mlir::triton::registerTritonToLinalgPass();
  mlir::triton::registerTritonToStructuredPass();
  mlir::triton::registerTritonToUnstructuredPass();
  mlir::triton::registerTritonArithToLinalgPasses();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::registerStructuredToMemrefPasses();
  mlir::triton::registerUnstructuredToMemref();
  mlir::triton::registerTritonPtrToMemref();
  mlir::triton::registerTritonToCoreDialectsPass();
  mlir::triton::registerReconcilePtrCasts();
  mlir::triton::gpu::registerAllocateSharedMemoryPass();
  mlir::triton::gpu::registerTritonGPUAllocateWarpGroups();
  mlir::triton::gpu::registerTritonGPUGlobalScratchAllocationPass();
  mlir::registerLLVMDIScope();

  // Core dialects to MK layer conversion passes
  mlir::triton::registerWaferMemrefToLLVMPass();
  mlir::triton::registerLinalgToMKPass();
  mlir::triton::registerMKTransformsPasses();
  mlir::triton::registerMKPipelinePasses();
  mlir::triton::registerWaferTransformsPasses();
  mlir::triton::registerCoreDialectsToMKPass();
  mlir::triton::registerLegalizeTensorFormLoopsPass();
  mlir::addr::registerAddrToLLVMPass();
  mlir::triton::registerLinalgTilingPass();
  mlir::triton::registerLinalgFusionPass();

  // Wafer specific conversion passes
  mlir::triton::registerMKToWaferPass();
  mlir::triton::alloc::registerAllocateSharedMemoryPass();
  mlir::triton::registerWaferToLLVMPass();
  mlir::triton::registerExportKernelSymbols();
  mlir::triton::registerKernelArgBufferPass();

  // Register LLVM 22's standard external models and Wafer's custom model.
  mlir::registerAllExtensions(registry);
  mlir::linalg::registerAllDialectInterfaceImplementations(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
  mlir::mk::registerBufferizableOpInterfaceExternalModels(registry);

  registry.insert<
      mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
      mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
      mlir::triton::gpu::TritonGPUDialect, mlir::math::MathDialect,
      mlir::arith::ArithDialect, mlir::scf::SCFDialect, mlir::gpu::GPUDialect,
      mlir::LLVM::LLVMDialect,
      mlir::ttx::TritonTilingExtDialect, mlir::tts::TritonStructuredDialect,
      mlir::linalg::LinalgDialect, mlir::func::FuncDialect,
      mlir::tensor::TensorDialect, mlir::memref::MemRefDialect,
      mlir::affine::AffineDialect, mlir::bufferization::BufferizationDialect,
      mlir::mk::MagicKernelDialect, mlir::wafer::WaferDialect,
      mlir::addr::AddressDialect, mlir::dsa::DsaDialect>();
}
