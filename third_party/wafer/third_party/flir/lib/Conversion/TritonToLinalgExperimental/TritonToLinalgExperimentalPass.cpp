//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//
#include "flagtree/Common/UnifiedHardware.h"

#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir-ext/Dialect/MathExt/IR/MathExt.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Conversion/MemrefCopyToDMA_FlagTree/MemrefCopyToDMAFlagTree.h"
#include "triton-shared/Conversion/NoBufferize_FlagTree/NoBufferizeFlagTree.h"
#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"
#include "triton-shared/Conversion/ReconcilePtrCasts/ReconcilePtrCasts.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToPtr.h"
#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton-shared/Conversion/TritonToUnstructured/TritonToUnstructured.h"
#include "triton-shared/Conversion/UnstructuredToMemref/UnstructuredToMemref.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace triton;
//using namespace mlir::mathext;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

namespace {

class TritonToLinalgExperimentalPass
    : public TritonToLinalgExperimentalBase<TritonToLinalgExperimentalPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    mathext::MathExtDialect,
                    linalg::LinalgDialect, affine::AffineDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    bufferization::BufferizationDialect, memref::MemRefDialect,
                    ttx::TritonTilingExtDialect, tts::TritonStructuredDialect,
                    tptr::TPtrDialect, ptr::PtrDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createWaferTritonToStructuredPass());

    // Erase dead code and fold constants created during lowering
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    pm.addPass(createTritonToUnstructuredPass());
    pm.addPass(createTritonArithToLinalgPass(/*tensorPtrToLinalg=*/true));

    // TODO: structured-to-memref converts the loop iter-args to memref, while
    // triton-to-ptr converts the loop iter-args to ptr. These two passes might
    // end up conflicting with each other in cases where we have a mixed of
    // structured and unstructured accesses. Fortunately, the structured ops do
    // not need to use the loop iter-args at all (see code in PtrAnalysis.cpp),
    // so if we run remove-dead-values after structured-to-memref, the memref
    // iter-args that are used in structured loads and stores should be removed.
    // Running this now may be too invasive and cause many IR changes, so
    // leave as a TODO for now.
    pm.addPass(createStructuredToMemrefPass());

    // Pass selection is controlled by unified hardware configuration.
    auto hardwareManager = mlir::flagtree::createUnifiedHardwareManager();
    auto dmaTag = hardwareManager -> getDMATag();
    if (dmaTag) pm.addPass(createMemrefCopyToDMAFlagTreePass());

    pm.addPass(createUnstructuredToMemrefPass());
    pm.addPass(createTritonPtrToMemrefPass());
    pm.addPass(createTritonToPtrPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createReconcilePtrCastsPass());

    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    auto sharedTag = hardwareManager -> getSharedMemoryTag();
    if (sharedTag) pm.addPass(createNoBufferizeFlagTreePass());

    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToLinalgExperimentalPass() {
  return std::make_unique<TritonToLinalgExperimentalPass>();
}
