

#include "dicp/TritonToStructured/TritonToStructuredPass.h"

#include <cassert>
#include <cstdint>
#include <optional>

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "dicp/Utils/InterleaveOptimization.h"
#include "dicp/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "dicp/TritonToStructured/CannonicalizerConverter.h"
#include "dicp/TritonToStructured/MemOpConverter.h"
#include "dicp/TritonToStructured/PtrAnalysis.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "triton-to-structured"

using namespace mlir;
using namespace triton;

void TritonToStructuredPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                  linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                  tensor::TensorDialect, bufferization::BufferizationDialect,
                  memref::MemRefDialect, hivm::HIVMDialect,
                  annotation::AnnotationDialect>();
}

void TritonToStructuredPass::populateTritonToStructuredCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  // TODO enable this optimization after fixing the bisheng bug it causes in
  // current version
  // patterns.add<CannonicalizerConverter::CmpConverter>(patterns.getContext());
  patterns.add<CannonicalizerConverter::PromotePointerIterArgsPattern>(
      patterns.getContext());
  patterns.add<CannonicalizerConverter::SimplifyTensorIterArgsPattern>(
      patterns.getContext());
  // Add addptr splat->broadcast hoisting converter
  patterns.add<CannonicalizerConverter::AddPtrSplatConverter>(
      patterns.getContext());
  // Move loads before broadcasts when safe
  patterns.add<CannonicalizerConverter::LoadBroadcastConverter>(
      patterns.getContext());
}

void TritonToStructuredPass::populateTritonToStructuredPatterns(
    RewritePatternSet &patterns, bool optimizeDynamicOffset,
    bool enableMaskFallbackConversion) {
  patterns.add<MemOpConverter::LoadConverter>(patterns.getContext(),
                                              optimizeDynamicOffset,
                                              enableMaskFallbackConversion);
  patterns.add<MemOpConverter::StoreConverter>(patterns.getContext(),
                                               optimizeDynamicOffset,
                                               enableMaskFallbackConversion);
}

LogicalResult
TritonToStructuredPass::processSplatBinaryOperations(ModuleOp moduleOp) {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<CannonicalizerConverter::SplatCmpConverter>(
      patterns.getContext());
  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    moduleOp.emitWarning("Splat binary op processing failed");
    return failure();
  }
  return success();
}

void TritonToStructuredPass::runOnOperation() {
  auto moduleOp = getOperation();
  ConversionTarget target(getContext());
  RewritePatternSet canonicalizerPatterns(&getContext());

  this->populateTritonToStructuredCanonicalizationPatterns(
      canonicalizerPatterns);
  if (failed(
          applyPatternsGreedily(moduleOp, std::move(canonicalizerPatterns)))) {
    moduleOp.emitWarning("Canonicalize failed");
  }

  RewritePatternSet tritonToStructuredPatterns(&getContext());
  populateTritonToStructuredPatterns(tritonToStructuredPatterns,
                                     optimizeDynamicOffset,
                                     enableMaskFallbackConversion);

  if (failed(applyPatternsGreedily(moduleOp,
                                   std::move(tritonToStructuredPatterns)))) {
    LLVM_DEBUG({ moduleOp->emitRemark("PtrAnalysis: rewrite MemOp failed"); });
  }

  if (failed(processSplatBinaryOperations(moduleOp))) {
    moduleOp.emitWarning("Splat binary op processing failed");
  }

  PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, getOperation()))) {
    moduleOp->emitWarning("Canonicalize failed");
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToStructuredPass() {
  return std::make_unique<TritonToStructuredPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToStructuredPass(bool enableMaskFallbackConversion,
                                     bool optimizeDynamicOffset) {
  return std::make_unique<TritonToStructuredPass>(enableMaskFallbackConversion,
                                                  optimizeDynamicOffset);
}
