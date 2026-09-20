/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "incubated/Conversion/TritonToStructuredIncubated/TritonToStructuredIncubatedPass.h"

#include <cassert>
#include <cstdint>
#include <optional>

#include "incubated/Conversion/UtilsIncubated/InterleaveOptimization.h"
#include "incubated/Conversion/UtilsIncubated/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#if __has_include("bishengir/Dialect/HIVM/IR/HIVM.h")
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#endif
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#if __has_include("bishengir/Dialect/Annotation/IR/Annotation.h")
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#endif
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

#include "incubated/Conversion/TritonToStructuredIncubated/CannonicalizerConverter.h"
#include "incubated/Conversion/TritonToStructuredIncubated/MemOpConverter.h"
#include "incubated/Conversion/TritonToStructuredIncubated/PtrAnalysis.h"
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

void TritonToStructuredIncubatedPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                  linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                  tensor::TensorDialect, bufferization::BufferizationDialect,
                  memref::MemRefDialect, hivm::HIVMDialect,
                  annotation::AnnotationDialect>();
}

void TritonToStructuredIncubatedPass::
    populateTritonToStructuredCanonicalizationPatterns(
        RewritePatternSet &patterns) {
  // TODO enable this optimization after fixing the bisheng bug it causes in
  // current version
  // patterns.add<CannonicalizerConverter::CmpConverter>(patterns.getContext());
  patterns.add<CannonicalizerConverter::PromotePointerIterArgsPattern>(
      patterns.getContext());
}

void TritonToStructuredIncubatedPass::populateTritonToStructuredPatterns(
    RewritePatternSet &patterns, bool optimizeDynamicOffset,
    bool enableMaskFallbackConversion, bool compileOn91095) {
  patterns.add<MemOpConverter::LoadConverter>(
      patterns.getContext(), optimizeDynamicOffset,
      enableMaskFallbackConversion, compileOn91095);
  patterns.add<MemOpConverter::StoreConverter>(
      patterns.getContext(), optimizeDynamicOffset,
      enableMaskFallbackConversion, false);
}

void TritonToStructuredIncubatedPass::runOnOperation() {
  auto moduleOp = getOperation();
  ConversionTarget target(getContext());
  RewritePatternSet canonicalizerPatterns(&getContext());

  this->populateTritonToStructuredCanonicalizationPatterns(
      canonicalizerPatterns);
  if (failed(applyPatternsAndFoldGreedily(moduleOp,
                                          std::move(canonicalizerPatterns)))) {
    moduleOp.emitWarning("Canonicalize failed");
  }

  RewritePatternSet tritonToStructuredPatterns(&getContext());
  populateTritonToStructuredPatterns(
      tritonToStructuredPatterns, optimizeDynamicOffset,
      enableMaskFallbackConversion, compileOn91095);

  if (failed(applyPatternsAndFoldGreedily(
          moduleOp, std::move(tritonToStructuredPatterns)))) {
    LLVM_DEBUG({ moduleOp->emitRemark("PtrAnalysis: rewrite MemOp failed"); });
  }

  PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, getOperation()))) {
    moduleOp->emitWarning("Canonicalize failed");
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToStructuredIncubatedPass() {
  return std::make_unique<TritonToStructuredIncubatedPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToStructuredIncubatedPass(bool enableMaskFallbackConversion,
                                              bool optimizeDynamicOffset,
                                              bool compileOn91095) {
  return std::make_unique<TritonToStructuredIncubatedPass>(
      enableMaskFallbackConversion, optimizeDynamicOffset, compileOn91095);
}
