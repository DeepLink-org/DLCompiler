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
#ifndef TRITON_ADAPTER_CONVERSION_TRITONTOSTRUCTURED_H
#define TRITON_ADAPTER_CONVERSION_TRITONTOSTRUCTURED_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define GEN_PASS_CLASSES
#include "incubated/Conversion/TritonToStructuredIncubated/Passes.h.inc"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createTritonToStructuredIncubatedPass();

std::unique_ptr<OperationPass<ModuleOp>>
createTritonToStructuredIncubatedPass(bool, bool, bool);

} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace triton;

class TritonToStructuredIncubatedPass
    : public TritonToStructuredIncubatedBase<TritonToStructuredIncubatedPass> {
public:
  TritonToStructuredIncubatedPass() = default;

  TritonToStructuredIncubatedPass(bool enableMaskFallbackConversion,
                                  bool optimizeDynamicOffset,
                                  bool compileOn91095) {
    this->enableMaskFallbackConversion = enableMaskFallbackConversion;
    this->optimizeDynamicOffset = optimizeDynamicOffset;
    this->compileOn91095 = compileOn91095;
  };
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;

private:
  void populateTritonToStructuredCanonicalizationPatterns(
      RewritePatternSet &patterns);

  void populateTritonToStructuredPatterns(RewritePatternSet &patterns,
                                          bool optimizeDynamicOffset,
                                          bool enableMaskFallbackConversion,
                                          bool compileOn91095);
};

#endif // TRITON_ADAPTER_CONVERSION_TRITONTOSTRUCTURED_H
