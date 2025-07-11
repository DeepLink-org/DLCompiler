#include "dicp/Conversion/LinalgToNPU/LinalgToNPU.h"
#include "dicp/Dialect/NPU/IR/NPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

// 必要头文件
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Location.h"
// #include "mlir/Dialect/Linalg/IR/LinalgAttributes.h" // 必须包含这个头文件
#include "mlir/Dialect/Linalg/IR/Linalg.h"
// #include "mlir/Dialect/Utils/DialectUtilsEnums.h.inc"


#define DEBUG_TYPE "linalg-to-npu"
#include "llvm/Support/Debug.h"
#include <iostream>

using namespace mlir;
using namespace dicp;

#define GEN_PASS_CLASSES
#include "dicp/Conversion/LinalgToNPU/Passes.h.inc"

namespace {

class LinalgToNPUPass : public LinalgToNPUBase<LinalgToNPUPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        npu::NPUDialect, linalg::LinalgDialect, affine::AffineDialect,
        scf::SCFDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    std::cout << "Running Linalg to NPU conversion pass...\n";
    npu::populateLinalgToNPUConversionPatterns(patterns);
    // patterns.add<ConvertLinalgGenericToArith>(context);
    std::cout << "Populating Linalg to NPU conversion patterns...\n";

    if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
    std::cout << "Linalg to NPU conversion pass completed successfully.\n";
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> npu::createLinalgToNPUPass() {
  return std::make_unique<LinalgToNPUPass>();
}
