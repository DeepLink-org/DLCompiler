#include "dicp/Conversion/LinalgToNPU/LinalgToNPU.h"
#include "dicp/Dialect/NPU/IR/NPUDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "linalg-to-npu"
#include "dicp/Conversion/LinalgToNPU/ConversionPatterns.hpp"

using namespace mlir;
using namespace dicp;

#define GEN_PASS_CLASSES
#include "dicp/Conversion/LinalgToNPU/Passes.h.inc"

namespace {

class LinalgToNPUPass : public LinalgToNPUBase<LinalgToNPUPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                npu::NPUDialect, linalg::LinalgDialect, affine::AffineDialect,
                scf::SCFDialect, tensor::TensorDialect,
                bufferization::BufferizationDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    // OpBuilder builder(moduleOp.getBodyRegion());
    // builder.setInsertionPointToStart(builder.getBlock());
    // builder.create<npu::CreateTpipOp>(builder.getUnknownLoc());

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<npu::NPUDialect>();
    // target.addDynamicallyLegalOp<arith::AddFOp>([](arith::AddFOp op) {
    //     return !isa<Fload16>(op.getResult().getType());
    //   });
    // target.addIllegalOp<arith::AddFOp>();

    npu::populateLinalgToNPUConversionPatterns(patterns);
    // patterns.add<CopyConverter>(patterns.getContext());

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> npu::createLinalgToNPUPass() {
  return std::make_unique<LinalgToNPUPass>();
}
