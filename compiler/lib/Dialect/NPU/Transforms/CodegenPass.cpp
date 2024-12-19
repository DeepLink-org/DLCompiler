
#include "dicp/DIalect/NPU/Transforms/Passes.h"
#include "dicp/Dialect/NPU/IR/NPUDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace dicp;

namespace mlir::dicp::npu {

class CodegenPass : public CodegenBase<CodegenPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                npu::NPUDialect, linalg::LinalgDialect, affine::AffineDialect,
                scf::SCFDialect, tensor::TensorDialect,
                bufferization::BufferizationDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    std::string filename = this->model_file;
    if (filename.empty()) {
      llvm_unreachable("codegen filename is empty");
    }

    auto moduleOp = getOperation();
    NPUCodegen npu_codegen;
    npu_codegen.init(moduleOp, filename);
    npu_codegen.run();
    npu_codegen.store();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> npu::createCodegenPass() {
  return std::make_unique<CodegenPass>();
}
