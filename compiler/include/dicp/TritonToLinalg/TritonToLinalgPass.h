

#ifndef TRITON_ADAPTER_CONVERSION_TRITONTOLINALG_H
#define TRITON_ADAPTER_CONVERSION_TRITONTOLINALG_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define GEN_PASS_CLASSES
#include "dicp/TritonToLinalg/Passes.h.inc"

extern int nd2nzFlag;
extern bool compileOn91095Flag;
extern bool existDotFlag;

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createTritonToLinalgPass();

std::unique_ptr<OperationPass<ModuleOp>>
createTritonToLinalgPass(bool, bool, bool, bool, bool);

} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace triton;
const std::string globalKernelAttr = "global_kernel";
const std::string kernelMixModeName = "mix_mode";
const std::string kernelParallelModeName = "parallel_mode";

class TritonTypeConverter : public mlir::TypeConverter {
public:
  explicit TritonTypeConverter();
};

class TritonToLinalgPass : public TritonToLinalgBase<TritonToLinalgPass> {

  static auto constexpr LAUNCH_GRID_RANK = getMaxEnumValForProgramIDDim() + 1;
  static unsigned int constexpr TRITON_PROGRAM_INFO_ARG_COUNT =
      LAUNCH_GRID_RANK * 2;

private:
  // grid构造 num_programs 3维, program_id 3维
  // remember 'xxxOp' is usually a Pointer, so that we can change target memory
  // without giving a reference argument
  void addProgramInfo(triton::FuncOp func, bool globalKernel);

  void convertTTFunc(triton::FuncOp func, const bool existDot,
                     const bool existSIMTOp);

  LogicalResult convertMultipleBlockControlFlow(Operation *funcOp,
                                                OpBuilder &builder);
  // 处理嵌套的if/else
  scf::IfOp transformNestedIfElse(Operation &nestedBranch, OpBuilder &builder);

  void addDynamicLegal(ConversionTarget &target,
                       TritonTypeConverter &tritonTypeConverter);

  void
  populateTritonToLinalgCanonicalizationPatterns(RewritePatternSet &patterns);

  void populateTritonToLinalgConversionPatterns(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns,
                                                unsigned int launchGridRank);

  LogicalResult processDescriptorOperations(ModuleOp moduleOp);
  LogicalResult processPtrBroadcastOperations(ModuleOp moduleOp);
  LogicalResult processImplicitPermuteOperations(ModuleOp moduleOp);
  LogicalResult processLegalStrideOperations(ModuleOp moduleOp);

public:
  TritonToLinalgPass() = default;

  TritonToLinalgPass(bool globalKernel, bool namedOps, bool enableNd2nzOnVector,
                     bool enableSelectAnalysis, bool compileOn91095) {
    this->globalKernel = globalKernel;
    this->namedOps = namedOps;
    this->enableNd2nzOnVector = enableNd2nzOnVector;
    this->enableSelectAnalysis = enableSelectAnalysis;
    this->compileOn91095 = compileOn91095;
  };

  void getDependentDialects(DialectRegistry &registry) const override;

  void runOnOperation() override;
};

#endif // TRITON_ADAPTER_CONVERSION_TRITONTOLINALG_H
