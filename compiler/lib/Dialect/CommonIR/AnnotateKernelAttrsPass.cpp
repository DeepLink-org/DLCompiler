#include "dicp/Dialect/CommonIR/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace mlir::dicp::CommonIR {
#define GEN_PASS_DEF_ANNOTATEKERNELATTRSPASS
#include "dicp/Dialect/CommonIR/Passes.h.inc"
} // namespace mlir::dicp::CommonIR

#define DEBUG_TYPE "annotate-kernel-attrs-pass"

namespace {

static constexpr llvm::StringRef kGlobalKernel = "global_kernel";
static constexpr llvm::StringRef kMixMode = "mix_mode";
static constexpr llvm::StringRef kParallelMode = "parallel_mode";

struct AnnotateKernelAttrsPass
    : public PassWrapper<AnnotateKernelAttrsPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "annotate-kernel-attrs"; }
  StringRef getDescription() const final {
    return "Annotate kernel func.func with mix_mode / parallel_mode / "
           "global_kernel so downstream lowering inserts the workspace and "
           "syncBlockLock stub arguments.";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    for (auto func : module.getOps<func::FuncOp>()) {
      if (func->hasAttr(kGlobalKernel))
        continue;
      if (func.isDeclaration())
        continue;

      auto *ctx = func.getContext();
      func->setAttr(kGlobalKernel, StringAttr::get(ctx, "local"));
      func->setAttr(kMixMode, StringAttr::get(ctx, "aiv"));
      func->setAttr(kParallelMode, StringAttr::get(ctx, "simd"));
    }
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnnotateKernelAttrsPass)
};

} // namespace

namespace mlir::dicp::CommonIR {
std::unique_ptr<OperationPass<ModuleOp>> createAnnotateKernelAttrsPass() {
  return std::make_unique<AnnotateKernelAttrsPass>();
}
} // namespace mlir::dicp::CommonIR
