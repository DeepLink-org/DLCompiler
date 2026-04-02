#include "dicp/Conversion/LinalgToLinked/Passes.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "debug-cpu-verify"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "dicp/Conversion/LinalgToLinked/Passes.h.inc"

namespace {

/// Returns true if the operation belongs to an external (non-MLIR-upstream)
/// dialect that should have been lowered away before CPU verification.
static bool isExternalDialectOp(Operation *op) {
  Dialect *dialect = op->getDialect();
  if (!dialect)
    return false;
  return isa<triton::TritonDialect, hivm::HIVMDialect,
             ttx::TritonTilingExtDialect, annotation::AnnotationDialect>(
      dialect);
}

/// Returns true if the operation is a hivm.hir.sync_block_* op that should be
/// removed before CPU verification.
static bool isSyncBlockOp(Operation *op) {
  return isa<hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp>(op);
}

class DebugCPUVerifyPass : public DebugCPUVerifyBase<DebugCPUVerifyPass> {
public:
  void runOnOperation() override {
    // First pass: remove all hivm.hir.sync_block_* operations
    SmallVector<Operation *> syncBlockOpsToErase;
    getOperation()->walk([&](Operation *op) {
      if (isSyncBlockOp(op))
        syncBlockOpsToErase.push_back(op);
    });
    for (Operation *op : syncBlockOpsToErase)
      op->erase();

    // Second pass: verify no external dialect operations remain
    bool failed = false;
    getOperation()->walk([&](Operation *op) {
      if (!isExternalDialectOp(op))
        return;
      op->emitError() << "external dialect op '" << op->getName()
                      << "' must be lowered before CPU verification";
      failed = true;
    });

    if (failed)
      return signalPassFailure();

    LLVM_DEBUG(llvm::dbgs() << "[debug-cpu-verify] PASSED — no external "
                               "dialect operations found\n");
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::dicp::linked::createDebugCPUVerifyPass() {
  return std::make_unique<DebugCPUVerifyPass>();
}
