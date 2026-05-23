//===- VerifyNoLocalBufferCopy.cpp - Verify no copy into local alloc ------===//
//
// This verification pass walks all `memref.copy` and
// `bufferization.materialize_in_destination` operations and checks whether
// the destination operand traces back through view-like adapters to a
// `memref.alloc` or `memref.alloca`.
//
// Such writes indicate that the tiling-and-shrink pipeline left behind an
// intermediate local buffer that should have been eliminated.
//
//===----------------------------------------------------------------------===//

#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"
#include "dicp/Utils/Utils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "verify-no-local-buffer-copy"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;

namespace mlir::dicp::LinalgExt {
#define GEN_PASS_DEF_VERIFYNOLOCALBUFFERCOPY
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace mlir::dicp::LinalgExt

namespace {

static bool diagnoseLocalBufferWrite(Operation *op, Value destRoot,
                                     StringRef opName, StringRef allocMessage,
                                     StringRef allocaMessage) {
  if (auto allocOp = destRoot.getDefiningOp<memref::AllocOp>()) {
    op->emitError(allocMessage).attachNote(allocOp.getLoc())
        << "allocation defined here: " << *allocOp;
    return true;
  }
  if (auto allocaOp = destRoot.getDefiningOp<memref::AllocaOp>()) {
    op->emitError(allocaMessage).attachNote(allocaOp.getLoc())
        << "allocation defined here: " << *allocaOp;
    return true;
  }

  LDBG("[Verify] accepted " << opName << " with destination root " << destRoot);
  return false;
}

struct VerifyNoLocalBufferCopyPass
    : public mlir::dicp::LinalgExt::impl::VerifyNoLocalBufferCopyBase<
          VerifyNoLocalBufferCopyPass> {
  using VerifyNoLocalBufferCopyBase::VerifyNoLocalBufferCopyBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    bool hasViolation = false;

    LDBG("[Pass] verify copies into local buffers");

    // Reject copies whose destination traces back to a local allocation.
    moduleOp->walk([&](memref::CopyOp copyOp) {
      Value root = mlir::dicp::traceToSourceRoot(copyOp.getTarget());
      LDBG("[Verify] memref.copy destination root: " << root);
      hasViolation |= diagnoseLocalBufferWrite(
          copyOp, root, "memref.copy",
          "memref.copy writes into a locally allocated buffer "
          "(memref.alloc); this should have been eliminated by the "
          "shrink-buffers pass",
          "memref.copy writes into a stack-allocated buffer "
          "(memref.alloca); this should have been eliminated");
    });

    // Reject materialization into local memref destinations for the same
    // reason: the shrink-buffers pipeline should have eliminated them.
    moduleOp->walk([&](bufferization::MaterializeInDestinationOp matOp) {
      Value dest = matOp.getDest();
      if (!isa<MemRefType>(dest.getType()))
        return; // Tensor destinations are not relevant here.

      Value root = mlir::dicp::traceToSourceRoot(dest);
      LDBG("[Verify] materialize_in_destination destination root: " << root);
      hasViolation |= diagnoseLocalBufferWrite(
          matOp, root, "materialize_in_destination",
          "materialize_in_destination writes into a locally allocated buffer "
          "(memref.alloc)",
          "materialize_in_destination writes into a stack-allocated buffer "
          "(memref.alloca)");
    });

    if (hasViolation) {
      LDBG("[Pass] verification failed");
      signalPassFailure();
      return;
    }

    LDBG("[Pass] verification passed");
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::dicp::LinalgExt::createVerifyNoLocalBufferCopyPass() {
  return std::make_unique<VerifyNoLocalBufferCopyPass>();
}
