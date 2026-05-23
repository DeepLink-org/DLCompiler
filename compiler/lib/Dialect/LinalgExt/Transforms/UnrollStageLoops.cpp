#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "unroll-stage-loops"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace mlir::dicp;
using namespace mlir::dicp::stage_attrs;

namespace mlir::dicp::LinalgExt {
#define GEN_PASS_DEF_UNROLLSTAGELOOPS
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace mlir::dicp::LinalgExt

namespace {

static constexpr int64_t kLargeLoopTripCountThreshold = 8;

static bool isLargeLoop(scf::ForOp forOp) {
  auto lb = getConstantIntValue(forOp.getLowerBound());
  auto ub = getConstantIntValue(forOp.getUpperBound());
  auto step = getConstantIntValue(forOp.getStep());
  if (!lb || !ub || !step || *step == 0)
    return false;

  int64_t tripCount = (*ub - *lb + *step - 1) / *step;
  return tripCount > kLargeLoopTripCountThreshold;
}

//===----------------------------------------------------------------------===//
// Pass: UnrollStageLoops
//===----------------------------------------------------------------------===//

struct UnrollStageLoopsPass
    : public mlir::dicp::LinalgExt::impl::UnrollStageLoopsBase<
          UnrollStageLoopsPass> {
  using UnrollStageLoopsBase::UnrollStageLoopsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    LDBG("[Pass] run on @" << func.getName());
    SmallVector<scf::ForOp> candidateLoops;
    func.walk([&](scf::ForOp forOp) {
      if (forOp->hasAttr("ExtractedLoadOrStore"))
        return WalkResult::advance();
      if (forOp->hasAttr(kNPUStageAttrName) && !isLargeLoop(forOp))
        candidateLoops.push_back(forOp);
      return WalkResult::advance();
    });

    if (candidateLoops.empty()) {
      LDBG("[Pass] no staged scf.for loops to unroll");
      return;
    }

    LDBG("[Pass] collected " << candidateLoops.size() << " candidate loops");

    for (auto [index, forOp] : llvm::enumerate(candidateLoops)) {
      LDBG("[Unroll] loop #" << index << ": " << forOp);
      if (failed(loopUnrollFull(forOp)))
        return forOp.emitError("failed to fully unroll loop ") << index,
               signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::dicp::LinalgExt::createUnrollStageLoopsPass() {
  return std::make_unique<UnrollStageLoopsPass>();
}
