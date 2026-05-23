#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "vector-tiling-pipeline"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace mlir::dicp;
using namespace mlir::dicp::LinalgExt;

namespace mlir::dicp::LinalgExt {
#define GEN_PASS_DEF_VECTORTILINGPIPELINE
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace mlir::dicp::LinalgExt

namespace {

static void addVectorTilingCorePipeline(PassManager &pm, unsigned vectorTile,
                                        bool tileAllBlocks) {
  pm.addNestedPass<func::FuncOp>(createPrepareVectorTilingPass());

  StageTilingPlanOptions taggingOpts;
  taggingOpts.tiledMixVectorLoopNumber = vectorTile;
  taggingOpts.tileAllBlocks = tileAllBlocks;
  pm.addNestedPass<func::FuncOp>(createStageTilingPlanPass(taggingOpts));

  pm.addNestedPass<func::FuncOp>(createDuplicateStageProducersPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizeTileableOpsPass());

  // Apply the staged tiling decisions before loop-level cleanup.
  pm.addPass(createApplyTilingPlanPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createFuseLoopPass());
  pm.addPass(createDeLinalgizePass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<func::FuncOp>(createLowerForallToForPass());
  pm.addNestedPass<func::FuncOp>(createUnrollStageLoopsPass());
}

static void addBufferSchedulingPipeline(PassManager &pm, bool tileAllBlocks) {
  ShrinkBuffersOptions shrinkOpts;
  shrinkOpts.tileAllBlocks = tileAllBlocks;
  pm.addPass(createShrinkBuffersPass(shrinkOpts));

  // Canonicalize buffer shapes before scheduling input bundles.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createPrepareInputBundleSchedulingPass());
  pm.addNestedPass<func::FuncOp>(createScheduleInputBundlesPass());
}

static void restoreModuleFromSnapshot(ModuleOp moduleOp, ModuleOp snapshot) {
  // Keep the original ModuleOp handle alive for the surrounding pass manager
  // and restore its observable state from the pre-pipeline snapshot.
  moduleOp->setAttrs(snapshot->getAttrDictionary());
  moduleOp.getBodyRegion().takeBody(snapshot.getBodyRegion());
}

class VectorTilingPipelinePass
    : public mlir::dicp::LinalgExt::impl::VectorTilingPipelineBase<
          VectorTilingPipelinePass> {
public:
  using VectorTilingPipelineBase::VectorTilingPipelineBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    ModuleOp snapshot = moduleOp.clone();
    PassManager pm(&getContext(), moduleOp.getOperationName());

    addVectorTilingCorePipeline(pm, vectorTile, tileAllBlocks);
    addBufferSchedulingPipeline(pm, tileAllBlocks);

    LDBG("[Pass] running vector tiling pipeline");
    if (failed(runPipeline(pm, moduleOp))) {
      LDBG(
          "[Pass] vector tiling pipeline failed; restoring original module IR");
      restoreModuleFromSnapshot(moduleOp, snapshot);
      moduleOp.emitWarning()
          << "vector tiling pipeline failed; restored pre-pipeline IR and "
             "continuing compilation";
    }

    removeTmpStageAttributes(moduleOp);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::dicp::LinalgExt::createVectorTilingPipelinePass(
    const VectorTilingPipelineOptions &options) {
  return std::make_unique<VectorTilingPipelinePass>(options);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::dicp::LinalgExt::createVectorTilingPipelinePass(unsigned vectorTile) {
  VectorTilingPipelineOptions opts;
  opts.vectorTile = vectorTile;
  return std::make_unique<VectorTilingPipelinePass>(opts);
}
