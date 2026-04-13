#ifndef LinalgEXT_DIALECT_Linalg_TRANSFORMS_PASSES_H_
#define LinalgEXT_DIALECT_Linalg_TRANSFORMS_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class TypeConverter;
class ConversionTarget;
namespace func {
class FuncOp;
}
} // namespace mlir

namespace mlir::dicp::LinalgExt {

#define GEN_PASS_DECL
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"

std::unique_ptr<OperationPass<mlir::ModuleOp>> createLinalgIfToSelectPass();

std::unique_ptr<OperationPass<mlir::ModuleOp>> createLinalgGenericToSCFPass();

std::unique_ptr<OperationPass<mlir::func::FuncOp>> createScalarTo1DTensorPass();

std::unique_ptr<OperationPass<mlir::func::FuncOp>>
createNormalizeSliceOpsPass();
std::unique_ptr<OperationPass<mlir::func::FuncOp>>
createVectorizeParallelLoopPass();

std::unique_ptr<OperationPass<mlir::func::FuncOp>>
createPipelineLoopUnrollPass();

std::unique_ptr<OperationPass<mlir::func::FuncOp>>
createCanonicalizeTileableOpsPass();

std::unique_ptr<OperationPass<mlir::func::FuncOp>>
createStageTilingPlanPass(const StageTilingPlanOptions &options = {});
std::unique_ptr<OperationPass<mlir::func::FuncOp>>
createStageTilingPlanPass(unsigned vectorTile);

std::unique_ptr<OperationPass<mlir::func::FuncOp>>
createDuplicateStageProducersPass();

std::unique_ptr<OperationPass<mlir::ModuleOp>> createApplyTilingPlanPass();

std::unique_ptr<OperationPass<mlir::ModuleOp>> createDeLinalgizePass();

std::unique_ptr<OperationPass<mlir::ModuleOp>> createFuseLoopPass();
std::unique_ptr<OperationPass<mlir::func::FuncOp>> createLowerForallToForPass();
std::unique_ptr<OperationPass<mlir::func::FuncOp>> createUnrollStageLoopsPass();

std::unique_ptr<OperationPass<mlir::ModuleOp>> createShrinkBuffersPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createShrinkBuffersPass(const ShrinkBuffersOptions &options);

std::unique_ptr<OperationPass<mlir::func::FuncOp>>
createPrepareVectorTilingPass();

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createVerifyNoLocalBufferCopyPass();

std::unique_ptr<OperationPass<mlir::func::FuncOp>>
createScheduleInputBundlesPass();

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createPrepareInputBundleSchedulingPass();

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createVectorTilingPipelinePass(const VectorTilingPipelineOptions &options = {});
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createVectorTilingPipelinePass(unsigned vectorTile);

#define GEN_PASS_REGISTRATION
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"

} // namespace mlir::dicp::LinalgExt

#endif
