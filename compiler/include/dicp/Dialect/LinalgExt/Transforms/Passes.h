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
createNPUUnroolPipelinePass();

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createNPUVectorTileTaggingPass(const NPUVectorTileTaggingOptions &options = {});
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createNPUVectorTileTaggingPass(unsigned vectorTile);

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createNPUVectorTileTransformPass();

std::unique_ptr<OperationPass<mlir::ModuleOp>> createDeLinalgizePass();

std::unique_ptr<OperationPass<mlir::ModuleOp>> createFuseLoopPass();
std::unique_ptr<OperationPass<mlir::func::FuncOp>> createLoopUnrollStagePass();

std::unique_ptr<OperationPass<mlir::func::FuncOp>> createShrinkBuffersPass();

#define GEN_PASS_REGISTRATION
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"

} // namespace mlir::dicp::LinalgExt

#endif
