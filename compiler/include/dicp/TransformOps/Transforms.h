#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"

#ifndef BISHENGIR_TRANSFORMS_TRANSFORMS_H
#define BISHENGIR_TRANSFORMS_TRANSFORMS_H

#define DICP_STAGE_PREFIX "dicp.stage."
namespace mlir::dicp {

static const llvm::StringLiteral kDicpStagePrefix = DICP_STAGE_PREFIX;
static const llvm::StringLiteral kStageOpToTileAttr =
    DICP_STAGE_PREFIX "op_to_tile.stage_{0}_sub_{1}_u{2}_";
static const llvm::StringLiteral kStageProducerToFuseAttr =
    DICP_STAGE_PREFIX "producer_to_fuse.stage_{0}_sub_{1}_u{2}";
static const llvm::StringLiteral kStageProducerAllocToFuseAttr =
    DICP_STAGE_PREFIX "alloc_producer";
static const llvm::StringLiteral kCrossTillUnitAttr =
    DICP_STAGE_PREFIX "till_unit_has_cross_user";
static const llvm::StringLiteral kHadFusedAttr =
    DICP_STAGE_PREFIX "op_had_fused";

static const llvm::StringLiteral kNPUStageAttrName = "dicp.npu.stage";
static const llvm::StringLiteral kOriginalOpNameAttr = "dicp.original_op_name";

void unionProducerUsers(mlir::RewriterBase &rewriter, mlir::Diagnostic &diag,
                        mlir::Operation *producerOp,
                        mlir::Operation *containingOp);
std::tuple<llvm::SmallVector<mlir::Operation *>, mlir::Operation *>

tileAndFuseAllSubsetOps(mlir::RewriterBase &rewriter, mlir::Diagnostic &diag,
                        mlir::Operation *producerOp,
                        mlir::Operation *containingOp, bool duplicateProducer);

llvm::SmallVector<mlir::Operation *>
tileAndFuseAllSubsetOpsThroughContainingOpBlockArgument(
    mlir::RewriterBase &rewriter, mlir::Diagnostic &diag,
    mlir::Operation *producerOp, LoopLikeOpInterface containingOp);

mlir::Operation *cloneAndFuseAllSubsetOps(mlir::RewriterBase &rewriter,
                                          mlir::Diagnostic &diag,
                                          mlir::Operation *producerOp,
                                          mlir::Operation *containingOp);

/// Callback function type for generating transform dialect operations.
/// \param builder The OpBuilder to use.
/// \param loc The location for generated ops.
/// \param rootHandle The handle to the root operation (usually the module or
/// block).
using TransformGenerationCallback =
    std::function<void(OpBuilder &builder, Location loc, Value rootHandle)>;

/// Shared utility to apply a unified transform sequence to a module.
class TransformApplier {
public:
  /// Applies a transformation defined by the generator callback to the module.
  /// Uses a transactional approach (clones module) to ensure safety.
  static void apply(ModuleOp module, TransformGenerationCallback generator);
};

} // namespace mlir::dicp

#endif // BISHENGIR_TRANSFORMS_TRANSFORMS_H