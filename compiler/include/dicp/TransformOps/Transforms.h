#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"

#ifndef BISHENGIR_TRANSFORMS_TRANSFORMS_H
#define BISHENGIR_TRANSFORMS_TRANSFORMS_H

namespace mlir::dicp {

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
