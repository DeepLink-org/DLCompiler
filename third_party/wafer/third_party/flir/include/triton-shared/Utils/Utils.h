#ifndef TRITON_SHARED_UTILITY_H
#define TRITON_SHARED_UTILITY_H

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
namespace mlir {
namespace triton {
// Return true if the input type is a triton pointer or a tensor of triton pointers
bool isPtrTypeLike(Type t);

// Extract a scalar value from v.
// If v is a scalar, return that directly. Otherwise, parse through operations
// (currently only support splat, sitofp, and truncf) that produce it to
// extract the underlying scalar value. We then reconstruct the chain of
// operations that can produce this constant with the original type. If no
// scalar value can be extracted, a nullptr is returned.
Value getScalarValue(Value operand, Location loc, OpBuilder &builder);

Value declareWaferRuntimeFunction(ModuleOp module, OpBuilder &builder, Location loc,
                          StringRef name, Type resultType,
                          ArrayRef<Type> argumentTypes);

bool isOperandMemorySpaceSPM(Value operand);

// NOTE: Reduction can lower to target reduction ops or target elementwise ops.

// Target support reduce instruction
// Different target ops have different supported types restriction
bool isTypeRestrictedTargetSupportedReductionOp(mlir::Operation *redOp);
// Integer and float type reduction op
bool isTargetSupportedReductionOp(mlir::Operation *redOp);

// Reduce to elementwise op
bool isTypeRestrictedTargetSupportedReduceToElementWiseOp(
    mlir::Operation *redOp);
// Integer and float type reduction op
bool isTargetSupportedReduceToElementWiseOp(mlir::Operation *redOp);

bool isTritonAllowedReductionOp(Operation *redOp);

// Reduction ops and elementwise op only support float types.
bool isTargetSupportedFloatType(Type elementType);
bool isTargetSupportedType(Type elementType);

bool isReductionOpAndTypeSupportedByTarget(mlir::Operation *redOp,
                                           Type elementType);
bool isReduceToElementWiseOpAndTypeSupportedByTarget(mlir::Operation *redOp,
                                                     Type elementType,
                                                     int64_t elemCount,
                                                     int64_t rank);

template <typename T> llvm::SmallVector<Operation *> getRegionOps(T linalgOp) {
  assert((linalgOp->getNumRegions() == 1 &&
          linalgOp->getRegion(0).hasOneBlock()) &&
         "It only applies to ops with one region and one block!");
  auto regionBlock = linalgOp.getBlock();
  return llvm::map_to_vector(regionBlock->without_terminator(),
                             [](Operation &op) { return &op; });
}

} // namespace triton

} // namespace mlir

// Gelu mode.
enum class GeluMode { None = 0, Tanh = 1 };

#endif // TRITON_SHARED_UTILITY_H
