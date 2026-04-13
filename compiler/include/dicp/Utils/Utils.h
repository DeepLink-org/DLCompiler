//===- Utils.h - DICP General Purpose Utilities ----------------*- C++ -*-===//
//
// Common utilities for DICP compiler infrastructure. This header contains
// generic helpers for IR manipulation, type analysis, and code generation
// that are not specific to any particular dialect or analysis pass.
//
//===----------------------------------------------------------------------===//

#ifndef DICP_UTILS_H
#define DICP_UTILS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include <functional>
#include <optional>

//===----------------------------------------------------------------------===//
// Backend Dispatch Macro
//===----------------------------------------------------------------------===//

/// Dispatch conversion pattern handlers based on backend string.
/// Executes ASCEND_HANDLER when backend == "ascend", otherwise DEFAULT_HANDLER.
#define DICP_DISPATCH_BACKEND_CONVERSION_PATTERNS(BACKEND_STR, ASCEND_HANDLER, \
                                                  DEFAULT_HANDLER)             \
  do {                                                                         \
    auto populatePatterns =                                                    \
        llvm::StringSwitch<std::function<void()>>(BACKEND_STR)                 \
            .Case("ascend", [&] { ASCEND_HANDLER; })                           \
            .Default([&] { DEFAULT_HANDLER; });                                \
    populatePatterns();                                                        \
  } while (0)

namespace mlir::dicp {

//===----------------------------------------------------------------------===//
// Operation Attribute Tags
//===----------------------------------------------------------------------===//

/// Tag constants for marking operations during transformation passes.
/// These strings are used as attribute names on operations.
namespace tags {
inline constexpr llvm::StringLiteral kGeneratedByMakeTensorPtr =
    "GeneratedByMakeTensorPtr";
inline constexpr llvm::StringLiteral kDiscreteMask = "DiscreteMask";
inline constexpr llvm::StringLiteral kDiscreteMemAccess = "DiscreteMemAccess";
} // namespace tags

//===----------------------------------------------------------------------===//
// Backend Configuration
//===----------------------------------------------------------------------===//

/// Returns the value of the "dicp.backend" attribute from the module.
/// Returns empty StringRef if the attribute is not present.
llvm::StringRef getBackend(ModuleOp module);

/// Returns true if the module is configured for the "ascend" backend.
bool isAscendBackend(ModuleOp module);

//===----------------------------------------------------------------------===//
// MemRef Type Utilities
//===----------------------------------------------------------------------===//

/// Returns the last (innermost) stride of a memref::ReinterpretCastOp
/// if it is a constant value. Returns std::nullopt otherwise.
std::optional<int64_t>
getLastStrideOfReinterpretCastOp(memref::ReinterpretCastOp op);

//===----------------------------------------------------------------------===//
// OpFoldResult Arithmetic Utilities
//===----------------------------------------------------------------------===//

/// Fold-aware index addition on OpFoldResult values.
/// Automatically simplifies constant expressions and eliminates zeros.
OpFoldResult addOfrs(OpBuilder &builder, Location loc, OpFoldResult lhs,
                     OpFoldResult rhs);

/// Fold-aware index multiplication on OpFoldResult values.
/// Automatically simplifies constant expressions and eliminates identity ops.
OpFoldResult mulOfrs(OpBuilder &builder, Location loc, OpFoldResult lhs,
                     OpFoldResult rhs);

/// Fold-aware index subtraction on OpFoldResult values.
/// Automatically simplifies constant expressions and eliminates zero RHS.
OpFoldResult subOfrs(OpBuilder &builder, Location loc, OpFoldResult lhs,
                     OpFoldResult rhs);

/// Fold-aware signed division on OpFoldResult values.
/// Returns an empty OpFoldResult and emits an error on division by zero.
OpFoldResult divOfrs(OpBuilder &builder, Location loc, OpFoldResult lhs,
                     OpFoldResult rhs);

/// Fold-aware signed remainder on OpFoldResult values.
/// Returns an empty OpFoldResult and emits an error on division by zero.
OpFoldResult remOfrs(OpBuilder &builder, Location loc, OpFoldResult lhs,
                     OpFoldResult rhs);

/// Fold-aware minimum on OpFoldResult values.
OpFoldResult minOfrs(OpBuilder &builder, Location loc, OpFoldResult lhs,
                     OpFoldResult rhs);

/// Fold-aware maximum on OpFoldResult values.
OpFoldResult maxOfrs(OpBuilder &builder, Location loc, OpFoldResult lhs,
                     OpFoldResult rhs);

//===----------------------------------------------------------------------===//
// Slice Proof Utilities
//===----------------------------------------------------------------------===//

/// Proves whether two OpFoldResult values are equal.
///
/// Uses fast path comparison first (identity, constant equality), then falls
/// back to ValueBoundsConstraintSet for symbolic reasoning.
///
/// \param lhs Left-hand side operand.
/// \param rhs Right-hand side operand.
/// \return Failure if cannot prove, true/false otherwise.
FailureOr<bool> proveOfrEqual(OpFoldResult lhs, OpFoldResult rhs);

/// Proves whether two N-dimensional slices are geometrically equivalent.
///
/// Uses ValueBoundsConstraintSet::areEquivalentSlices for symbolic reasoning.
/// This can prove equivalence even when sizes are symbolic values.
///
/// \param ctx MLIR context for constraint set.
/// \param offsetsA, sizesA, stridesA First slice geometry.
/// \param offsetsB, sizesB, stridesB Second slice geometry.
/// \return Failure if cannot prove, true/false otherwise.
///
/// **Note:** This is DIFFERENT from simple llvm::equal comparison!
/// It uses symbolic reasoning via ValueBoundsConstraintSet.
FailureOr<bool> proveSlicesEquivalent(MLIRContext *ctx,
                                      ArrayRef<OpFoldResult> offsetsA,
                                      ArrayRef<OpFoldResult> sizesA,
                                      ArrayRef<OpFoldResult> stridesA,
                                      ArrayRef<OpFoldResult> offsetsB,
                                      ArrayRef<OpFoldResult> sizesB,
                                      ArrayRef<OpFoldResult> stridesB);

/// Proves whether two N-dimensional slices are disjoint (non-overlapping).
///
/// Uses ValueBoundsConstraintSet::areOverlappingSlices and negates the result.
/// This is a pure mathematical proof function with no side effects.
///
/// \param ctx MLIR context for constraint set.
/// \param offsetsA, sizesA, stridesA First slice geometry.
/// \param offsetsB, sizesB, stridesB Second slice geometry.
/// \return Failure if cannot prove, true if provably disjoint, false if
///         provably overlapping.
FailureOr<bool> proveSlicesDisjoint(MLIRContext *ctx,
                                    ArrayRef<OpFoldResult> offsetsA,
                                    ArrayRef<OpFoldResult> sizesA,
                                    ArrayRef<OpFoldResult> stridesA,
                                    ArrayRef<OpFoldResult> offsetsB,
                                    ArrayRef<OpFoldResult> sizesB,
                                    ArrayRef<OpFoldResult> stridesB);

//===----------------------------------------------------------------------===//
// Index/Constant Utilities
//===----------------------------------------------------------------------===//

/// Creates a constant index value operation at the given location.
Value createConstIndexValueOp(Location loc, OpBuilder &b, int64_t value);

/// Extracts a constant int64_t value from an OpFoldResult if it holds
/// an Attribute. Returns std::nullopt if the OpFoldResult holds a Value.
std::optional<int64_t> getConstantOfAttr(const OpFoldResult &arg);

/// Traces a value through view-like ops and buffer/tensor adapters to its root.
/// This strips `ViewLikeOpInterface`, `memref.cast`,
/// `memref.reinterpret_cast`, `bufferization.to_tensor`, and
/// `bufferization.to_buffer`.
Value traceToSourceRoot(Value value);

//===----------------------------------------------------------------------===//
// Tensor/Slice Utilities
//===----------------------------------------------------------------------===//

/// Creates a new tensor by transposing the 'source' value according to 'order'.
Value getTransposedValue(Value source, Location loc,
                         ConversionPatternRewriter &rewriter,
                         llvm::ArrayRef<int> order);

/// Returns a vector of `n` `utils::IteratorType::parallel` attributes.
SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n);

//===----------------------------------------------------------------------===//
// Broadcast Analysis Utilities
//===----------------------------------------------------------------------===//

/// Identifies dimensions in the source tensor that are broadcast
/// (where source dim size is 1 but destination dim size differs).
/// Returns indices of broadcast dimensions.
SmallVector<int64_t> getBroadcastDims(RankedTensorType src,
                                      RankedTensorType dst);

/// Identifies dimensions that are NOT broadcast (source and dest match).
/// Returns the sizes of non-broadcast dimensions.
/// Note: This returns sizes, not indices, for use in collapsed tensor shape
/// construction.
SmallVector<int64_t> getUnbroadcastDims(RankedTensorType src,
                                        RankedTensorType dst);

//===----------------------------------------------------------------------===//
// IR Traversal Utilities
//===----------------------------------------------------------------------===//

/// Creates a series of `scf.for` loops for the given dimensions in `loopDims`.
/// Nesting is simulated by adjusting the insertion point to the body of the
/// last created loop. The `bodyFunc` is inserted into the innermost scope.
///
/// \param rewriter    The MLIR OpBuilder used to create operations.
/// \param loc         The source location information for debuggability.
/// \param target      The memref value whose dimensions are being looped over.
/// \param loopDims    An array of dimension indices to create loops for.
/// \param bodyFunc    A callable that defines the operations to insert in the
///                    innermost loop. It takes a SmallVector of induction
///                    variables (one per loop).
template <typename Func>
void createSimpleNestedLoops(OpBuilder &rewriter, Location loc, Value target,
                             ArrayRef<int> loopDims, Func bodyFunc) {
  SmallVector<Value> ivs;
  for (int dim : loopDims) {
    auto dimSize = rewriter.create<memref::DimOp>(loc, target, dim);
    auto loop = rewriter.create<scf::ForOp>(
        loc, rewriter.create<arith::ConstantIndexOp>(loc, 0), dimSize,
        rewriter.create<arith::ConstantIndexOp>(loc, 1));
    ivs.push_back(loop.getInductionVar());
    rewriter.setInsertionPointToStart(loop.getBody());
  }
  bodyFunc(ivs);
}

//===----------------------------------------------------------------------===//
// Typeless Value Specialization
//===----------------------------------------------------------------------===//

/// Enumeration for typeless value kinds (used for initial values).
enum class TypelessValue { Undefined = 0, Zero = 1, Min = 2, Max = 3 };

/// Specializes a typeless value kind into a concrete MLIR constant.
/// Returns failure if the type is not supported.
FailureOr<Value> specializeTypelessValueToConstant(TypelessValue, Type,
                                                   Location, OpBuilder &);

//===----------------------------------------------------------------------===//
// Shape Verification Utilities
//===----------------------------------------------------------------------===//

/// Returns success if the operation has fully static shapes for all
/// operands and results. MemRef cast and reinterpret_cast ops are
/// always accepted as they are layout-preserving.
LogicalResult verifyStaticShape(Operation *op);

} // namespace mlir::dicp

#endif // DICP_UTILS_H
