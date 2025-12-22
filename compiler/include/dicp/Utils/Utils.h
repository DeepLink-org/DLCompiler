#ifndef TRITON_UTILS_H
#define TRITON_UTILS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSwitch.h"

#include <functional>
#include <optional>

// Dispatch conversion pattern handlers based on backend string. Executes
// ASCEND_HANDLER when backend == "ascend", otherwise DEFAULT_HANDLER.
#define DISPATCH_BACKEND_CONVERSION_PATTERNS(BACKEND_STR, ASCEND_HANDLER,      \
                                             DEFAULT_HANDLER)                  \
  do {                                                                         \
    auto populatePatterns =                                                    \
        llvm::StringSwitch<std::function<void()>>(BACKEND_STR)                 \
            .Case("ascend", [&] { ASCEND_HANDLER; })                           \
            .Default([&] { DEFAULT_HANDLER; });                                \
    populatePatterns();                                                        \
  } while (0)

namespace mlir::dicp {

// Tags used for marking specific operations for later processing or
// identification.
const std::string GeneratedByMakeTensorPtrTAG = "GeneratedByMakeTensorPtr";
const std::string MayImplicitTransposeWithLastAxisTAG =
    "MayImplicitTransposeWithLastAxis";
const std::string discreteMaskAttrName = "DiscreteMask";
const std::string discreteAttrName = "DiscreteMemAccess";

// Gets the string attribute "dicp.backend" from the module if it exists.
llvm::StringRef getBackend(ModuleOp module);

bool isAscendBackend(ModuleOp module);

bool isaPermutedMemRefType(MemRefType);

// Retrieves the last (innermost) stride of a memref::ReinterpretCastOp if it is
// a constant.
std::optional<int64_t>
getLastStrideOfReinterpretCastOp(memref::ReinterpretCastOp op);

// Creates a new tensor by transposing the 'source' value according to the
// 'order'.
Value getTransposedValue(Value source, const Location loc,
                         ConversionPatternRewriter &rewriter,
                         llvm::ArrayRef<int> order);

// Returns a vector of `n` `utils::IteratorType::parallel` attributes.
SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n);

// Reconstructs the scalar value from an operand that might be a tensor/vector
// containing a single splat value, handling implicit casts like `sitofp` or
// `truncf`.
Value getScalarValue(Value operand, Location loc,
                     ConversionPatternRewriter &rewriter);

// Identifies the dimensions in the source tensor that are broadcast to match
// the destination tensor's shape (where source dim size is 1).
SmallVector<int64_t> getBroadcastDims(RankedTensorType src,
                                      RankedTensorType dst);

// Identifies the dimensions that are NOT broadcast (i.e., source shape matches
// destination shape).
SmallVector<int64_t> getUnbroadcastDims(RankedTensorType src,
                                        RankedTensorType dst);

// Enumeration for types of operations that interact with memory indirectly
// (e.g., loads/computations on pointers).
enum class IndirectLoadInterfaceOpType { Undefined = 0, Load = 1, Calc = 2 };

// Traces back from a 'rootOp' through its operands' definitions to find the
// first operation that satisfies the specified 'condFn'.
mlir::Operation *
findFirstMatchingOperandDef(mlir::Operation *rootOp,
                            const std::function<bool(Operation *)> &condFn);

/// Maximum expected rank for loop tiling in tensor operations.
static constexpr int kMaxTiledRank = 4;

/// This function generates a series of `scf.for` loops for the given dimensions
/// in `loopDims`. Although the loops are created sequentially, nesting is
/// simulated by adjusting the insertion point to the body of the last created
/// loop. This allows the `bodyFunc` to be inserted into the innermost scope.
///
/// \param rewriter The MLIR OpBuilder used to create operations.
/// \param loc The source location information for debuggability.
/// \param target The memref value whose dimensions are being looped over.
/// \param loopDims An array of dimension indices to create loops for.
/// \param bodyFunc A callable that defines the operations to insert in the
/// innermost loop.
///                 It takes a SmallVector of induction variables (one per
///                 loop).
///
template <typename Func>
void createSimpleNestedLoops(OpBuilder &rewriter, Location loc, Value target,
                             ArrayRef<int> loopDims, Func bodyFunc) {
  // Implementation details omitted in header but provided in the question's
  // context.
  // ...
}

// Recursively creates a potentially nested structure of `scf.for` loops.
// This allows for defining complex loop nests where the body is generated by
// 'bodyBuilder'.
scf::ForOp createNestedLoops(
    OpBuilder &builder, Location loc, unsigned currentDim, unsigned totalDims,
    ValueRange LBs, ValueRange UBs, ValueRange steps, SmallVector<Value> &ivs,
    ValueRange initArgs,
    function_ref<void(OpBuilder &, Location, SmallVector<Value> &, ValueRange)>
        bodyBuilder);

enum class TypelessValue { Undefined = 0, Zero = 1, Min = 2, Max = 3 };

FailureOr<Value> specializeTypelessValueToConstant(TypelessValue, Type,
                                                   Location, OpBuilder &);

} // namespace mlir::dicp

#endif // TRITONNPU_UTILS_UTILS_H