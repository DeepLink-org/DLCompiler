#ifndef TRITON_CONVERSION_MEMREFCOPYTODMAFLAGTREE_MEMREFCOPYTODMAFLAGTREE_H
#define TRITON_CONVERSION_MEMREFCOPYTODMAFLAGTREE_MEMREFCOPYTODMAFLAGTREE_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
class TypeConverter;
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/MemrefCopyToDMA_FlagTree/Passes.h.inc"

void populateMemrefCopyToDMAFlagTreeConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createMemrefCopyToDMAFlagTreePass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_STRUCTUREDTOMEMREF_STRUCTUREDTOMEMREF_H
