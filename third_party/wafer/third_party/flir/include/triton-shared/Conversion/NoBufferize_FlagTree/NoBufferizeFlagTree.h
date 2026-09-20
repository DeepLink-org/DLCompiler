#ifndef TRITON_CONVERSION_NOBUFFERIZEFLAGTREE_NOBUFFERIZEFLAGTREE_H
#define TRITON_CONVERSION_NOBUFFERIZEFLAGTREE_NOBUFFERIZEFLAGTREE_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
class TypeConverter;
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/NoBufferize_FlagTree/Passes.h.inc"

void populateNoBufferizeFlagTreeConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createNoBufferizeFlagTreePass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_NOBUFFERIZEFLAGTREE_NOBUFFERIZEFLAGTREE_H
