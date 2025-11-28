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

std::unique_ptr<OperationPass<mlir::ModuleOp>> createLinalgIfToSelectPass();

std::unique_ptr<OperationPass<mlir::ModuleOp>> createLinalgGenericToSCFPass();

std::unique_ptr<OperationPass<mlir::func::FuncOp>> createScalarTo1DTensorPass();

std::unique_ptr<OperationPass<mlir::func::FuncOp>> createNormalizeSliceOpsPass();

#define GEN_PASS_REGISTRATION
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"

} // namespace mlir::dicp::LinalgExt

#endif
