#ifndef TRITON_ADAPTER_ASCENDLEGALIZE_H
#define TRITON_ADAPTER_ASCENDLEGALIZE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_DECL_ASCENDLEGALIZE
#include "dicp/AscendLegalize/Passes.h.inc"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAscendLegalizePass();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_ASCENDLEGALIZE_H
