

#ifndef TRITON_ADAPTER_ADD_MULTI_BUFFER_TO_GMLOAD_PASS_H
#define TRITON_ADAPTER_ADD_MULTI_BUFFER_TO_GMLOAD_PASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

/// Sub-pipeline pass that applies multi-buffering to GM load operations.
class AddMultiBufferToGMLoadPass
    : public PassWrapper<AddMultiBufferToGMLoadPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddMultiBufferToGMLoadPass)

  AddMultiBufferToGMLoadPass() = default;

  void runOnOperation() override;
};

std::unique_ptr<OperationPass<ModuleOp>> createAddMultiBufferToGMLoadPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_ADD_MULTI_BUFFER_TO_GMLOAD_PASS_H
