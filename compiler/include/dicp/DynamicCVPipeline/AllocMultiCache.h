

#ifndef TRITON_ADAPTER_ALLOC_MULTI_CACHE_PASS_H
#define TRITON_ADAPTER_ALLOC_MULTI_CACHE_PASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

// AllocMultiCachePass for allocating multi-cache
class AllocMultiCachePass
    : public PassWrapper<AllocMultiCachePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AllocMultiCachePass)

  // Constructor
  AllocMultiCachePass() = default;

  // Run the pass
  void runOnOperation() override;
};

// Create the pass
std::unique_ptr<OperationPass<ModuleOp>> createAllocMultiCachePass();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_ALLOC_MULTI_CACHE_PASS_H
