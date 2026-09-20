#ifndef WAFER_TRANSFORMS_PASSES_H
#define WAFER_TRANSFORMS_PASSES_H
#include "mlir/Pass/Pass.h"
namespace mlir::triton {
std::unique_ptr<OperationPass<ModuleOp>> createInsertBarrierPass();
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "wafer/Transforms/Passes.h.inc"
}
#endif
