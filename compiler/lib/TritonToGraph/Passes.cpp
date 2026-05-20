

#include "dicp/TritonToGraph/Passes.h"
#include "dicp/TritonToGraph/ControlFlowGraphBuilder.h"

#include "mlir/Pass/PassRegistry.h"

#define GEN_PASS_REGISTRATION
#include "dicp/TritonToGraph/Passes.h.inc"

namespace mlir {
namespace triton {

// registerTritonToCFGPasses() 由 Passes.h.inc 生成
// createBuildCFGPass() 的实现需要在 ControlFlowGraphBuilder.cpp 中

} // namespace triton
} // namespace mlir
