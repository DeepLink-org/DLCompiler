

#include "dicp/DynamicCVPipeline/SeparateMemoryFromCompute/AddMultiBufferToGMLoadPass.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "AddMultiBufferToGMLoad";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace triton;

namespace {

/// A single marked op with its dependency chain inside the enclosing region.
struct MarkedLoad {
  Operation *markedOp;
  SmallVector<Operation *> chain;
  memref::AllocOp allocOp = nullptr;
};

} // anonymous namespace

void AddMultiBufferToGMLoadPass::runOnOperation() {
  auto module = getOperation();
  LDBG("Enter AddMultiBufferToGMLoad pass");

  // All marked ops collected from the module IR.
  SmallVector<MarkedLoad> markedLoads;

  // Step 1: Scan the module IR for ops carrying the `gm_load_bufferable`
  //   attribute, and compute the dependency chain for each marked op.

  if (markedLoads.empty()) {
    LDBG("No marked loads found, nothing to transform");
    return;
  }

  LDBG("Marked loads collected, start transformation");

  // Step 2: For each marked load, apply multi-buffer transformation.
  //   Allocate buffer slots, build producer/consumer logic, and rewrite
  //   the original op to consume data from the selected buffer slot.

  // Step 3: Clean up transformed IR.
  //   Erase replaced original ops and prune dead values introduced
  //   during the transformation.

  LDBG("Process successfully");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAddMultiBufferToGMLoadPass() {
  return std::make_unique<AddMultiBufferToGMLoadPass>();
}

} // namespace triton
} // namespace mlir
