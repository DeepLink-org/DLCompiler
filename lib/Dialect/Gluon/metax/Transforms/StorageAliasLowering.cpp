#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h"

#include <tuple>

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gluon-storage-alias-lowering"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[alias] " << X << "\n")

using namespace mlir;

namespace mlir::triton::gluon {

#define GEN_PASS_DEF_GLUONSTORAGEALIASLOWERINGPASS
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h.inc"

using StorageAliasOffsetMap =
    DenseMap<Value, std::tuple<int64_t, int64_t, int64_t>>;

LogicalResult computeOrValidateStorageAliasSizes(ModuleOp module);
LogicalResult processBufferOverlapOps(ModuleOp module,
                                      StorageAliasOffsetMap &offsetMap);
LogicalResult materializeStorageAliasAllocations(
    ModuleOp module, const StorageAliasOffsetMap &offsetMap);

namespace {
class GluonStorageAliasLoweringPass
    : public impl::GluonStorageAliasLoweringPassBase<
          GluonStorageAliasLoweringPass> {
public:
  using BaseT = impl::GluonStorageAliasLoweringPassBase<
      GluonStorageAliasLoweringPass>;
  using BaseT::BaseT;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    LDBG("computing storage-alias backing sizes");
    if (failed(computeOrValidateStorageAliasSizes(module))) {
      signalPassFailure();
      return;
    }

    LDBG("computing storage-alias overlap offsets");
    StorageAliasOffsetMap offsetMap;
    if (failed(processBufferOverlapOps(module, offsetMap))) {
      signalPassFailure();
      return;
    }

    LDBG("materializing storage-alias Shared allocations");
    if (failed(materializeStorageAliasAllocations(module, offsetMap)))
      signalPassFailure();
  }
};
} // namespace

} // namespace mlir::triton::gluon

#undef LDBG
#undef DEBUG_TYPE
