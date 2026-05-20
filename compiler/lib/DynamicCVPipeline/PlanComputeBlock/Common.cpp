

#include "dicp/DynamicCVPipeline/PlanComputeBlock/Common.h"

namespace mlir {
namespace CVPipeline {

void initializeIndegreeForBlock(Block *block,
                                llvm::DenseMap<Operation *, int> &indegree,
                                const MemoryDependenceGraph &memGraph) {
  auto &bm = ComputeBlockIdManager::getInstance();

  block->walk([&](Operation *op) {
    if (op->getBlock() != block) {
      return;
    }
    indegree[op] = 0;
    // We need to consider op itself && op's region-contained ops.
    op->walk([&](Operation *nestedOp) {
      for (auto inValue : nestedOp->getOperands()) {
        if (auto defOp = inValue.getDefiningOp()) {
          if (defOp->getBlock() == block && !bm.isSameBlock(defOp, op)) {
            indegree[op]++;
          }
        }
      }

      for (auto memDepUser : memGraph.getExecBefore(nestedOp)) {
        if (memDepUser->getBlock() == block &&
            !bm.isSameBlock(memDepUser, op)) {
          indegree[op]++;
        }
      }
    });
  });
}

Operation *getAncestorInBlock(Operation *inner, Block *block) {
  Operation *cur = inner;
  while (cur) {
    if (cur->getBlock() == block) {
      return cur;
    }
    cur = cur->getParentOp();
  }
  return nullptr;
}

} // namespace CVPipeline
} // namespace mlir
