

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_COMMON_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_COMMON_H

#include "dicp/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "dicp/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace CVPipeline {

Operation *getAncestorInBlock(Operation *inner, Block *block);
void initializeIndegreeForBlock(Block *block,
                                llvm::DenseMap<Operation *, int> &indegree,
                                const MemoryDependenceGraph &memGraph);

} // namespace CVPipeline
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_COMMON_H
