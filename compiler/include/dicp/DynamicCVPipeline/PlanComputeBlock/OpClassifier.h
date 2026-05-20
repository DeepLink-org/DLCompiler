

#ifndef TRITON_ADAPTER_OP_CLASSIFIER_H
#define TRITON_ADAPTER_OP_CLASSIFIER_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "dicp/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

// Core type enumeration for operations
enum OpCoreType {
  OP_UNDETERMINED = 0,
  OP_CUBE_ONLY = 1,
  OP_VECTOR_ONLY = 2,
  OP_CUBE_AND_VECTOR = 3
};

// OpClassifierPass for categorizing operations as CUBE or VECTOR
class OpClassifierPass
    : public PassWrapper<OpClassifierPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpClassifierPass)

  // Constructor
  OpClassifierPass() = default;

  // Run the pass
  void runOnOperation();

private:
  // Map from operation to its core type
  llvm::DenseMap<Operation *, OpCoreType> opCoreTypes;

  // All operations in the module
  llvm::SmallVector<Operation *> allOps;

  // Seed operations for CUBE upstream propagation
  llvm::SmallVector<Operation *> cubeSeeds;

  std::shared_ptr<AliasAnalysis> aliasAnalysis;
  std::shared_ptr<CVPipeline::MemoryDependenceGraph> memDepGraph;

  // Mark an operation as CUBE
  void markCube(Operation *op);

  // Pattern matching for CUBE operations
  int patternMatchCUBE();

  // Upstream pattern matching helpers
  void matchToTensorPattern(Operation *def);
  void matchTransposePattern(Operation *def);
  void matchFillPattern(Operation *def);

  // Downstream pattern matching helpers
  void matchStorePattern(Operation *user);
  void matchExtractSlicePattern(Operation *user);
  void matchMaterializePattern(Operation *user);

  // Propagate CUBE core type upstream
  int propagateCubeUpstream();

  // Get upstream operations based on both SSA and memory dependencies
  void
  getUpstreamOpsWithMemoryDeps(Operation *cur,
                               llvm::SmallVectorImpl<Operation *> &upstreamOps);
};

// Create the pass
std::unique_ptr<OperationPass<ModuleOp>> createOpClassifierPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_OP_CLASSIFIER_H
