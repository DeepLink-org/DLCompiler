

#include "dicp/DynamicCVPipeline/PlanComputeBlock/OpClassifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include <queue>

using namespace mlir;
static constexpr const char *DEBUG_TYPE = "op-classifier";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")
using namespace mlir::triton;

// Mark an operation as CUBE
void OpClassifierPass::markCube(Operation *op) {
  if (op && opCoreTypes.count(op)) {
    opCoreTypes[op] = static_cast<OpCoreType>(opCoreTypes[op] | OP_CUBE_ONLY);
    LLVM_DEBUG(DBGS() << "CUBE: " << *op << "\n");
  }
}

// ============================================================================
// Pattern: to_tensor → matmul (Upstream)
// ============================================================================
// Matches cases where matmul's input comes from bufferization.to_tensor.
// to_tensor converts a memref to a tensor, typically used for loading matrix
// data from memory. IR Example
//   %memref = memref.alloc() : memref<1024x1024xf32>
//   %tensor = bufferization.to_tensor %memref : memref<1024x1024xf32> ->
//   tensor<1024x1024xf32> %result = linalg.matmul ins(%tensor, %tensor_b)
//   outs(%init)
// Matching Logic
//   1. Check if matmul's operand defining op is bufferization.to_tensor
//   2. If matched, mark to_tensor as CUBE and add to cubeSeeds
//   3. Also trace back to_tensor's memref source and mark its defining op as
//   CUBE
// Purpose: Ensure matmul's input data loading executes on CUBE for efficient
// matrix data transfer.
// ============================================================================
void OpClassifierPass::matchToTensorPattern(Operation *def) {
  auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(def);
  if (!toTensorOp)
    return;

  markCube(toTensorOp);
  cubeSeeds.push_back(toTensorOp);

  // Also mark the memref allocation as CUBE
  Value memref = toTensorOp.getBuffer();
  if (Operation *memrefDef = memref.getDefiningOp()) {
    markCube(memrefDef);
    cubeSeeds.push_back(memrefDef);
  }
}

// ============================================================================
// Pattern: transpose → matmul (Upstream)
// ============================================================================
// Matches cases where matmul's input comes from linalg.transpose.
// transpose performs matrix transposition, commonly used as preprocessing for
// matrix multiplication inputs.
// IR Example
//   %input = tensor.empty() : tensor<1024x1024xf32>
//   %transposed = linalg.transpose ins(%input: tensor<1024x1024xf32>)
//   outs(%out: tensor<1024x1024xf32>)
//                   permutation = [1, 0]
//   %result = linalg.matmul ins(%transposed, %tensor_b) outs(%init)
// Matching Logic
//   1. Check if matmul's operand defining op is linalg.transpose
//   2. Mark transpose itself as CUBE (transpose operation is suitable for CUBE
//   execution)
//   3. Only mark transpose's input as CUBE seed when its operands come from:
//      - bufferization dialect ops (e.g., to_tensor, clone), excluding
//      AllocTensorOp
//      - tensor.empty op
//   4. Check both transpose's operands and DpsInits (output initial values),
//   either one
//      satisfying the condition is sufficient
// Special Handling
//   When transpose's input comes from compute ops (e.g., arith.truncf),
//   transpose itself remains CUBE, but its upstream chain stays VECTOR to avoid
//   unnecessary CUBE expansion.
// Purpose: transpose is a common preprocessing op for matmul; executing on CUBE
// leverages matrix transfer capabilities.
// ============================================================================
void OpClassifierPass::matchTransposePattern(Operation *def) {
  auto transposeOp = dyn_cast<linalg::TransposeOp>(def);
  if (!transposeOp)
    return;

  markCube(transposeOp);

  // Helper lambda to check if an operand's defining op qualifies for CUBE seed
  auto shouldMarkCubeSeed = [](Operation *opDef) -> bool {
    if (!opDef)
      return false;
    return (isa<bufferization::BufferizationDialect>(opDef->getDialect()) &&
            !isa<bufferization::AllocTensorOp>(opDef)) ||
           isa<tensor::EmptyOp>(opDef);
  };

  // Check input tensor
  auto operands = transposeOp->getOperands();
  for (const auto &op : operands) {
    if (shouldMarkCubeSeed(op.getDefiningOp())) {
      markCube(op.getDefiningOp());
      cubeSeeds.push_back(op.getDefiningOp());
      break; // No need to check other operands, one is enough to seed the
             // transpose as CUBE
    }
  }

  // Check outs (DpsInits)
  auto outs = transposeOp.getDpsInits();
  for (const auto &out : outs) {
    if (shouldMarkCubeSeed(out.getDefiningOp())) {
      markCube(out.getDefiningOp());
      cubeSeeds.push_back(out.getDefiningOp());
      break;
    }
  }
}

// ============================================================================
// Pattern: linalg.fill → matmul (Upstream)
// ============================================================================
// Matches cases where matmul's output initial value comes from linalg.fill.
// fill initializes the output matrix (typically to 0).
// IR Example
//   %value = arith.constant 0.0 : f32
//   %out = tensor.empty() : tensor<1024x1024xf32>
//   %init = linalg.fill ins(%value: f32) outs(%out: tensor<1024x1024xf32>)
//   %result = linalg.matmul ins(%a, %b) outs(%init)
// Matching Logic
//   1. Check if matmul's operand defining op is linalg.fill
//   2. If matched, mark fill as CUBE and add to cubeSeeds
// Purpose: fill operation initializes matmul's output buffer; executing on CUBE
// leverages efficient data filling capabilities.
// ============================================================================
void OpClassifierPass::matchFillPattern(Operation *def) {
  auto fillOp = dyn_cast<linalg::FillOp>(def);
  if (!fillOp)
    return;

  markCube(fillOp);
  cubeSeeds.push_back(fillOp);
}

// ============================================================================
// Pattern: matmul → hivm.hir.store (Downstream)
// ============================================================================
// Matches cases where matmul's output is directly stored to memory.
// hivm.store is the HIVM dialect's store operation, writing tensor to memory.
// IR Example
//   %result = linalg.matmul ins(%a, %b) outs(%init)
//   hivm.store %result, %memref[%offset] : tensor<1024x1024xf32> to memref
// Matching Logic
//   1. Check if matmul result's user is hivm.store
//   2. If matched, mark store as CUBE and add to cubeSeeds
// Purpose: Ensure matmul result's store operation executes on CUBE for
// efficient matrix data write-back.
// ============================================================================
void OpClassifierPass::matchStorePattern(Operation *user) {
  if (!isa<hivm::StoreOp>(user))
    return;

  markCube(user);
  cubeSeeds.push_back(user);
}

// ============================================================================
// Pattern: matmul → tensor.extract_slice → hivm.store (Downstream)
// ============================================================================
// Matches cases where matmul's output is sliced before being stored.
// extract_slice extracts a sub-region (slice) of a tensor, commonly used after
// tiled matrix computation for result extraction.
// IR Example
//   %result = linalg.matmul ins(%a, %b) outs(%init) : tensor<1024x1024xf32>
//   %slice = tensor.extract_slice %result[0, 0][256, 256][1, 1] :
//   tensor<1024x1024xf32> to tensor<256x256xf32> hivm.store %slice,
//   %memref[%offset] : tensor<256x256xf32> to memref
// Matching Logic
//   1. Check if matmul result's user is tensor.extract_slice
//   2. If matched, mark extract_slice as CUBE and add to cubeSeeds
//   3. Further check all users of extract_slice and mark hivm.store as CUBE
// Purpose: After tiled matrix computation, slice and store operations on CUBE
// improve local data transfer efficiency.
// ============================================================================
void OpClassifierPass::matchExtractSlicePattern(Operation *user) {
  auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
  if (!extractSliceOp)
    return;

  markCube(extractSliceOp);
  cubeSeeds.push_back(extractSliceOp);
  // Also mark downstream hivm.hir.store as CUBE
  for (Operation *sliceUser : extractSliceOp->getUsers()) {
    if (isa<hivm::StoreOp>(sliceUser)) {
      markCube(sliceUser);
      cubeSeeds.push_back(sliceUser);
    }
  }
}

// ============================================================================
// Pattern: matmul → materialize_in_destination (Downstream)
// ============================================================================
// Matches cases where matmul's output is written to a destination buffer via
// materialize_in_destination. This operation materializes tensor results into a
// pre-allocated destination location.
// IR Example
//   %result = linalg.matmul ins(%a, %b) outs(%init) : tensor<1024x1024xf32>
//   %memref = memref.alloc() : memref<1024x1024xf32>
//   bufferization.materialize_in_destination %result in %memref
//     : tensor<1024x1024xf32> to memref<1024x1024xf32>
// Matching Logic
//   1. Check if matmul result's user is
//   bufferization.materialize_in_destination
//   2. If matched, mark it as CUBE and add to cubeSeeds
// Purpose: matmul result materialization on CUBE leverages efficient matrix
// data transfer capabilities.
// ============================================================================
void OpClassifierPass::matchMaterializePattern(Operation *user) {
  if (!isa<bufferization::MaterializeInDestinationOp>(user))
    return;

  markCube(user);
  cubeSeeds.push_back(user);
}

// Pattern matching for CUBE operations
int OpClassifierPass::patternMatchCUBE() {
  LDBG("--- Step 1: pattern match --->\n");

  for (Operation *op : allOps) {
    if (!isa<linalg::MatmulOp>(op))
      continue;

    // matmul always CUBE
    opCoreTypes[op] = OP_CUBE_ONLY;
    LLVM_DEBUG(DBGS() << "CUBE (matmul): " << *op << "\n");

    // ---- Upstream pattern matching ----
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def)
        continue;

      matchToTensorPattern(def);
      matchTransposePattern(def);
      matchFillPattern(def);
    }

    // ---- Downstream pattern matching ----
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        matchStorePattern(user);
        matchExtractSlicePattern(user);
        matchMaterializePattern(user);
      }
    }
  }

  LLVM_DEBUG(DBGS() << "seeds: " << cubeSeeds.size() << " to_tensor(s)\n");
  for (Operation *seed : cubeSeeds) {
    LLVM_DEBUG(DBGS() << "CUBE: " << *seed << "\n");
  }

  return 0;
}

// Helper function to get upstream operations based on both SSA and memory
// dependencies
void OpClassifierPass::getUpstreamOpsWithMemoryDeps(
    Operation *cur, llvm::SmallVectorImpl<Operation *> &upstreamOps) {
  // Collect SSA dependencies (direct operands)
  for (Value operand : cur->getOperands()) {
    Operation *def = operand.getDefiningOp();
    if (def && def != cur) {
      upstreamOps.push_back(def);
    }
  }
  if (!isa<bufferization::ToTensorOp>(cur)) {
    // Only consider memory dependencies for non-to_tensor ops,
    // as to_tensor already has SSA deps to its memref source
    return;
  }
  // Collect memory dependencies
  if (memDepGraph) {
    // Get operations that define memory used by current op
    for (Operation *memDef : memDepGraph->getMemDefs(cur)) {
      LLVM_DEBUG(DBGS() << "memDef: cur " << *cur << " -> memDef " << *memDef
                        << "\n");
      if (isa<memref::CopyOp>(memDef)) {
        upstreamOps.push_back(memDef);
      }
    }
  }
}

// Propagate CUBE core type upstream
int OpClassifierPass::propagateCubeUpstream() {
  LDBG("--- Step 2: CUBE upstream BFS --->");
  llvm::DenseSet<Operation *> cubeVisited;
  std::queue<Operation *> cubeQueue;

  for (Operation *seed : cubeSeeds) {
    cubeVisited.insert(seed);
    cubeQueue.push(seed);
  }

  while (!cubeQueue.empty()) {
    Operation *cur = cubeQueue.front();
    cubeQueue.pop();

    // Get upstream operations considering both SSA and memory dependencies
    llvm::SmallVector<Operation *> upstreamOps;
    getUpstreamOpsWithMemoryDeps(cur, upstreamOps);

    // Next step: Reverse-propagate CUBE-only marking to upstream ops,
    // but stop at visited/matmul ops and skip values defined inside nested
    // linalg regions.
  }

  return 0;
}

// Run the pass
void OpClassifierPass::runOnOperation() {
  LDBG("\n--- Step 1: Running OpClassifierPass --->\n");

  ModuleOp module = getOperation();

  aliasAnalysis = std::make_shared<AliasAnalysis>(module);
  memDepGraph = std::make_shared<CVPipeline::MemoryDependenceGraph>(
      module, *aliasAnalysis);

  // Initialize the pass
  // Collect all operations
  module.walk([&](Operation *op) { allOps.push_back(op); });

  // Initialize all ops to UNDETERMINED
  for (Operation *op : allOps) {
    opCoreTypes[op] = OP_UNDETERMINED;
  }

  // Step 1: Pattern match around each linalg.matmul to find CUBE seeds
  if (patternMatchCUBE() != 0) {
    signalPassFailure();
    return;
  }

  // Step 2: CUBE upstream BFS from seed loads
  if (propagateCubeUpstream() != 0) {
    signalPassFailure();
    return;
  }
}

// Create the pass
namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createOpClassifierPass() {
  return std::make_unique<OpClassifierPass>();
}

} // namespace triton
} // namespace mlir
