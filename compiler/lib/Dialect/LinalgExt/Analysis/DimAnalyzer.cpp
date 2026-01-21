#include "dicp/Dialect/LinalgExt/Analysis/DimAnalyzer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "npu-stage-dim-analyzer"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace dicp;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

std::string mlir::dicp::toString(DimKind k) {
  switch (k) {
  case DimKind::Unknown:
    return "Unknown";
  case DimKind::Parallel:
    return "Parallel";
  case DimKind::Reduction:
    return "Reduction";
  case DimKind::Broadcast:
    return "Broadcast";
  case DimKind::Complex:
    return "Complex";
  }
  return "INVALID";
}

//===----------------------------------------------------------------------===//
// DimensionDisjointSet Implementation
//===----------------------------------------------------------------------===//

int64_t DimensionDisjointSet::allocate(size_t n) {
  size_t start = parent.size();
  resize(start + n);
  LDBG("  [DSU] Allocated " << n << " new dims. Range: [" << start << ", "
                            << start + n - 1 << "]");
  return static_cast<int64_t>(start);
}

int64_t DimensionDisjointSet::find(int64_t i) {
  if (i < 0 || i >= (int64_t)parent.size())
    return -1;
  // Path compression: Point directly to the root to speed up future lookups.
  if (parent[i] == i)
    return i;
  return parent[i] = find(parent[i]);
}

void DimensionDisjointSet::unionSets(int64_t i, int64_t j) {
  int64_t rootI = find(i);
  int64_t rootJ = find(j);
  if (rootI != -1 && rootJ != -1 && rootI != rootJ) {
    DimKind kI = kind[rootI];
    DimKind kJ = kind[rootJ];

    // Merge properties based on priority logic (e.g., Reduction takes
    // precedence).
    DimKind mergedKind = mergeKinds(kI, kJ);

    // Union by attaching I to J (could be optimized with rank/size).
    parent[rootI] = rootJ;
    kind[rootJ] = mergedKind;

    LDBG("  [DSU] Union(ID:" << i << " [" << toString(kI) << "] -> ID:" << j
                             << " [" << toString(kJ)
                             << "]) => Merged Kind: " << toString(mergedKind));
  }
}

void DimensionDisjointSet::setKind(int64_t i, DimKind k) {
  int64_t root = find(i);
  if (root != -1) {
    DimKind oldK = kind[root];
    // Update the kind, ensuring we don't downgrade a strong property (like
    // Reduction).
    kind[root] = mergeKinds(kind[root], k);
    if (oldK != kind[root]) {
      LDBG("  [DSU] SetKind ID:" << i << " (Root:" << root << ") changed from "
                                 << toString(oldK) << " to "
                                 << toString(kind[root]));
    }
  }
}

DimKind DimensionDisjointSet::getKind(int64_t i) {
  int64_t root = find(i);
  return (root != -1) ? kind[root] : DimKind::Unknown;
}

void DimensionDisjointSet::resize(size_t n) {
  size_t oldSize = parent.size();
  if (n > oldSize) {
    parent.resize(n);
    // Initialize new elements to point to themselves (roots) with Unknown kind.
    std::iota(parent.begin() + oldSize, parent.end(), oldSize);
    kind.resize(n, DimKind::Unknown);
  }
}

DimKind DimensionDisjointSet::mergeKinds(DimKind a, DimKind b) {
  if (a == b)
    return a;
  // Complex is the strongest property: if a dimension is complex anywhere, it's
  // complex everywhere.
  if (a == DimKind::Complex || b == DimKind::Complex)
    return DimKind::Complex;
  // Reduction is stronger than Parallel/Broadcast: forces serialization/atomic
  // handling.
  if (a == DimKind::Reduction || b == DimKind::Reduction)
    return DimKind::Reduction;
  // Broadcast + Parallel is treated as Parallel for tiling purposes.
  // (Tiling a broadcasted loop is valid and often efficient).
  if ((a == DimKind::Broadcast && b == DimKind::Parallel) ||
      (a == DimKind::Parallel && b == DimKind::Broadcast))
    return DimKind::Parallel;
  // If one is Unknown, take the known one.
  return (a != DimKind::Unknown) ? a : b;
}

//===----------------------------------------------------------------------===//
// DimAnalyzer Implementation
//===----------------------------------------------------------------------===//

DimAnalyzer::DimAnalyzer(const StageInfo &stage) : stage_(stage) {
  // Populate the set for fast O(1) membership checks during traversal.
  for (auto *op : stage_.ops) {
    stageOps_.insert(op);
  }
}

std::vector<int64_t> DimAnalyzer::getOrAllocateDims(Value v) {
  if (valueDims_.count(v))
    return valueDims_[v];

  auto type = dyn_cast<ShapedType>(v.getType());
  if (!type || !type.hasRank()) {
    LDBG("  [Warn] Skipping unranked/non-shaped value: " << v);
    return {};
  }

  int64_t rank = type.getRank();
  int64_t startId = dsu_.allocate(rank);
  std::vector<int64_t> dims(rank);
  std::iota(dims.begin(), dims.end(), startId);

  // Default assumption: Dimensions are Parallel unless proven otherwise.
  // This helps when operations (like elementwise) don't impose constraints.
  for (auto id : dims)
    dsu_.setKind(id, DimKind::Parallel);

  valueDims_[v] = dims;
  return dims;
}

void DimAnalyzer::bindDimensions(Value v1, Value v2) {
  auto d1 = getOrAllocateDims(v1);
  auto d2 = getOrAllocateDims(v2);
  if (d1.empty() || d2.empty())
    return;

  if (d1.size() != d2.size()) {
    LDBG("  [Warn] Rank mismatch binding " << v1 << " <-> " << v2);
    return;
  }
  // 1-to-1 binding of dimensions (e.g., for Copy, Cast, or Elementwise).
  for (size_t i = 0; i < d1.size(); ++i) {
    dsu_.unionSets(d1[i], d2[i]);
  }
}

SmallVector<int64_t> DimAnalyzer::analyzeAndGetTilingDims() {
  LDBG("\n>>> [Analysis] Starting Analysis for Stage ID: " << stage_.id);
  // 1. Build the constraint graph via BFS traversal.
  processBFS();

  // 2. Identify Anchor Op.
  // Heuristic: The last LinalgOp in the stage is usually the "Compute" or
  // "Write" op. Tiling decisions should be based on this op's loop structure.
  linalg::LinalgOp anchorOp;
  for (auto it = stage_.ops.rbegin(); it != stage_.ops.rend(); ++it) {
    if (auto op = dyn_cast<linalg::LinalgOp>(*it)) {
      anchorOp = op;
      break;
    }
  }

  if (!anchorOp) {
    LDBG(">>> [Analysis] No LinalgOp anchor found. Tiling unknown.");
    return {};
  }

  LDBG(">>> [Analysis] Anchor Op: " << anchorOp->getName());

  // 3. Map Anchor Loops to Global Dimension IDs.
  SmallVector<int64_t> chosenLoops;
  auto iterTypes = anchorOp.getIteratorTypesArray();
  auto maps = anchorOp.getIndexingMapsArray();
  std::vector<int64_t> loopToDSU(iterTypes.size(), -1);

  // Iterate over operands to find which Value Dimension corresponds to which
  // Loop.
  auto operands = anchorOp->getOperands();
  int mapIdx = 0;
  for (auto val : operands) {
    if (mapIdx >= (int)maps.size())
      break;
    if (!isa<ShapedType>(val.getType())) {
      mapIdx++;
      continue;
    }

    auto valDims = getOrAllocateDims(val);
    AffineMap map = maps[mapIdx++];

    // Analyze the AffineMap: (d0, d1) -> (d0, d1)
    // If result[i] is a simple DimExpr(d_k), then Loop k corresponds to Value
    // Dim i.
    for (unsigned dimIdx = 0; dimIdx < map.getNumResults(); ++dimIdx) {
      if (dimIdx >= valDims.size())
        continue;
      if (auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(dimIdx))) {
        unsigned loopPos = dimExpr.getPosition();
        if (loopPos < loopToDSU.size()) {
          // Link the loop to the global DSU ID of the operand dimension.
          loopToDSU[loopPos] = valDims[dimIdx];
        }
      }
    }
  }

  // 4. Evaluate Loops for Tiling.
  LDBG(">>> [Analysis] Loop Classification:");
  for (size_t i = 0; i < loopToDSU.size(); ++i) {
    DimKind k = DimKind::Unknown;
    if (loopToDSU[i] != -1) {
      // Get the global property from DSU (propagated from all ops in the
      // stage).
      k = dsu_.getKind(loopToDSU[i]);
    } else {
      // Fallback: If loop isn't linked to any data dimension (rare), rely on
      // local iterator type.
      if (linalg::isReductionIterator(iterTypes[i]))
        k = DimKind::Reduction;
      else if (linalg::isParallelIterator(iterTypes[i]))
        k = DimKind::Parallel;
    }

    LDBG("    Loop " << i << ": " << toString(k));

    // Policy: We only auto-tile global Parallel loops.
    // (Future work: support Tiling Reduction if atomic updates are supported).
    if (k == DimKind::Parallel) {
      chosenLoops.push_back(i);
    }
  }
  return chosenLoops;
}

void DimAnalyzer::processBFS() {
  BFSQueue bfsQueue;
  VisitedSet visited;
  DenseSet<Value> definedInStage;

  // Identify all values defined within the stage to find boundary inputs.
  for (auto *op : stage_.ops)
    for (auto res : op->getResults())
      definedInStage.insert(res);

  // 1. Seeds: Operands used in stage but defined externally (Inputs).
  for (auto *op : stage_.ops) {
    for (auto operand : op->getOperands()) {
      if (!definedInStage.contains(operand)) {
        if (visited.insert(operand).second) {
          bfsQueue.push(operand);
          getOrAllocateDims(operand); // Pre-allocate IDs for inputs.
        }
      }
    }
  }

  // 2. Seeds: Internal roots (Fallback).
  // If the graph is fully internal or disconnected, start from the first op.
  if (bfsQueue.empty() && !stage_.ops.empty()) {
    for (auto res : stage_.ops[0]->getResults()) {
      bfsQueue.push(res);
      visited.insert(res);
    }
  }

  // Standard BFS Traversal
  while (!bfsQueue.empty()) {
    Value current = bfsQueue.front();
    bfsQueue.pop();

    for (Operation *user : current.getUsers()) {
      // Only process users that are part of the current stage.
      if (!stageOps_.contains(user))
        continue;

      // Dispatch processing to specific Op handler.
      // This establishes constraints between 'current' and 'user's results.
      processOperation(user, current, bfsQueue, visited);

      // Enqueue results for downstream propagation.
      for (Value result : user->getResults()) {
        if (visited.insert(result).second) {
          bfsQueue.push(result);
          getOrAllocateDims(result);
        }
      }
    }
  }
}

bool DimAnalyzer::processOperation(Operation *op, Value current, BFSQueue &q,
                                   VisitedSet &v) {
  // Dispatcher: Directs operation to the specific semantic handler.
  if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op))
    processMatmulOp(matmulOp);
  else if (auto reduceOp = dyn_cast<linalg::ReduceOp>(op))
    processReduceOp(reduceOp);
  else if (auto transOp = dyn_cast<linalg::TransposeOp>(op))
    processTransposeOp(transOp);
  else if (auto bcastOp = dyn_cast<linalg::BroadcastOp>(op))
    processBroadcastOp(bcastOp);
  else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
    processLinalgOpGeneric(linalgOp);

  // Tensor manipulation ops
  else if (auto castOp = dyn_cast<tensor::CastOp>(op))
    bindDimensions(castOp.getSource(), castOp.getDest());
  else if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op))
    processReshapeOp(op);
  else if (auto concatOp = dyn_cast<tensor::ConcatOp>(op))
    processConcatOp(concatOp);
  else if (auto padOp = dyn_cast<tensor::PadOp>(op))
    processPadOp(padOp);
  else if (auto extSlice = dyn_cast<tensor::ExtractSliceOp>(op))
    processExtractSliceOp(extSlice);
  else if (auto insSlice = dyn_cast<tensor::InsertSliceOp>(op))
    processInsertSliceOp(insSlice);

  // Bufferization & MemRef ops
  else if (auto copyOp = dyn_cast<memref::CopyOp>(op))
    processMemrefCopyOp(copyOp, current, q, v);
  else if (isa<memref::CastOp, memref::ReinterpretCastOp>(op))
    processMemrefCastOp(op);
  else if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(op))
    processBufferizationToTensor(toTensor);
  else if (auto matOp = dyn_cast<bufferization::MaterializeInDestinationOp>(op))
    processMaterializeOp(matOp, current, q, v);

  // Elementwise ops (Arith, Math)
  else if (isa<arith::ArithDialect, math::MathDialect>(op->getDialect()))
    processElementwise(op, current);
  else {
    // Default fallback: assume 1-to-1 preservation if results exist.
    if (op->getNumResults() > 0)
      bindDimensions(current, op->getResult(0));
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Specific Handlers
//===----------------------------------------------------------------------===//

void DimAnalyzer::processMemrefCopyOp(memref::CopyOp op, Value current,
                                      BFSQueue &q, VisitedSet &v) {
  LDBG("  [Op] Processing MemRef Copy");
  Value src = op.getSource();
  Value dst = op.getTarget();
  bindDimensions(src, dst);

  // Special Case: Copy sends data to 'dst', which is an operand (outs), not a
  // result. We must explicitly enqueue 'dst' to continue BFS.
  if (current == src) {
    if (v.insert(dst).second) {
      q.push(dst);
      getOrAllocateDims(dst);
      LDBG("    -> Enqueued Copy Destination: " << dst);
    }
  }
}

void DimAnalyzer::processMaterializeOp(
    bufferization::MaterializeInDestinationOp op, Value current, BFSQueue &q,
    VisitedSet &v) {
  LDBG("  [Op] Processing MaterializeInDestination");
  Value src = op.getSource();
  Value dst = op.getDest();
  bindDimensions(src, dst);

  // Similar to Copy: Propagate to destination buffer.
  if (current == src) {
    if (v.insert(dst).second) {
      q.push(dst);
      getOrAllocateDims(dst);
      LDBG("    -> Enqueued Materialize Destination: " << dst);
    }
  }
}

void DimAnalyzer::processMemrefCastOp(Operation *op) {
  LDBG("  [Op] Processing MemRef Cast/Reinterpret");
  Value src = op->getOperand(0);
  Value dst = op->getResult(0);

  auto srcType = dyn_cast<ShapedType>(src.getType());
  auto dstType = dyn_cast<ShapedType>(dst.getType());

  if (srcType && srcType.hasRank() && dstType && dstType.hasRank()) {
    if (srcType.getRank() == dstType.getRank()) {
      bindDimensions(src, dst);
    } else {
      // Rank changing casts (e.g. collapse/expand via reinterpret) break strict
      // 1-to-1 binding. We treat dst dims as new/separate.
      LDBG("    Rank change detected, breaking strict binding.");
      getOrAllocateDims(dst);
    }
  } else {
    getOrAllocateDims(dst);
  }
}

void DimAnalyzer::processBufferizationToTensor(bufferization::ToTensorOp op) {
  LDBG("  [Op] Processing ToTensor");
  // Converts MemRef to Tensor. Dimensions are strictly preserved.
  Value memrefValue = op.getOperand();
  Value tensorResult = op.getResult();
  bindDimensions(memrefValue, tensorResult);
}

void DimAnalyzer::processTransposeOp(linalg::TransposeOp op) {
  LDBG("  [Op] Processing TransposeOp");
  Value input = op.getInput();
  Value result = op.getResult()[0];
  auto perm = op.getPermutation();

  auto inputDims = getOrAllocateDims(input);
  auto resDims = getOrAllocateDims(result);

  if (inputDims.empty() || resDims.empty())
    return;

  // Bind Input[Perm[i]] <-> Result[i]
  for (size_t i = 0; i < perm.size(); ++i) {
    int64_t srcIdx = perm[i];
    if (srcIdx < (int)inputDims.size() && i < resDims.size()) {
      dsu_.unionSets(inputDims[srcIdx], resDims[i]);
    }
  }
}

void DimAnalyzer::processMatmulOp(linalg::MatmulOp op) {
  LDBG("  [Op] Processing MatmulOp");
  // Standard Matmul: [M, K] * [K, N] -> [M, N]
  Value lhs = op.getInputs()[0];
  Value rhs = op.getInputs()[1];
  Value out = op.getResults()[0];

  auto lhsDims = getOrAllocateDims(lhs);
  auto rhsDims = getOrAllocateDims(rhs);
  auto outDims = getOrAllocateDims(out);

  // Allocate implicit loops for M, N, K and set their properties.
  int64_t loopM = dsu_.allocate(1);
  int64_t loopN = dsu_.allocate(1);
  int64_t loopK = dsu_.allocate(1);
  dsu_.setKind(loopM, DimKind::Parallel);
  dsu_.setKind(loopN, DimKind::Parallel);
  dsu_.setKind(loopK, DimKind::Reduction);

  // Bind operand dimensions to these loops.
  // Assumes standard layout: LHS=[..., M, K], RHS=[..., K, N], Out=[..., M, N]
  if (lhsDims.size() >= 2 && rhsDims.size() >= 2 && outDims.size() >= 2) {
    dsu_.unionSets(lhsDims[lhsDims.size() - 2], loopM);
    dsu_.unionSets(lhsDims[lhsDims.size() - 1], loopK);
    dsu_.unionSets(rhsDims[rhsDims.size() - 2], loopK);
    dsu_.unionSets(rhsDims[rhsDims.size() - 1], loopN);
    dsu_.unionSets(outDims[outDims.size() - 2], loopM);
    dsu_.unionSets(outDims[outDims.size() - 1], loopN);
  }
}

void DimAnalyzer::processReduceOp(linalg::ReduceOp op) {
  LDBG("  [Op] Processing ReduceOp");
  Value input = op.getInputs()[0];
  Value output = op.getResults()[0];
  auto inputDims = getOrAllocateDims(input);
  auto outputDims = getOrAllocateDims(output);
  auto reduceIndices = op.getDimensions();
  std::set<int64_t> reduceSet(reduceIndices.begin(), reduceIndices.end());

  int outIdx = 0;
  for (size_t i = 0; i < inputDims.size(); ++i) {
    if (reduceSet.count(i)) {
      // Input dimension is being reduced -> Mark as Reduction.
      dsu_.setKind(inputDims[i], DimKind::Reduction);
    } else if (outIdx < (int)outputDims.size()) {
      // Input dimension is preserved -> Bind to Output dimension.
      dsu_.unionSets(inputDims[i], outputDims[outIdx++]);
    }
  }
}

void DimAnalyzer::processBroadcastOp(linalg::BroadcastOp op) {
  LDBG("  [Op] Processing BroadcastOp");
  auto inDims = getOrAllocateDims(op.getInput());
  auto resDims = getOrAllocateDims(op.getResult()[0]);
  auto broadcastIndices = op.getDimensions();
  std::set<int64_t> bcastSet(broadcastIndices.begin(), broadcastIndices.end());

  int inIdx = 0;
  for (size_t i = 0; i < resDims.size(); ++i) {
    if (bcastSet.count(i)) {
      // New dimension added by broadcast -> Mark as Broadcast.
      dsu_.setKind(resDims[i], DimKind::Broadcast);
    } else if (inIdx < (int)inDims.size()) {
      // Existing dimension -> Bind to input.
      dsu_.unionSets(resDims[i], inDims[inIdx++]);
    }
  }
}

void DimAnalyzer::processReshapeOp(Operation *op) {
  LDBG("  [Op] Processing Reshape");
  bool isExpand = isa<tensor::ExpandShapeOp>(op);
  auto srcDims = getOrAllocateDims(op->getOperand(0));
  auto dstDims = getOrAllocateDims(op->getResult(0));

  SmallVector<ReassociationIndices> indices;
  if (isExpand) {
    indices = cast<tensor::ExpandShapeOp>(op).getReassociationIndices();
  } else {
    indices = cast<tensor::CollapseShapeOp>(op).getReassociationIndices();
  }

  // Map between Collapsed (1 dim) and Expanded (N dims).
  auto &collapsed = isExpand ? srcDims : dstDims;
  auto &expanded = isExpand ? dstDims : srcDims;

  if (indices.size() != collapsed.size())
    return;

  // Bind the single collapsed dimension to ALL corresponding expanded
  // dimensions. This is a conservative approach: it effectively groups them all
  // into one equivalence class.
  for (size_t i = 0; i < indices.size(); ++i) {
    int64_t colID = collapsed[i];
    for (int64_t expIdx : indices[i]) {
      if (expIdx < (int64_t)expanded.size())
        dsu_.unionSets(colID, expanded[expIdx]);
    }
  }
}

void DimAnalyzer::processElementwise(Operation *op, Value current) {
  LDBG("  [Op] Processing Elementwise");
  // Elementwise ops (Add, Sub, etc.) strictly preserve shape.
  // Bind input dimensions to result dimensions 1-to-1.
  if (op->getNumResults() > 0)
    bindDimensions(current, op->getResult(0));
}

void DimAnalyzer::processConcatOp(tensor::ConcatOp op) {
  // Concat preserves all dimensions except the concatenation axis.
  // Even on the concat axis, the logical meaning of the dimension usually
  // matches (e.g., stacking Batches). We bind inputs to output 1-to-1.
  Value result = op.getResult();
  for (Value operand : op.getOperands())
    bindDimensions(operand, result);
}

void DimAnalyzer::processPadOp(tensor::PadOp op) {
  // Padding extends the size but preserves the logical axis.
  bindDimensions(op.getSource(), op.getResult());
}

void DimAnalyzer::processExtractSliceOp(tensor::ExtractSliceOp op) {
  auto srcDims = getOrAllocateDims(op.getSource());
  auto dstDims = getOrAllocateDims(op.getResult());
  auto dropped = op.getDroppedDims();
  int dstIdx = 0;
  for (size_t i = 0; i < srcDims.size(); ++i) {
    // If dimension is NOT dropped (rank-reduced), bind it to the next output
    // dimension.
    if (!dropped.test(i) && dstIdx < (int)dstDims.size()) {
      dsu_.unionSets(srcDims[i], dstDims[dstIdx++]);
    }
    // Dropped dimensions are effectively ignored for tiling propagation of the
    // result.
  }
}

void DimAnalyzer::processInsertSliceOp(tensor::InsertSliceOp op) {
  // InsertSlice modifies 'Dest'. The Result shape matches 'Dest'.
  bindDimensions(op.getDest(), op.getResult());
}

void DimAnalyzer::processLinalgOpGeneric(linalg::LinalgOp op) {
  LDBG("  [Op] Processing Generic: " << op->getName());
  auto maps = op.getIndexingMapsArray();
  auto iterTypes = op.getIteratorTypesArray();

  // Allocate IDs for the op's loop iterators.
  int64_t loopStart = dsu_.allocate(op.getNumLoops());

  // Set properties based on iterator types (Parallel vs Reduction).
  for (int i = 0; i < (int)iterTypes.size(); ++i) {
    DimKind k = linalg::isReductionIterator(iterTypes[i]) ? DimKind::Reduction
                                                          : DimKind::Parallel;
    dsu_.setKind(loopStart + i, k);
  }

  // Bind Operands to Loops using AffineMaps.
  auto operands = op->getOperands();
  int mapIdx = 0;
  for (auto val : operands) {
    if (mapIdx >= (int)maps.size())
      break;
    if (!isa<ShapedType>(val.getType())) {
      mapIdx++;
      continue;
    }

    AffineMap map = maps[mapIdx++];
    auto valDims = getOrAllocateDims(val);

    // If map is (d0, d1) -> (d0, d1), bind ValDim[0] to Loop[0], etc.
    for (unsigned d = 0; d < map.getNumResults(); ++d) {
      if (d >= valDims.size())
        continue;
      if (auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(d))) {
        dsu_.unionSets(valDims[d], loopStart + dimExpr.getPosition());
      }
    }
  }
}