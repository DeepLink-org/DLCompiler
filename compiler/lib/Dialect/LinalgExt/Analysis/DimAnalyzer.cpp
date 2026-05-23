#include "dicp/Dialect/LinalgExt/Analysis/DimAnalyzer.h"

#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <type_traits>

#define DEBUG_TYPE "npu-stage-dim-analyzer"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace dicp;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

llvm::StringRef mlir::dicp::toString(DimKind k) {
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
  }
}

void DimensionDisjointSet::setKind(int64_t i, DimKind k) {
  int64_t root = find(i);
  if (root != -1) {
    // Update the kind, ensuring we don't downgrade a strong property (like
    // Reduction).
    kind[root] = mergeKinds(kind[root], k);
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

DimAnalyzer::DimAnalyzer(const SmallVector<Operation *, 16> &ops)
    : inputOps(ops) {
  // Populate the set for fast O(1) membership checks during traversal.
  for (auto *op : inputOps) {
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
  // 1. Build the constraint graph via BFS traversal to propagate dimension
  // properties.
  runAnalysis();

  // 2. Identify Anchor Op. Heuristic: We traverse the stage operations in
  // reverse order. The last compute, memory movement, or SIMD operation
  // typically dictates the loop structure and tiling strategy for the entire
  // stage.
  Operation *anchorOp = nullptr;
  for (Operation *op : llvm::reverse(inputOps)) {
    if (isa<linalg::LinalgOp, memref::CopyOp,
            bufferization::MaterializeInDestinationOp>(op) ||
        isSIMDLikeOp(op)) {
      anchorOp = op;
      break;
    }
  }

  if (!anchorOp) {
    LDBG(">>> [Analysis] No valid anchor operation found "
         "(Linalg/Copy/Materialize/SIMD). Tiling dimensions cannot be "
         "determined.");
    return {};
  }

  LDBG(">>> [Analysis] Selected Anchor Op: '" << *anchorOp << "'");

  // 3. Extract Iteration Space (Loop -> DSU ID mapping). We need a unified way
  // to represent the implicit or explicit loops of the anchor op.
  SmallVector<int64_t> loopToDSU;
  SmallVector<DimKind> loopDefaultKinds;

  // Helper lambda: Infers a fully parallel iteration space from a ShapedType.
  // Used for operations without explicit affine indexing maps (e.g., Copies,
  // Elementwise).
  auto extractFromShapedValue = [&](Value v) {
    if (!v || !isa<ShapedType>(v.getType()))
      return;
    auto dims = getOrAllocateDims(v);
    loopToDSU.assign(dims.begin(), dims.end());
    // Data movement and simple elementwise ops imply fully parallel iteration
    // spaces.
    loopDefaultKinds.assign(dims.size(), DimKind::Parallel);
  };

  // Dispatch based on the precise type of the Anchor Operation.
  TypeSwitch<Operation *>(anchorOp)
      .Case<linalg::LinalgOp>([&](linalg::LinalgOp linalgOp) {
        LDBG("    -> Anchor is a LinalgOp. Extracting affine maps and "
             "iterators.");
        auto iterTypes = linalgOp.getIteratorTypesArray();
        auto maps = linalgOp.getIndexingMapsArray();
        const size_t numLoops = iterTypes.size();

        loopToDSU.assign(numLoops, -1);
        loopDefaultKinds.reserve(numLoops);

        // Extract default loop properties from the Linalg iterator types.
        for (auto iterType : iterTypes) {
          loopDefaultKinds.push_back(linalg::isReductionIterator(iterType)
                                         ? DimKind::Reduction
                                         : DimKind::Parallel);
        }

        // Map the Linalg loops to global DSU IDs using the operand affine maps.
        size_t mapIdx = 0;
        for (Value val : linalgOp->getOperands()) {
          if (mapIdx >= maps.size())
            break;
          if (!isa<ShapedType>(val.getType())) {
            mapIdx++;
            continue;
          }

          auto valDims = getOrAllocateDims(val);
          AffineMap map = maps[mapIdx++];

          // Example map: (d0, d1) -> (d0, d1)
          // Binds global DSU dimension 'valDims[dimIdx]' to local loop
          // 'loopPos'.
          for (unsigned dimIdx = 0; dimIdx < map.getNumResults(); ++dimIdx) {
            if (dimIdx >= valDims.size())
              continue;
            if (auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(dimIdx))) {
              unsigned loopPos = dimExpr.getPosition();
              if (loopPos < numLoops) {
                loopToDSU[loopPos] = valDims[dimIdx];
              }
            }
          }
        }
      })
      .Case<memref::CopyOp>([&](memref::CopyOp copyOp) {
        LDBG("    -> Anchor is a memref::CopyOp. Using target shape for "
             "iteration space.");
        extractFromShapedValue(copyOp.getTarget());
      })
      .Case<bufferization::MaterializeInDestinationOp>(
          [&](bufferization::MaterializeInDestinationOp matOp) {
            LDBG("    -> Anchor is a MaterializeInDestinationOp. Using "
                 "destination shape.");
            extractFromShapedValue(matOp.getDest());
          })
      .Default([&](Operation *op) {
        LDBG("    -> Anchor is a SIMD-like Op. Inferring iteration space from "
             "shaped operands/results.");
        // Fallback for `isSIMDLikeOp` (e.g., arith/math ops on
        // tensors/memrefs). Find the first shaped result or operand to define
        // the loop space.
        Value shapedVal;
        for (Value res : op->getResults()) {
          if (isa<ShapedType>(res.getType())) {
            shapedVal = res;
            break;
          }
        }
        if (!shapedVal) {
          for (Value opnd : op->getOperands()) {
            if (isa<ShapedType>(opnd.getType())) {
              shapedVal = opnd;
              break;
            }
          }
        }

        if (shapedVal) {
          extractFromShapedValue(shapedVal);
        } else {
          LDBG("    [Warn] SIMD-like anchor has no shaped operands or results. "
               "Cannot infer loops.");
        }
      });

  // 4. Evaluate Loops for Tiling.
  SmallVector<int64_t> chosenLoops;
  LDBG(">>> [Analysis] Loop Classification for Anchor:");

  for (size_t i = 0; i < loopToDSU.size(); ++i) {
    int64_t dsuId = loopToDSU[i];
    DimKind defaultKind = loopDefaultKinds[i];
    DimKind k = DimKind::Unknown;

    // Retrieve the globally propagated property for this dimension from the
    // DSU.
    if (dsuId != -1) {
      k = dsu_.getKind(dsuId);
    }

    // Fallback: If the loop isn't linked to any known data dimension or lacks
    // strong global properties, rely on the default local iterator type.
    if (k == DimKind::Unknown) {
      k = defaultKind;
    }

    LDBG("    Loop " << i << ": DSU ID = "
                     << (dsuId == -1 ? "None" : std::to_string(dsuId))
                     << ", Resolved Kind = " << toString(k));

    // Policy: We only auto-tile globally Parallel loops.
    // Future expansion: Support tiling Reduction loops if atomic updates are
    // enabled.
    if (k == DimKind::Parallel) {
      chosenLoops.push_back(i);
    }
  }

  // 5. Final Reporting.
  if (chosenLoops.empty()) {
    LDBG(">>> [Analysis] No suitable dimensions found for tiling.");
  } else {
    LDBG(">>> [Analysis] Chosen Tiling Dims: ["
         << llvm::join(
                llvm::map_range(chosenLoops,
                                [](int64_t v) { return std::to_string(v); }),
                ", ")
         << "]");
  }

  return chosenLoops;
}

void DimAnalyzer::runAnalysis() { processBFS(); }

std::optional<int64_t> DimAnalyzer::getDimRoot(Value v, int64_t dimIdx) {
  auto it = valueDims_.find(v);
  if (it == valueDims_.end())
    return std::nullopt;
  if (dimIdx < 0 || dimIdx >= (int64_t)it->second.size())
    return std::nullopt;
  return dsu_.find(it->second[dimIdx]);
}

DimKind DimAnalyzer::getDimKind(Value v, int64_t dimIdx) {
  auto it = valueDims_.find(v);
  if (it == valueDims_.end())
    return DimKind::Unknown;
  if (dimIdx < 0 || dimIdx >= (int64_t)it->second.size())
    return DimKind::Unknown;
  return dsu_.getKind(it->second[dimIdx]);
}

void DimAnalyzer::processBFS() {
  BFSQueue bfsQueue;
  VisitedSet visited;
  DenseSet<Value> definedInStage;

  // Identify all values defined within the stage to find boundary inputs.
  for (auto *op : inputOps)
    for (auto res : op->getResults())
      definedInStage.insert(res);

  // 1. Seeds: Operands used in stage but defined externally (Inputs).
  for (auto *op : inputOps) {
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
  if (bfsQueue.empty() && !inputOps.empty()) {
    for (auto res : inputOps[0]->getResults()) {
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
  // Dispatch to the most specific semantic handler first, then fall back to
  // progressively more generic tensor/buffer and elementwise behavior.
  if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op)) {
    processMatmulOp(matmulOp);
    return true;
  }
  if (auto reduceOp = dyn_cast<linalg::ReduceOp>(op)) {
    processReduceOp(reduceOp);
    return true;
  }
  if (auto transOp = dyn_cast<linalg::TransposeOp>(op)) {
    processTransposeOp(transOp);
    return true;
  }
  if (auto bcastOp = dyn_cast<linalg::BroadcastOp>(op)) {
    processBroadcastOp(bcastOp);
    return true;
  }
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    processLinalgOpGeneric(linalgOp);
    return true;
  }

  if (auto castOp = dyn_cast<tensor::CastOp>(op)) {
    bindDimensions(castOp.getSource(), castOp.getDest());
    return true;
  }
  if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
    processReshapeOp(expandOp);
    return true;
  }
  if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
    processReshapeOp(collapseOp);
    return true;
  }
  if (auto concatOp = dyn_cast<tensor::ConcatOp>(op)) {
    processConcatOp(concatOp);
    return true;
  }
  if (auto padOp = dyn_cast<tensor::PadOp>(op)) {
    processPadOp(padOp);
    return true;
  }
  if (auto extSlice = dyn_cast<tensor::ExtractSliceOp>(op)) {
    processExtractSliceOp(extSlice);
    return true;
  }
  if (auto insSlice = dyn_cast<tensor::InsertSliceOp>(op)) {
    processInsertSliceOp(insSlice);
    return true;
  }

  if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
    processMemrefCopyOp(copyOp, current, q, v);
    return true;
  }
  if (isa<memref::CastOp, memref::ReinterpretCastOp>(op)) {
    processMemrefCastOp(op);
    return true;
  }
  if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(op)) {
    processBufferizationToTensor(toTensor);
    return true;
  }
  if (auto matOp = dyn_cast<bufferization::MaterializeInDestinationOp>(op)) {
    processMaterializeOp(matOp, current, q, v);
    return true;
  }

  if (isa<func::CallOp>(op)) {
    processCallOp(op);
    return true;
  }
  if (isa<arith::ArithDialect, math::MathDialect>(op->getDialect())) {
    processElementwise(op, current);
    return true;
  }

  // Default fallback: assume 1-to-1 preservation if results exist.
  if (op->getNumResults() > 0)
    bindDimensions(current, op->getResult(0));
  return true;
}

//===----------------------------------------------------------------------===//
// Specific Handlers
//===----------------------------------------------------------------------===//

void DimAnalyzer::processMemrefCopyOp(memref::CopyOp op, Value current,
                                      BFSQueue &q, VisitedSet &v) {
  Value src = op.getSource();
  Value dst = op.getTarget();
  bindDimensions(src, dst);

  // Special Case: Copy sends data to 'dst', which is an operand (outs), not a
  // result. We must explicitly enqueue 'dst' to continue BFS.
  if (current == src) {
    if (v.insert(dst).second) {
      q.push(dst);
      getOrAllocateDims(dst);
    }
  }
}

void DimAnalyzer::processMaterializeOp(
    bufferization::MaterializeInDestinationOp op, Value current, BFSQueue &q,
    VisitedSet &v) {
  Value src = op.getSource();
  Value dst = op.getDest();
  bindDimensions(src, dst);

  // Similar to Copy: Propagate to destination buffer.
  if (current == src) {
    if (v.insert(dst).second) {
      q.push(dst);
      getOrAllocateDims(dst);
    }
  }
}

void DimAnalyzer::processMemrefCastOp(Operation *op) {
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
  // Converts MemRef to Tensor. Dimensions are strictly preserved.
  Value memrefValue = op.getOperand();
  Value tensorResult = op.getResult();
  bindDimensions(memrefValue, tensorResult);
}

void DimAnalyzer::processTransposeOp(linalg::TransposeOp op) {
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

template <typename ReshapeOp> void DimAnalyzer::processReshapeOp(ReshapeOp op) {
  constexpr bool isExpand = std::is_same_v<ReshapeOp, tensor::ExpandShapeOp>;
  Value input = op.getSrc();
  Value output = op.getResult();
  auto inputDims = getOrAllocateDims(input);
  auto outputDims = getOrAllocateDims(output);
  ArrayRef<int64_t> inputShape =
      cast<RankedTensorType>(input.getType()).getShape();
  ArrayRef<int64_t> outputShape =
      cast<RankedTensorType>(output.getType()).getShape();

  // Build symmetric reassociation index maps so that both sides have the same
  // number of groups. The reshaped side carries N-to-1 mappings; the other
  // side gets trivial identity indices {0}, {1}, ...
  SmallVector<ReassociationIndices> inputIndices, outputIndices;
  if constexpr (isExpand) {
    for (size_t i = 0; i < inputDims.size(); ++i)
      inputIndices.push_back({static_cast<int64_t>(i)});
    outputIndices = op.getReassociationIndices();
  } else {
    inputIndices = op.getReassociationIndices();
    for (size_t i = 0; i < outputDims.size(); ++i)
      outputIndices.push_back({static_cast<int64_t>(i)});
  }

  assert(inputIndices.size() == outputIndices.size() &&
         "Reassociation group count must match between input and output");

  for (const auto &[inputIdx, outputIdx] :
       llvm::zip_equal(inputIndices, outputIndices)) {

    // Case 1: Trivial identity group (1-to-1). Safe to bind directly.
    if (inputIdx.size() == 1 && outputIdx.size() == 1) {
      dsu_.unionSets(inputDims[inputIdx[0]], outputDims[outputIdx[0]]);
      continue;
    }

    // Case 2: Non-trivial reassociation group.
    // Filter out unit (size=1) dimensions — they carry no data and are
    // freely insertable/removable.
    auto filteredInputIdx = llvm::to_vector(llvm::make_filter_range(
        inputIdx, [&](int64_t idx) { return inputShape[idx] != 1; }));
    auto filteredOutputIdx = llvm::to_vector(llvm::make_filter_range(
        outputIdx, [&](int64_t idx) { return outputShape[idx] != 1; }));

    // Default: mark all output dims in this group as Complex (mutated).
    for (int64_t idx : outputIdx)
      dsu_.setKind(outputDims[idx], DimKind::Complex);

    // Special case: reshape only adds/removes unit dims.
    // e.g. [1, a, 1] -> [a] or [a] -> [a, 1]: the non-unit dim is preserved.
    if (filteredInputIdx.size() == 1 && filteredOutputIdx.size() == 1) {
      int64_t inIdx = filteredInputIdx[0];
      int64_t outIdx = filteredOutputIdx[0];
      dsu_.setKind(outputDims[outIdx], DimKind::Parallel);
      dsu_.unionSets(inputDims[inIdx], outputDims[outIdx]);
    } else {
      // Genuine multi-dim reshape — union all dims and keep Complex.
      int64_t groupRoot = inputDims[inputIdx[0]];
      for (int64_t idx : inputIdx)
        dsu_.unionSets(groupRoot, inputDims[idx]);
      for (int64_t idx : outputIdx)
        dsu_.unionSets(groupRoot, outputDims[idx]);
    }
  }
}

// Explicit template instantiations.
template void DimAnalyzer::processReshapeOp(tensor::ExpandShapeOp);
template void DimAnalyzer::processReshapeOp(tensor::CollapseShapeOp);

void DimAnalyzer::processElementwise(Operation *op, Value current) {
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
  // InsertSlice preserves the destination iteration space and forwards the
  // source slice dimensions into the corresponding result dimensions.
  bindDimensions(op.getDest(), op.getResult());

  auto srcDims = getOrAllocateDims(op.getSource());
  auto resDims = getOrAllocateDims(op.getResult());
  auto dropped = op.getSourceType().getRank() == op.getType().getRank()
                     ? llvm::SmallBitVector(resDims.size(), false)
                     : op.getDroppedDims();

  int srcIdx = 0;
  for (size_t resIdx = 0; resIdx < resDims.size(); ++resIdx) {
    if (dropped.test(resIdx))
      continue;
    if (srcIdx >= static_cast<int>(srcDims.size()))
      break;
    dsu_.unionSets(srcDims[srcIdx++], resDims[resIdx]);
  }
}

void DimAnalyzer::processCallOp(Operation *op) {
  // Function calls have opaque semantics — we cannot infer any relationship
  // between input and output dimensions.  Conservatively mark every result
  // dimension as Complex to prevent tiling across this boundary.
  for (Value result : op->getResults()) {
    auto dims = getOrAllocateDims(result);
    for (int64_t id : dims)
      dsu_.setKind(id, DimKind::Complex);
  }
}

void DimAnalyzer::processLinalgOpGeneric(linalg::LinalgOp op) {
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
