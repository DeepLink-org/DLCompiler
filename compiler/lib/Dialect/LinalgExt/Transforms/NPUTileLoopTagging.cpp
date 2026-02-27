#include "dicp/Dialect/LinalgExt/Analysis/DimAnalyzer.h"
#include "dicp/Dialect/LinalgExt/Analysis/StageDependencyAnalyzer.h"
#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"
#include "dicp/TransformOps/DicpTransformOps.h"
#include "dicp/TransformOps/Transforms.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "npu-tile-loop-tagging"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace dicp;
using namespace LinalgExt;

namespace mlir {
namespace dicp {
namespace LinalgExt {
#define GEN_PASS_DEF_NPUVECTORTILETAGGING
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace LinalgExt
} // namespace dicp
} // namespace mlir

namespace {

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Check if the operation is a candidate for elementwise-to-generic conversion.
static bool isConvertibleElementwiseOp(Operation *op) {
  if (!op->hasAttr(kNPUStageAttrName) || isa<linalg::GenericOp>(op))
    return false;
  return OpTrait::hasElementwiseMappableTraits(op) &&
         llvm::all_of(op->getOperandTypes(), llvm::IsaPred<RankedTensorType>);
}

/// Create empty tensors for linalg outputs if matching operands aren't found.
static SmallVector<Value>
getOrCreateOperandsMatchingResultTypes(OpBuilder &b, Operation *op) {
  Location loc = op->getLoc();
  ValueRange operands = op->getOperands();
  return llvm::map_to_vector(op->getResultTypes(), [&](Type t) -> Value {
    auto it =
        llvm::find_if(operands, [&](Value v) { return v.getType() == t; });
    if (it != operands.end())
      return *it;
    LDBG("getOrCreateOperandsMatchingResultTypes: Creating empty tensor for "
         "type "
         << t);
    return b.create<tensor::EmptyOp>(
        loc, tensor::getMixedSizes(b, loc, operands.front()),
        cast<RankedTensorType>(t).getElementType());
  });
}

/// Retrieve the loop or tensor rank of an operation.
static int64_t getRank(Operation *op) {
  return TypeSwitch<Operation *, int64_t>(op)
      .Case<linalg::LinalgOp>(
          [](auto linalgOp) { return linalgOp.getNumLoops(); })
      .Default([](Operation *op) -> int64_t {
        if (op->getNumResults() > 0)
          if (auto type = dyn_cast<ShapedType>(op->getResult(0).getType()))
            return type.getRank();
        return 0;
      });
}

/// Calculate tile size based on the target trip count.
static int64_t calculateTileSize(Operation *anchorOp, int64_t dimIdx,
                                 int64_t tripCount) {
  if (tripCount <= 0) {
    LDBG("calculateTileSize: tripCount is invalid: " << tripCount);
    return -1;
  }
  auto getDimSize = [&](Value v) -> int64_t {
    auto type = dyn_cast_or_null<ShapedType>(v.getType());
    if (!type || dimIdx >= type.getRank())
      return -1;
    return type.getDimSize(dimIdx);
  };
  int64_t totalSize =
      TypeSwitch<Operation *, int64_t>(anchorOp)
          .Case<linalg::CopyOp>(
              [&](auto op) { return getDimSize(op.getInputs().front()); })
          .Case<linalg::LinalgOp>([&](auto op) {
            return op.getDpsInits().empty()
                       ? -1
                       : getDimSize(op.getDpsInits().front());
          })
          .Default([&](auto op) {
            return op->getNumResults() > 0 ? getDimSize(op->getResult(0)) : -1;
          });

  if (totalSize <= 0) {
    LDBG("calculateTileSize: Could not determine total size for dim "
         << dimIdx << " on op " << anchorOp->getName());
    return -1;
  }
  if (totalSize % tripCount != 0) {
    LDBG("calculateTileSize: Total size "
         << totalSize << " is not divisible by tripCount " << tripCount);
    return -1;
  }
  return totalSize / tripCount;
}

//===----------------------------------------------------------------------===//
// Data Structures
//===----------------------------------------------------------------------===//

/// Metadata for a single tiling group (anchor and fused producers).
struct TilingUnit {
  Operation *anchorOp = nullptr;
  SmallVector<int64_t> tileSizes;
  int64_t tilingDimIndex = -1;
  int64_t rank = 0;
  std::vector<Operation *> producerOps;

  std::string anchorTag;
  std::string producerComputeTag;
  std::string producerAllocTag;
  std::string crossUserTag;
};

/// Represents a sub-stage containing multiple tiling units.
struct TiledSubStage {
  SubStage base;
  std::vector<TilingUnit> units;
  explicit TiledSubStage(SubStage s) : base(std::move(s)) {}
};

//===----------------------------------------------------------------------===//
// Normalization Patterns
//===----------------------------------------------------------------------===//

/// Sink bufferization.to_tensor immediately after its alloc operand.
struct SinkToTensorToAlloc
    : public OpRewritePattern<bufferization::ToTensorOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(bufferization::ToTensorOp op,
                                PatternRewriter &rewriter) const override {
    Operation *allocOp = op.getOperand().getDefiningOp();
    if (!isa_and_nonnull<memref::AllocOp>(allocOp) ||
        op->getPrevNode() == allocOp)
      return failure();
    LDBG("SinkToTensorToAlloc: Sinking to_tensor after alloc " << *allocOp);
    rewriter.modifyOpInPlace(op, [&]() { op->moveAfter(allocOp); });
    return success();
  }
};

/// Convert elementwise operations to linalg.generic to enable tiling.
struct ConvertElementwiseToGenericPattern : public RewritePattern {
  ConvertElementwiseToGenericPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isConvertibleElementwiseOp(op))
      return failure();
    LDBG("ConvertElementwiseToGenericPattern: Converting op " << op->getName());
    auto rank = getRank(op);
    SmallVector<AffineMap> maps(op->getNumResults() + op->getNumOperands(),
                                rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iterTypes(rank,
                                               utils::IteratorType::parallel);
    auto outputs = getOrCreateOperandsMatchingResultTypes(rewriter, op);
    auto genericOp = rewriter.create<linalg::GenericOp>(
        op->getLoc(), op->getResultTypes(), op->getOperands(), outputs, maps,
        iterTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
          auto resTypes = llvm::map_to_vector(op->getResultTypes(), [](Type t) {
            return cast<TensorType>(t).getElementType();
          });
          auto *scalarOp = b.create(loc, op->getName().getIdentifier(),
                                    args.take_front(op->getNumOperands()),
                                    resTypes, op->getAttrs());
          scalarOp->removeAttr(kNPUStageAttrName);
          b.create<linalg::YieldOp>(loc, scalarOp->getResults());
        });
    if (auto attr = op->getAttr(kNPUStageAttrName))
      genericOp->setAttr(kNPUStageAttrName, attr);
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

/// Lower various copy ops to linalg.copy for unified tiling treatment.
template <typename OpType>
struct ConvertCopyLikeOpToLinalgCopy : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    auto stageAttr = op->template getAttrOfType<IntegerAttr>(kNPUStageAttrName);
    if (!stageAttr)
      return failure();

    LDBG("ConvertCopyLikeOpToLinalgCopy: Converting " << op->getName());
    Value src, dst;
    if constexpr (std::is_same_v<OpType,
                                 bufferization::MaterializeInDestinationOp>) {
      src = op.getSource();
      dst = op.getDest();
    } else {
      src = op.getSource();
      dst = op.getTarget();
    }
    auto toMemref = [&](Value v) -> Value {
      if (!isa<RankedTensorType>(v.getType()))
        return v;
      auto type =
          MemRefType::get(cast<RankedTensorType>(v.getType()).getShape(),
                          cast<RankedTensorType>(v.getType()).getElementType());
      auto res =
          rewriter.create<bufferization::ToBufferOp>(op.getLoc(), type, v);
      res->setAttr(kNPUStageAttrName, stageAttr);
      return res;
    };
    auto copy = rewriter.create<linalg::CopyOp>(op.getLoc(), toMemref(src),
                                                toMemref(dst));
    copy->setAttrs(op->getAttrs());
    copy->setAttr(kOriginalOpNameAttr,
                  rewriter.getStringAttr(op->getName().getStringRef()));
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TilingProcessor
//===----------------------------------------------------------------------===//

/// Analyzes stages to identify tiling anchors and assign unique tags.
class TilingProcessor {
public:
  explicit TilingProcessor(int64_t tripCount, bool enableFallback = true)
      : tripCount(tripCount), enableFallback(enableFallback) {}

  /// Analyze the sub-stage and tag operations for subsequent transformation.
  LogicalResult analyzeAndTag(TiledSubStage &ts) const {
    LDBG("[TilingProcessor] Analyzing SubStage "
         << ts.base.index << " in Stage " << ts.base.stageId);
    auto candidates = collectCandidates(ts.base);

    auto dims = analyzeDims(ts.base);
    if (!dims) {
      LDBG("  [Warning] Dim analysis failed for SubStage " << ts.base.index);
      return failure();
    }
    LDBG("  [Info] Analyzed tiling dims: " << dims->front());

    SetVector<Operation *> claimedOps;

    // 1. Process high-priority candidates
    if (!candidates.empty()) {
      for (auto &cand : candidates) {
        if (claimedOps.contains(cand.op)) {
          LDBG("  [Skip] Candidate " << cand.op->getName()
                                     << " already claimed by previous unit.");
          continue;
        }

        llvm::SetVector<Operation *> slice;
        BackwardSliceOptions opt;
        opt.omitBlockArguments = true;
        opt.filter = [&](Operation *op) {
          return llvm::is_contained(ts.base.ops, op) &&
                 op->getBlock() == cand.op->getBlock();
        };
        (void)getBackwardSlice(cand.op, &slice, opt);
        slice.remove(cand.op);

        SmallVector<Operation *> producers;
        for (Operation *p : slice) {
          if (!claimedOps.contains(p))
            producers.push_back(p);
        }

        if (auto unit = tryCreateUnit(cand.op, producers, *dims, ts)) {
          tagUnit(*unit);
          claimedOps.insert(cand.op);
          claimedOps.insert(producers.begin(), producers.end());
          ts.units.push_back(std::move(*unit));
          LDBG("  [Success] Created Unit "
               << ts.units.size() - 1 << " with Anchor " << cand.op->getName());
        } else {
          LDBG("  [Fail] Could not create unit for anchor "
               << cand.op->getName());
        }
      }
    } else {
      LDBG("  [Info] No high-priority tiling candidates found.");
    }

    // 2. Process Fallback: Cover remaining ops if enabled
    if (enableFallback) {
      processFallback(ts, claimedOps, *dims);
    }

    return ts.units.empty() ? failure() : success();
  }

private:
  int64_t tripCount;
  bool enableFallback;

  enum class Priority { Normalized = 1, Yield = 2, Fallback = 3 };
  struct Candidate {
    Operation *op;
    Priority prio;
    size_t irIdx;
  };

  /// Identify dimensions suitable for tiling.
  std::optional<SmallVector<int64_t>> analyzeDims(const SubStage &ss) const {
    StageInfo info{ss.stageId, ss.ops};
    auto dims = DimAnalyzer(info).analyzeAndGetTilingDims();
    if (dims.empty())
      return std::nullopt;
    std::sort(dims.begin(), dims.end());
    return dims;
  }

  /// Collect potential anchor operations for tiling, sorted by priority.
  std::vector<Candidate> collectCandidates(const SubStage &ss) const {
    DenseMap<Operation *, Candidate> best;
    auto update = [&](Operation *o, Priority p, size_t i) {
      auto &entry = best[o];
      if (!entry.op || p < entry.prio || (p == entry.prio && i < entry.irIdx))
        entry = {o, p, i};
    };

    for (auto [i, op] : llvm::enumerate(ss.ops)) {
      if (auto copy = dyn_cast<linalg::CopyOp>(op))
        if (copy->hasAttr(kOriginalOpNameAttr))
          update(op, Priority::Normalized, i);
    }

    if (!ss.ops.empty()) {
      if (auto yield = dyn_cast<scf::YieldOp>(
              ss.ops.back()->getBlock()->getTerminator())) {
        for (Value v : yield.getOperands()) {
          Operation *def = v.getDefiningOp();
          if (!def || !llvm::is_contained(ss.ops, def))
            continue;

          bool feedsNorm = llvm::any_of(def->getUsers(), [](Operation *u) {
            return isa<linalg::CopyOp>(u) && u->hasAttr(kOriginalOpNameAttr);
          });
          if (isa<TilingInterface>(def) && !feedsNorm) {
            auto it = llvm::find(ss.ops, def);
            update(def, Priority::Yield, std::distance(ss.ops.begin(), it));
          }
        }
      }
    }

    for (int i = ss.ops.size() - 1; i >= 0; --i) {
      if (isa<TilingInterface>(ss.ops[i])) {
        update(ss.ops[i], Priority::Fallback, i);
        break;
      }
    }

    std::vector<Candidate> res;
    for (auto &kv : best)
      res.push_back(kv.second);
    llvm::sort(res, [](const Candidate &a, const Candidate &b) {
      return std::tie(a.prio, a.irIdx) < std::tie(b.prio, b.irIdx);
    });
    LDBG("collectCandidates: Found " << res.size() << " potential anchors.");
    return res;
  }

  /// Attempt to construct a tiling unit for a given anchor.
  std::optional<TilingUnit> tryCreateUnit(Operation *anchor,
                                          ArrayRef<Operation *> producers,
                                          ArrayRef<int64_t> dims,
                                          const TiledSubStage &ts) const {
    int64_t dimIdx = dims.front();
    int64_t tileSize = calculateTileSize(anchor, dimIdx, tripCount);
    if (tileSize <= 0) {
      LDBG("tryCreateUnit: Invalid tile size calculated for "
           << anchor->getName());
      return std::nullopt;
    }

    TilingUnit u;
    u.anchorOp = anchor;
    u.tilingDimIndex = dimIdx;
    u.tileSizes = {tileSize};
    u.rank = getRank(anchor);
    u.producerOps.assign(producers.begin(), producers.end());

    auto fmt = [&](StringRef pattern) {
      return llvm::formatv(pattern.data(), ts.base.stageId, ts.base.index,
                           ts.units.size())
          .str();
    };
    u.anchorTag = fmt(kStageOpToTileAttr);
    u.producerComputeTag = fmt(kStageProducerToFuseAttr);
    u.producerAllocTag = kStageProducerAllocToFuseAttr.str();
    u.crossUserTag = kCrossTillUnitAttr.str();
    return u;
  }

  /// Attach string attributes to operations to guide the transform dialect.
  /// This also packs the necessary tile sizes into a dictionary attribute
  /// to pass data elegantly to the subsequent Transform pass.
  void tagUnit(TilingUnit &u) const {
    MLIRContext *ctx = u.anchorOp->getContext();
    OpBuilder b(ctx);

    LDBG("tagUnit: Tagging anchor " << u.anchorOp->getName() << " with "
                                    << u.anchorTag);
    // Mark the anchor with a UnitAttr for the Transform pass to MatchOp against
    u.anchorOp->setAttr(u.anchorTag, b.getUnitAttr());

    // Calculate full tile sizes array
    SmallVector<int64_t> sizes(u.rank, 0);
    if (u.tilingDimIndex >= 0 && u.tilingDimIndex < u.rank &&
        !u.tileSizes.empty()) {
      sizes[u.tilingDimIndex] = u.tileSizes.front();
    }

    // Embed metadata into a dictionary attribute to safely pass to the
    // Transform pass
    SmallVector<NamedAttribute> meta;
    meta.push_back(b.getNamedAttr("anchor_tag", b.getStringAttr(u.anchorTag)));
    meta.push_back(
        b.getNamedAttr("compute_tag", b.getStringAttr(u.producerComputeTag)));
    meta.push_back(b.getNamedAttr("tile_sizes", b.getDenseI64ArrayAttr(sizes)));
    u.anchorOp->setAttr("npu.tiling_meta", b.getDictionaryAttr(meta));
    LDBG("tagUnit: Attached npu.tiling_meta to anchor.");

    SetVector<Operation *> currentUnitOps(u.producerOps.begin(),
                                          u.producerOps.end());
    currentUnitOps.insert(u.anchorOp);

    for (Operation *p : u.producerOps) {
      if (llvm::any_of(p->getUsers(), [&](Operation *user) {
            return !currentUnitOps.contains(user);
          })) {
        LDBG("tagUnit: Producer "
             << p->getName()
             << " has users outside unit, tagging as cross-user.");
        p->setAttr(u.crossUserTag, b.getUnitAttr());
      }

      bool isMem =
          TypeSwitch<Operation *, bool>(p)
              .Case<tensor::EmptyOp, memref::AllocOp, memref::AllocaOp>(
                  [](auto) { return true; })
              .Case<bufferization::ToTensorOp>([](auto op) {
                return isa_and_nonnull<tensor::EmptyOp, memref::AllocOp,
                                       memref::AllocaOp>(
                    op.getOperand().getDefiningOp());
              })
              .Default(false);

      StringRef tag = isMem ? u.producerAllocTag : u.producerComputeTag;
      LDBG("tagUnit: Tagging producer " << p->getName() << " with " << tag);
      p->setAttr(tag, b.getUnitAttr());
    }
  }

  /// Scans for remaining operations in the substage that haven't been claimed
  /// by any tiling unit. It traverses backwards to find the last valid
  /// operation (lowest priority) and creates a fallback tiling unit.
  void processFallback(TiledSubStage &ts, SetVector<Operation *> &claimedOps,
                       ArrayRef<int64_t> dims) const {
    LDBG("processFallback: Checking for uncovered operations in SubStage "
         << ts.base.index);

    for (Operation *op : llvm::reverse(ts.base.ops)) {
      if (claimedOps.contains(op))
        continue;

      if (!isa<TilingInterface>(op))
        continue;

      LDBG("processFallback: Found candidate anchor: " << op->getName());

      llvm::SetVector<Operation *> slice;
      BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      opt.filter = [&](Operation *p) {
        return llvm::is_contained(ts.base.ops, p) &&
               p->getBlock() == op->getBlock();
      };
      (void)getBackwardSlice(op, &slice, opt);
      slice.remove(op);

      SmallVector<Operation *> producers;
      for (Operation *p : slice) {
        if (!claimedOps.contains(p))
          producers.push_back(p);
      }

      if (auto unit = tryCreateUnit(op, producers, dims, ts)) {
        tagUnit(*unit);
        claimedOps.insert(op);
        claimedOps.insert(producers.begin(), producers.end());
        ts.units.push_back(std::move(*unit));
        LDBG("processFallback: Successfully created fallback Unit "
             << ts.units.size() - 1 << " with Anchor " << op->getName());
        break;
      } else {
        LDBG("processFallback: Failed to create unit for " << op->getName());
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// Pass Main Entry
//===----------------------------------------------------------------------===//

/// Normalization and Tagging Pass for NPU loop tiling.
class NPUVectorTileTaggingPass
    : public mlir::dicp::LinalgExt::impl::NPUVectorTileTaggingBase<
          NPUVectorTileTaggingPass> {
public:
  using NPUVectorTileTaggingBase::NPUVectorTileTaggingBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    Block *targetBlock = CubeVectorSplitter::findTargetBlock(
        *(module.getOps<mlir::func::FuncOp>().begin()));
    llvm::SmallVector<Stage,4> stages;
    (void)CubeVectorSplitter::splitBlock(*targetBlock, stages);
    
    return;

    int64_t tripCount = static_cast<int64_t>(tiledMixVectorLoopNumber);
    LDBG("Run NPUVectorTileTaggingPass with tripCount=" << tripCount);

    auto blocks = StagePartitioner::findBlocksWithHivmSyncOps(module);
    if (blocks.empty()) {
      LDBG("No blocks with HIVM sync ops found. Skipping pass.");
      return;
    }

    bool anyStage = false;
    for (Block *b : blocks) {
      if (failed(StagePartitioner::analyzeAndTagBlock(b, ctx, anyStage))) {
        LDBG("StagePartitioner failed to analyze/tag block.");
        return;
      }
    }
    if (!anyStage) {
      LDBG("No stages identified in the module. Skipping.");
      return;
    }

    // Normalize operations to a tileable form.
    RewritePatternSet patterns(ctx);
    patterns.add<SinkToTensorToAlloc, ConvertElementwiseToGenericPattern,
                 ConvertCopyLikeOpToLinalgCopy<
                     bufferization::MaterializeInDestinationOp>,
                 ConvertCopyLikeOpToLinalgCopy<memref::CopyOp>>(ctx);
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      LDBG("Greedy pattern rewrite for normalization failed.");
      return;
    }

    // Analyze stages and assign transform tags.
    TilingProcessor processor(tripCount);
    for (Block *block : blocks) {
      for (int stageId : StagePartitioner::getStageIdsInBlock(block)) {
        for (auto &subStage :
             StagePartitioner::partition(block, stageId).subStages) {
          TiledSubStage ts(std::move(subStage));
          if (failed(processor.analyzeAndTag(ts))) {
            LDBG("Processor failed to analyze SubStage "
                 << ts.base.index << " of Stage " << stageId);
          }
        }
      }
    }

    LDBG("NPUVectorTileTaggingPass completed successfully.");
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
mlir::dicp::LinalgExt::createNPUVectorTileTaggingPass(
    const NPUVectorTileTaggingOptions &options) {
  return std::make_unique<NPUVectorTileTaggingPass>(options);
}

std::unique_ptr<OperationPass<mlir::ModuleOp>>
mlir::dicp::LinalgExt::createNPUVectorTileTaggingPass(unsigned vectorTile) {
  NPUVectorTileTaggingOptions opt;
  opt.tiledMixVectorLoopNumber = vectorTile;
  return std::make_unique<NPUVectorTileTaggingPass>(opt);
}