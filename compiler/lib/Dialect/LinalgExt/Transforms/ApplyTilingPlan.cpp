#include "dicp/Dialect/LinalgExt/Analysis/StageUtils.h"
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"
#include "dicp/TransformOps/DicpTransformOps.h"
#include "dicp/TransformOps/Transforms.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "apply-tiling-plan"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace dicp;
using namespace LinalgExt;
using namespace mlir::dicp::stage_attrs;

namespace mlir {
namespace dicp {
namespace LinalgExt {
#define GEN_PASS_DEF_APPLYTILINGPLAN
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace LinalgExt
} // namespace dicp
} // namespace mlir

namespace {

static bool hasProducerFusionAttr(Operation *op) {
  return op && (getProducerFuseTag(op).has_value() ||
                op->hasAttr(kStageProducerAllocToFuseAttr));
}

static bool hasFusionDrivingAttr(Operation *op) {
  return op && (getTileMeta(op) || getAnchorTileTag(op).has_value() ||
                hasProducerFusionAttr(op));
}

static void reportLegalityViolation(Operation *op, Twine message,
                                    StringRef debugReason, bool &hasViolation) {
  hasViolation = true;
  op->emitError(message);
  LDBG("[Legality] " << debugReason << ": " << *op);
}

static LogicalResult validateVectorTileTransformPreconditions(ModuleOp module) {
  bool hasViolation = false;
  LDBG("[Legality] run vector-tile transform precheck");

  module.walk([&](Operation *op) {
    if (isControlFlowOp(op) && hasFusionDrivingAttr(op)) {
      reportLegalityViolation(
          op,
          "control-flow operations must not carry fusion-driving "
          "stage-planning attributes",
          "control-flow op carries fusion-driving attrs", hasViolation);
      return;
    }

    if (isa<BranchOpInterface>(op) && getStageId(op).has_value()) {
      reportLegalityViolation(
          op, "BranchOpInterface operations must not carry `dicp.tmp.stage.id`",
          "branch op carries stage id", hasViolation);
      return;
    }

    if (getStageId(op).has_value()) {
      if (isa<memref::SubViewOp, tensor::ExtractSliceOp>(op)) {
        reportLegalityViolation(
            op,
            "memref.subview and tensor.extract_slice must not carry "
            "`dicp.tmp.stage.id`; fusion for these ops is unsupported",
            "slice-like op carries stage id", hasViolation);
        return;
      }
    }
  });

  if (hasViolation) {
    LDBG("[Legality] precheck failed");
    return failure();
  }

  LDBG("[Legality] precheck passed");
  return success();
}

//===----------------------------------------------------------------------===//
// Data Structures
//===----------------------------------------------------------------------===//

/// Extracted metadata from the tagging pass.
struct TilingMeta {
  StringRef anchorTag;
  StringRef computeTag;
  SmallVector<int64_t> tileSizes;
};

//===----------------------------------------------------------------------===//
// TransformGenerator
//===----------------------------------------------------------------------===//

/// Generates Transform Dialect IR based on the analyzed tiling units.
class TransformGenerator {
public:
  TransformGenerator(OpBuilder &b, Location loc, Value root)
      : b(b), loc(loc), root(root) {}

  /// Process all tiling metadata extracted from the module.
  void generate(ArrayRef<TilingMeta> metas) {
    for (const auto &meta : metas) {
      generateUnit(meta);
    }
  }

private:
  OpBuilder &b;
  Location loc;
  Value root;

  /// Emit tiling and fusion sequence for a single unit.
  void generateUnit(const TilingMeta &meta) {
    LDBG("[TransformGen] emit unit for anchor tag " << meta.anchorTag);
    Value anchor = getMatch(meta.anchorTag);

    auto tile = b.create<transform::TileUsingForallOp>(
        loc, anchor, meta.tileSizes, transform::TileSizesSpec(), nullptr);
    Value loop = tile.getForallOp();

    Value producers = getMatch(meta.computeTag, true);

    auto foreachOp = b.create<transform::ForeachOp>(
        loc,
        TypeRange{b.getType<transform::AnyOpType>(),
                  b.getType<transform::AnyOpType>()},
        producers, false);

    OpBuilder::InsertionGuard g(b);
    b.createBlock(&foreachOp.getBody(), {}, {b.getType<transform::AnyOpType>()},
                  {loc});

    ImplicitLocOpBuilder ib(loc, b);
    // Correct number of args for fusion
    auto fused = ib.create<transform::ExtendedFuseIntoContainingOp>(
        ib.getType<transform::AnyOpType>(), ib.getType<transform::AnyOpType>(),
        foreachOp.getBody().getArgument(0), loop);

    auto apply = ib.create<transform::ApplyPatternsOp>(
        fused.getNewContainingOp().front(), [](OpBuilder &pb, Location ploc) {
          pb.create<transform::ApplyCanonicalizationPatternsOp>(ploc);
        });
    apply.setApplyCse(true);
    ib.create<transform::YieldOp>(fused.getResults());
  }

  /// Helper to create a transform.match op for a given attribute tag.
  Value getMatch(StringRef attr, bool reverse = false) {
    auto match = b.create<transform::MatchOp>(
        loc, b.getType<transform::AnyOpType>(), root, nullptr, nullptr,
        b.getDictionaryAttr(b.getNamedAttr(attr, b.getUnitAttr())), nullptr,
        nullptr);
    return reverse ? b.create<transform::ReverseOp>(
                          loc, b.getType<transform::AnyOpType>(), match)
                         .getResult()
                   : match.getResult();
  }
};

//===----------------------------------------------------------------------===//
// Pass Main Entry
//===----------------------------------------------------------------------===//

/// Main pass for applying the tiling and fusion sequence via Transform Dialect.
class ApplyTilingPlanPass
    : public mlir::dicp::LinalgExt::impl::ApplyTilingPlanBase<
          ApplyTilingPlanPass> {
public:
  using ApplyTilingPlanBase::ApplyTilingPlanBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    LDBG("[Pass] run ApplyTilingPlanPass");

    if (failed(validateVectorTileTransformPreconditions(module))) {
      signalPassFailure();
      return;
    }

    // 1. Collect metadata from the tagging pass and clean up attributes.
    SmallVector<TilingMeta> metaList;
    module.walk([&](Operation *op) {
      if (auto tileMetaAttr = op->getAttrOfType<DictionaryAttr>(kTileMetaTag)) {
        LDBG("Tile Meta Op: " << *op);
        TilingMeta meta;
        meta.anchorTag =
            cast<StringAttr>(tileMetaAttr.get(kTileMetaAnchorTag)).getValue();
        meta.computeTag =
            cast<StringAttr>(tileMetaAttr.get(kTileMetaFuseTag)).getValue();
        auto sizesAttr =
            cast<DenseI64ArrayAttr>(tileMetaAttr.get(kTileMetaTileSizesTag));
        meta.tileSizes = llvm::to_vector(sizesAttr.asArrayRef());
        metaList.push_back(meta);

        LDBG("[Pass] found tiling meta for anchor " << meta.anchorTag);
        // Clean up the metadata attribute to avoid polluting final IR.
      }
    });

    if (metaList.empty()) {
      LDBG("[Pass] no tiling metadata found, skip transform application");
      return;
    }

    LDBG("[Pass] collected " << metaList.size() << " tiling metas");
    // 2. Apply transformation via Transform Dialect interpreter.
    LDBG("[Pass] apply Transform Dialect generation for " << metaList.size()
                                                          << " tiling units");
    TransformApplier::apply(
        module, [&](OpBuilder &b, Location loc, Value root) {
          TransformGenerator(b, loc, root).generate(metaList);
        });

    // 3. Propagate kNPUStageAttrName from loop body to the loop op itself.
    // This runs AFTER cleanup so that attributes survive canonicalization.
    // The current propagation rule remains conservative: only loop-body ops
    // that still carry kTileMetaTag are used as the source of the stage id.
    module.walk([&](LoopLikeOpInterface loop) {
      Operation *loopOp = loop.getOperation();
      if (loopOp->hasAttr(kNPUStageAttrName))
        return;
      for (Operation &op : loopOp->getRegion(0).front()) {
        if (op.hasAttr(kTileMetaTag)) {
          auto attr = op.getAttrOfType<IntegerAttr>(kNPUStageAttrName);
          LDBG("[Pass] propagate stage id " << attr.getInt() << " to loop "
                                            << *loopOp);
          loopOp->setAttr(kNPUStageAttrName, attr);
          break;
        }
      }
    });

    LDBG("[Pass] ApplyTilingPlanPass completed successfully");
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
mlir::dicp::LinalgExt::createApplyTilingPlanPass() {
  return std::make_unique<ApplyTilingPlanPass>();
}
