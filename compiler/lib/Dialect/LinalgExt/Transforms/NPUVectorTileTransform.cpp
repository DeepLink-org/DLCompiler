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
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "npu-tile-loop-transform"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace dicp;
using namespace LinalgExt;

namespace mlir {
namespace dicp {
namespace LinalgExt {
#define GEN_PASS_DEF_NPUVECTORTILETRANSFORM
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace LinalgExt
} // namespace dicp
} // namespace mlir

namespace {

//===----------------------------------------------------------------------===//
// Data Structures
//===----------------------------------------------------------------------===//

/// Extracted metadata from the tagging pass.
struct TilingMeta {
  std::string anchorTag;
  std::string computeTag;
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
    LDBG("generateUnit: Emitting transform sequence for anchor tag: "
         << meta.anchorTag);
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
class NPUVectorTileTransformPass
    : public mlir::dicp::LinalgExt::impl::NPUVectorTileTransformBase<
          NPUVectorTileTransformPass> {
public:
  using NPUVectorTileTransformBase::NPUVectorTileTransformBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    LDBG("Running NPUVectorTileTransformPass...");

    // 1. Collect metadata from the tagging pass and clean up attributes.
    SmallVector<TilingMeta> metaList;
    module.walk([&](Operation *op) {
      if (auto dict = op->getAttrOfType<DictionaryAttr>("npu.tiling_meta")) {
        TilingMeta meta;
        meta.anchorTag =
            cast<StringAttr>(dict.get("anchor_tag")).getValue().str();
        meta.computeTag =
            cast<StringAttr>(dict.get("compute_tag")).getValue().str();
        auto sizesAttr = cast<DenseI64ArrayAttr>(dict.get("tile_sizes"));
        meta.tileSizes = llvm::to_vector(sizesAttr.asArrayRef());
        metaList.push_back(meta);

        LDBG("Found tiling meta for anchor: " << meta.anchorTag);
        // Clean up the metadata attribute to avoid polluting final IR.
        op->removeAttr("npu.tiling_meta");
      }
    });

    if (metaList.empty()) {
      LDBG("No tiling metadata found in the module. Skipping transform "
           "application.");
      return;
    }

    // 2. Apply transformation via Transform Dialect interpreter.
    LDBG("Applying Transform Dialect generation for " << metaList.size()
                                                      << " tiling units.");
    TransformApplier::apply(
        module, [&](OpBuilder &b, Location loc, Value root) {
          TransformGenerator(b, loc, root).generate(metaList);
        });

    // 3. Final IR cleanup.
    LDBG("Running final cleanup pipeline (CSE/Canonicalizer).");
    PassManager pm(ctx, module.getOperationName());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pm, module))) {
      LDBG("Final cleanup pipeline failed.");
      signalPassFailure();
    }

    LDBG("NPUVectorTileTransformPass completed successfully.");
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
mlir::dicp::LinalgExt::createNPUVectorTileTransformPass() {
  return std::make_unique<NPUVectorTileTransformPass>();
}