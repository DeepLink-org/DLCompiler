#include "triton/Dialect/Gluon/metax/Transforms/LayoutCandidates.h"

#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "mlir/IR/Diagnostics.h"
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <initializer_list>
#include <limits>

#define DEBUG_TYPE "gluon-mma-layout-candidates"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[gluon-layout-autotune][mma] " << X << "\n")

namespace ttg = mlir::triton::gpu;
namespace candidate = mlir::triton::gluon::layout_autotune;

namespace mlir::triton::gluon {

#define GEN_PASS_DEF_GLUONMMALAYOUTCANDIDATEPASS
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h.inc"

namespace {

ttg::MACAMmaEncodingAttr getMmaParent(Attribute encoding) {
  if (auto mma = dyn_cast_or_null<ttg::MACAMmaEncodingAttr>(encoding))
    return mma;
  if (auto dot = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(encoding))
    return dyn_cast<ttg::MACAMmaEncodingAttr>(dot.getParent());
  return {};
}

bool usesMma(DotOp dot, ttg::MACAMmaEncodingAttr mma) {
  return llvm::any_of(dot->getOperandTypes(), [&](Type type) {
           auto tensor = dyn_cast<RankedTensorType>(type);
           return tensor && getMmaParent(tensor.getEncoding()) == mma;
         }) || getMmaParent(cast<RankedTensorType>(dot.getType()).getEncoding()) ==
                    mma;
}

bool isDivisibleBy(int64_t extent, std::initializer_list<unsigned> factors) {
  if (extent <= 0)
    return false;
  uint64_t quotient = static_cast<uint64_t>(extent);
  for (unsigned factor : factors) {
    if (factor == 0 || quotient % factor != 0)
      return false;
    quotient /= factor;
  }
  return true;
}

bool isGeometryLegal(DotOp dot, ttg::MACAMmaEncodingAttr mma) {
  auto a = dyn_cast<RankedTensorType>(dot.getA().getType());
  auto b = dyn_cast<RankedTensorType>(dot.getB().getType());
  auto d = dyn_cast<RankedTensorType>(dot.getD().getType());
  if (!a || !b || !d || a.getRank() != 2 || b.getRank() != 2 ||
      d.getRank() != 2 || mma.getWarpsPerCTA().size() != 2 ||
      mma.getElementsMNK().size() != 3)
    return false;

  ArrayRef<unsigned> warps = mma.getWarpsPerCTA();
  ArrayRef<unsigned> elements = mma.getElementsMNK();
  SmallVector<int> atom = ttg::getMmaThreadShape(/*needTrans=*/false);
  if (atom.size() != 3 || llvm::any_of(atom, [](int value) { return value <= 0; }))
    return false;

  return isDivisibleBy(d.getShape()[0],
                       {warps[0], static_cast<unsigned>(atom[0]), elements[0]}) &&
         isDivisibleBy(d.getShape()[1],
                       {warps[1], static_cast<unsigned>(atom[1]), elements[1]}) &&
         isDivisibleBy(a.getShape()[1],
                       {static_cast<unsigned>(atom[2]), elements[2]}) &&
         isDivisibleBy(b.getShape()[0],
                       {static_cast<unsigned>(atom[2]), elements[2]});
}

FailureOr<ttg::MACAMmaEncodingAttr>
buildMmaNeighbor(FuncOp function, ttg::MACAMmaEncodingAttr baseline,
                 ArrayRef<unsigned> warps, ArrayRef<unsigned> elements,
                 unsigned colMajor) {
  auto candidate = ttg::MACAMmaEncodingAttr::get(
      function.getContext(), baseline.getVersionMajor(),
      baseline.getVersionMinor(), warps, elements, colMajor,
      baseline.getCTALayout(), baseline.getIsATrans(), baseline.getIsBTrans(),
      baseline.getElementsStride());
  if (!candidate)
    return failure();

  bool legal = true;
  function.walk([&](DotOp dot) {
    if (!usesMma(dot, baseline))
      return;
    if (!isGeometryLegal(dot, candidate)) {
      legal = false;
      return;
    }

    for (Type type : dot->getOperandTypes()) {
      auto tensor = dyn_cast<RankedTensorType>(type);
      if (!tensor || getMmaParent(tensor.getEncoding()) != baseline)
        continue;
      Attribute encoding = candidate;
      if (auto operand =
              dyn_cast<ttg::DotOperandEncodingAttr>(tensor.getEncoding()))
        encoding = ttg::DotOperandEncodingAttr::get(
            function.getContext(), operand.getOpIdx(), candidate,
            operand.getKWidth());
      (void)ttg::toLinearLayout(tensor.cloneWithEncoding(encoding));
    }
    auto result = cast<RankedTensorType>(dot.getType());
    if (getMmaParent(result.getEncoding()) == baseline)
      (void)ttg::toLinearLayout(result.cloneWithEncoding(candidate));
  });
  if (!legal)
    return failure();
  return candidate;
}

struct MmaEvolution {
  ttg::MACAMmaEncodingAttr layout;
  unsigned changedAxes;
};

SmallVector<unsigned, 2> oneStepFactors(unsigned baseline) {
  SmallVector<unsigned, 2> factors{baseline};
  if (baseline <= std::numeric_limits<unsigned>::max() / 2)
    factors.push_back(baseline * 2);
  return factors;
}

SmallVector<MmaEvolution> evolveMma(
    FuncOp function, ttg::MACAMmaEncodingAttr baseline) {
  SmallVector<MmaEvolution> neighbors;
  ArrayRef<unsigned> baselineWarps = baseline.getWarpsPerCTA();
  ArrayRef<unsigned> baselineElements = baseline.getElementsMNK();
  if (baselineWarps.size() != 2 || baselineElements.size() != 3 ||
      baseline.getColMajor() > 1)
    return neighbors;

  const uint64_t numWarps = static_cast<uint64_t>(baselineWarps[0]) *
                            static_cast<uint64_t>(baselineWarps[1]);
  if (numWarps == 0 || numWarps > std::numeric_limits<unsigned>::max())
    return neighbors;

  SetVector<ttg::MACAMmaEncodingAttr> unique;
  for (unsigned wM = 1; wM <= numWarps; ++wM) {
    if (numWarps % wM != 0)
      continue;
    SmallVector<unsigned, 2> warps{wM,
                                   static_cast<unsigned>(numWarps / wM)};
    for (unsigned eM : oneStepFactors(baselineElements[0])) {
      for (unsigned eN : oneStepFactors(baselineElements[1])) {
        for (unsigned eK : oneStepFactors(baselineElements[2])) {
          SmallVector<unsigned, 3> elements{eM, eN, eK};
          for (unsigned colMajor : {baseline.getColMajor(),
                                    1u - baseline.getColMajor()}) {
            ScopedDiagnosticHandler suppress(
                function.getContext(), [](Diagnostic &) { return success(); });
            FailureOr<ttg::MACAMmaEncodingAttr> layout = buildMmaNeighbor(
                function, baseline, warps, elements, colMajor);
            if (succeeded(layout) && *layout != baseline)
              unique.insert(*layout);
          }
        }
      }
    }
  }

  for (ttg::MACAMmaEncodingAttr layout : unique) {
    ArrayRef<unsigned> warps = layout.getWarpsPerCTA();
    ArrayRef<unsigned> elements = layout.getElementsMNK();
    unsigned changedAxes =
        (warps != baselineWarps) +
        (elements[0] != baselineElements[0]) +
        (elements[1] != baselineElements[1]) +
        (elements[2] != baselineElements[2]) +
        (layout.getColMajor() != baseline.getColMajor());
    neighbors.push_back({layout, changedAxes});
  }

  // Prefer complete, diverse evolutions before single-axis neighbors. This is
  // a stable portfolio order, not a performance prediction; every retained
  // assignment is still compiled and measured on the target.
  llvm::stable_sort(neighbors, [&](const MmaEvolution &lhs,
                                   const MmaEvolution &rhs) {
    if (lhs.changedAxes != rhs.changedAxes)
      return lhs.changedAxes > rhs.changedAxes;
    auto lhsWarps = lhs.layout.getWarpsPerCTA();
    auto rhsWarps = rhs.layout.getWarpsPerCTA();
    if ((lhsWarps[0] >= lhsWarps[1]) != (rhsWarps[0] >= rhsWarps[1]))
      return lhsWarps[0] >= lhsWarps[1];
    auto lhsElements = lhs.layout.getElementsMNK();
    auto rhsElements = rhs.layout.getElementsMNK();
    if (lhsElements[1] != rhsElements[1])
      return lhsElements[1] > rhsElements[1];
    if (lhs.layout.getColMajor() != rhs.layout.getColMajor())
      return lhs.layout.getColMajor() > rhs.layout.getColMajor();
    if (lhsElements[2] != rhsElements[2])
      return lhsElements[2] > rhsElements[2];
    if (lhsElements[0] != rhsElements[0])
      return lhsElements[0] > rhsElements[0];
    if (lhsWarps[0] != rhsWarps[0])
      return lhsWarps[0] > rhsWarps[0];
    return lhsWarps[1] < rhsWarps[1];
  });
  return neighbors;
}

class GluonMmaLayoutCandidatePass final
    : public impl::GluonMmaLayoutCandidatePassBase<
          GluonMmaLayoutCandidatePass> {
public:
  GluonMmaLayoutCandidatePass() = default;
  explicit GluonMmaLayoutCandidatePass(int capability) {
    computeCapability = capability;
  }

  void runOnOperation() override {
    FuncOp function = getOperation();
    if (computeCapability < 80) {
      LDBG("function=" << function.getSymName()
                        << " action=skip reason=unsupported-target");
      return;
    }

    SetVector<ttg::MACAMmaEncodingAttr> layouts;
    candidate::walkValueTypes(function, [&](Type type) {
      auto tensor = dyn_cast<RankedTensorType>(type);
      if (tensor)
        if (auto mma = getMmaParent(tensor.getEncoding()))
          layouts.insert(mma);
    });

    struct OwnerEvolution {
      ttg::MACAMmaEncodingAttr source;
      SmallVector<MmaEvolution> alternatives;
    };
    SmallVector<OwnerEvolution> evolutions;
    for (ttg::MACAMmaEncodingAttr baseline : layouts) {
      evolutions.push_back({baseline, evolveMma(function, baseline)});
    }

    SmallVector<ArrayAttr> lanes{ArrayAttr::get(function.getContext(), {})};
    auto addLane = [&](ArrayRef<std::pair<unsigned, unsigned>> selections) {
      SmallVector<Attribute> replacements;
      for (auto [owner, alternative] : selections) {
        const OwnerEvolution &evolution = evolutions[owner];
        if (alternative >= evolution.alternatives.size())
          continue;
        auto target = evolution.alternatives[alternative].layout;
        replacements.push_back(candidate::getEncodingReplacement(
            function.getContext(), evolution.source, target));
      }
      if (replacements.empty())
        return;
      ArrayAttr lane = ArrayAttr::get(function.getContext(), replacements);
      if (!llvm::is_contained(lanes, lane)) {
        LDBG("function=" << function.getSymName() << " lane=" << lanes.size()
                          << " replacements=" << replacements.size());
        lanes.push_back(lane);
      }
    };

    // Compose full-function assignments without an owner Cartesian product.
    // Correlated lanes evolve every owner in the same portfolio direction;
    // owner-local lanes retain coverage when different MMA components need
    // different concrete profiles. Candidate zero remains the current C0.
    for (unsigned alternative = 0; lanes.size() <= 5; ++alternative) {
      SmallVector<std::pair<unsigned, unsigned>> correlated;
      for (auto [owner, evolution] : llvm::enumerate(evolutions))
        if (alternative < evolution.alternatives.size())
          correlated.emplace_back(owner, alternative);
      if (correlated.empty())
        break;
      addLane(correlated);
      for (auto [owner, evolution] : llvm::enumerate(evolutions)) {
        if (lanes.size() > 5)
          break;
        if (alternative < evolution.alternatives.size())
          addLane({std::pair<unsigned, unsigned>{owner, alternative}});
      }
    }
    candidate::setCandidates(function, candidate::CandidateStage::Mma, lanes);
  }
};

} // namespace

std::unique_ptr<Pass> createGluonMmaLayoutCandidatePass() {
  return std::make_unique<GluonMmaLayoutCandidatePass>();
}

std::unique_ptr<Pass>
createGluonMmaLayoutCandidatePass(int computeCapability) {
  return std::make_unique<GluonMmaLayoutCandidatePass>(computeCapability);
}

} // namespace mlir::triton::gluon

#undef LDBG
#undef DEBUG_TYPE
