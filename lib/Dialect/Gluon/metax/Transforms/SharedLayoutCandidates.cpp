#include "triton/Dialect/Gluon/metax/Transforms/LayoutCandidates.h"

#include "triton/Dialect/Gluon/metax/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <limits>
#include <optional>

#define DEBUG_TYPE "gluon-shared-layout-candidates"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[gluon-layout-autotune][shared] " << X << "\n")

namespace ttg = mlir::triton::gpu;
namespace candidate = mlir::triton::gluon::layout_autotune;

namespace mlir::triton::gluon {

#define GEN_PASS_DEF_GLUONSHAREDLAYOUTCANDIDATEPASS
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h.inc"

namespace {

enum class SharedEvolution : unsigned { Baseline, Vec128, Vec64, Phase };

StringRef getEvolutionName(SharedEvolution evolution) {
  switch (evolution) {
  case SharedEvolution::Baseline:
    return "baseline";
  case SharedEvolution::Vec128:
    return "vec128";
  case SharedEvolution::Vec64:
    return "vec64";
  case SharedEvolution::Phase:
    return "phase";
  }
  llvm_unreachable("unknown Shared layout evolution");
}

struct SharedRecord {
  SmallVector<ttg::MemDescType> types;
  SmallVector<unsigned> mmaContiguousSpans;
};

using SharedRecords =
    llvm::MapVector<ttg::SwizzledSharedEncodingAttr, SharedRecord>;

struct SharedLane {
  ArrayAttr assignment;
  SmallVector<LinearLayout> mappings;
};

bool checkedMultiply(unsigned lhs, unsigned rhs, unsigned &product) {
  if (lhs != 0 && rhs > std::numeric_limits<unsigned>::max() / lhs)
    return false;
  product = lhs * rhs;
  return true;
}

void collectSharedRecords(FuncOp function, SharedRecords &records) {
  candidate::walkValueTypes(function, [&](Type type) {
    auto memdesc = dyn_cast<ttg::MemDescType>(type);
    auto shared = memdesc ? dyn_cast<ttg::SwizzledSharedEncodingAttr>(
                                memdesc.getEncoding())
                          : ttg::SwizzledSharedEncodingAttr();
    if (!shared)
      return;
    const unsigned rank = shared.getOrder().size();
    if (rank == 0 || memdesc.getRank() < rank ||
        memdesc.getAllocShape().size() < rank)
      return;
    memdesc = ttg::MemDescType::get(
        memdesc.getShape().take_back(rank), memdesc.getElementType(), shared,
        memdesc.getMemorySpace(), memdesc.getMutableMemory(),
        memdesc.getAllocShape().take_back(rank));
    auto &concreteTypes = records[shared].types;
    if (!llvm::is_contained(concreteTypes, memdesc))
      concreteTypes.push_back(memdesc);
  });

  function.walk([&](ttg::LocalLoadOp load) {
    auto memdesc = dyn_cast<ttg::MemDescType>(load.getSrc().getType());
    auto tensor = dyn_cast<RankedTensorType>(load.getResult().getType());
    if (!memdesc || !tensor)
      return;

    auto shared =
        dyn_cast<ttg::SwizzledSharedEncodingAttr>(memdesc.getEncoding());
    auto dot = dyn_cast<ttg::DotOperandEncodingAttr>(tensor.getEncoding());
    if (!shared || !dot || shared.getOrder().size() != 2)
      return;
    auto mma = dyn_cast<ttg::MACAMmaEncodingAttr>(dot.getParent());
    auto record = records.find(shared);
    if (!mma || record == records.end())
      return;

    const unsigned opIdx = dot.getOpIdx();
    const unsigned contiguousDim = shared.getOrder()[0];
    if (opIdx > 1 || contiguousDim > 1)
      return;

    SmallVector<unsigned> elems = mma.getElemsPerThreadOrTrans(dot);
    SmallVector<int> threadShape = mma.getThreadShape(
        opIdx == 0 ? mma.getIsATrans() : mma.getIsBTrans());
    if (elems.size() != 2 || threadShape.size() != 3 ||
        llvm::any_of(threadShape, [](int extent) { return extent <= 0; }))
      return;

    const unsigned elemIdx = contiguousDim;
    const unsigned threadIdx =
        opIdx == 0 ? (contiguousDim == 0 ? 0 : 2)
                   : (contiguousDim == 0 ? 2 : 1);
    const StringRef mmaAxis =
        opIdx == 0 ? (contiguousDim == 0 ? "M" : "K")
                   : (contiguousDim == 0 ? "K" : "N");
    unsigned span = 0;
    if (!checkedMultiply(static_cast<unsigned>(threadShape[threadIdx]),
                         elems[elemIdx], span) ||
        span == 0)
      return;

    auto &spans = record->second.mmaContiguousSpans;
    if (!llvm::is_contained(spans, span))
      spans.push_back(span);
    LDBG("function=" << function.getSymName() << " layout=" << shared
                      << " opIdx=" << opIdx
                      << " contiguous-dim=" << contiguousDim
                      << " mma-axis=" << mmaAxis
                      << " mma-span=" << span);
  });
}

FailureOr<unsigned> getVectorElements(const SharedRecord &record,
                                      unsigned vectorBits) {
  std::optional<unsigned> commonVec;
  for (ttg::MemDescType type : record.types) {
    Type elementType = type.getElementType();
    if (!elementType.isIntOrFloat())
      return failure();
    const unsigned bitWidth = elementType.getIntOrFloatBitWidth();
    if (vectorBits % bitWidth != 0)
      return failure();
    const unsigned vec = vectorBits / bitWidth;
    if (!llvm::isPowerOf2_64(vec) || (commonVec && *commonVec != vec))
      return failure();
    commonVec = vec;
  }
  if (!commonVec || llvm::any_of(record.mmaContiguousSpans,
                                 [&](unsigned span) {
                                   return span % *commonVec != 0;
                                 }))
    return failure();
  return *commonVec;
}

FailureOr<ttg::SwizzledSharedEncodingAttr>
evolveShared(ttg::SwizzledSharedEncodingAttr baseline,
             const SharedRecord &record, SharedEvolution evolution) {
  unsigned vec = baseline.getVec();
  unsigned perPhase = baseline.getPerPhase();
  unsigned maxPhase = baseline.getMaxPhase();
  switch (evolution) {
  case SharedEvolution::Baseline:
    return baseline;
  case SharedEvolution::Vec128:
  case SharedEvolution::Vec64: {
    FailureOr<unsigned> targetVec = getVectorElements(
        record, evolution == SharedEvolution::Vec128 ? 128 : 64);
    unsigned swizzleSpan = 0;
    if (failed(targetVec) ||
        !checkedMultiply(vec, maxPhase, swizzleSpan) ||
        swizzleSpan % *targetVec != 0)
      return failure();
    vec = *targetVec;
    maxPhase = swizzleSpan / vec;
    break;
  }
  case SharedEvolution::Phase:
    perPhase = perPhase == 1 ? 2 : perPhase / 2;
    break;
  }
  if (!llvm::isPowerOf2_64(vec) || !llvm::isPowerOf2_64(perPhase) ||
      !llvm::isPowerOf2_64(maxPhase))
    return failure();
  return ttg::SwizzledSharedEncodingAttr::get(
      baseline.getContext(), vec, perPhase, maxPhase, baseline.getOrder(),
      baseline.getCTALayout());
}

bool isLocallyLegal(ttg::MemDescType type,
                    ttg::SwizzledSharedEncodingAttr candidateLayout) {
  const unsigned rank = type.getRank();
  ArrayRef<unsigned> order = candidateLayout.getOrder();
  if (rank < 2 || order.size() != rank || order[0] >= rank ||
      order[1] >= rank)
    return false;
  ArrayRef<int64_t> effectiveShape = type.getAllocShape().take_back(rank);
  if (llvm::any_of(effectiveShape, [](int64_t extent) {
        return extent <= 0 || !llvm::isPowerOf2_64(extent);
      }))
    return false;

  SmallVector<int64_t> shapePerCTA =
      ttg::getShapePerCTA(candidateLayout, effectiveShape);
  const int64_t minorExtent = shapePerCTA[order[0]];
  const int64_t rowExtent = shapePerCTA[order[1]];
  unsigned swizzleWidth = 0;
  if (minorExtent <= 0 || rowExtent <= 0 || candidateLayout.getVec() == 0 ||
      !checkedMultiply(candidateLayout.getVec(),
                       candidateLayout.getMaxPhase(), swizzleWidth))
    return false;
  return candidateLayout.getVec() <= static_cast<uint64_t>(minorExtent) &&
         minorExtent % candidateLayout.getVec() == 0 &&
         candidateLayout.getPerPhase() <= static_cast<uint64_t>(rowExtent) &&
         swizzleWidth <= static_cast<uint64_t>(minorExtent);
}

bool isPhysicalChange(ArrayRef<ttg::MemDescType> types,
                      ttg::SwizzledSharedEncodingAttr baseline,
                      ttg::SwizzledSharedEncodingAttr evolved) {
  if (baseline == evolved)
    return false;
  bool changed = false;
  for (ttg::MemDescType type : types) {
    auto candidateType = ttg::MemDescType::get(
        type.getShape(), type.getElementType(), evolved, type.getMemorySpace(),
        type.getMutableMemory(), type.getAllocShape());
    if (!isLocallyLegal(candidateType, evolved))
      return false;
    changed |= ttg::toLinearLayout(type) != ttg::toLinearLayout(candidateType);
  }
  return changed;
}

SharedLane buildLane(FuncOp function, const SharedRecords &records,
                     SharedEvolution evolution) {
  SmallVector<Attribute> replacements;
  SmallVector<LinearLayout> mappings;
  for (const auto &[baseline, record] : records) {
    FailureOr<ttg::SwizzledSharedEncodingAttr> evolved =
        evolveShared(baseline, record, evolution);
    bool useEvolved = succeeded(evolved) &&
                      isPhysicalChange(record.types, baseline, *evolved);
    ttg::SwizzledSharedEncodingAttr selected =
        useEvolved ? *evolved : baseline;
    if (useEvolved) {
      replacements.push_back(candidate::getEncodingReplacement(
          function.getContext(), baseline, selected));
    } else if (evolution != SharedEvolution::Baseline) {
      LDBG("function=" << function.getSymName() << " layout=" << baseline
                        << " candidate=" << getEvolutionName(evolution)
                        << " action=fallback-baseline");
    }
    for (ttg::MemDescType type : record.types) {
      auto selectedType = ttg::MemDescType::get(
          type.getShape(), type.getElementType(), selected,
          type.getMemorySpace(), type.getMutableMemory(), type.getAllocShape());
      mappings.push_back(ttg::toLinearLayout(selectedType));
    }
  }
  return {ArrayAttr::get(function.getContext(), replacements),
          std::move(mappings)};
}

class GluonSharedLayoutCandidatePass final
    : public impl::GluonSharedLayoutCandidatePassBase<
          GluonSharedLayoutCandidatePass> {
public:
  void runOnOperation() override {
    FuncOp function = getOperation();
    SharedRecords records;
    collectSharedRecords(function, records);

    SmallVector<SharedLane> lanes;
    lanes.push_back(
        buildLane(function, records, SharedEvolution::Baseline));
    for (SharedEvolution evolution : {SharedEvolution::Vec128,
                                      SharedEvolution::Vec64,
                                      SharedEvolution::Phase}) {
      SharedLane lane = buildLane(function, records, evolution);
      if (lane.assignment.empty() ||
          llvm::any_of(lanes, [&](const SharedLane &existing) {
            return existing.mappings == lane.mappings;
          }))
        continue;
      lanes.push_back(std::move(lane));
    }
    SmallVector<ArrayAttr> assignments = llvm::map_to_vector(
        lanes, [](const SharedLane &lane) { return lane.assignment; });
    candidate::setCandidates(function, candidate::CandidateStage::Shared,
                             assignments);
    LDBG("function=" << function.getSymName() << " owners=" << records.size()
                      << " lanes=" << lanes.size());
  }
};

} // namespace

std::unique_ptr<Pass> createGluonSharedLayoutCandidatePass() {
  return std::make_unique<GluonSharedLayoutCandidatePass>();
}

} // namespace mlir::triton::gluon

#undef LDBG
#undef DEBUG_TYPE
