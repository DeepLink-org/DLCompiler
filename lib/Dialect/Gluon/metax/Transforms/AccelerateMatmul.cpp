/*
 * 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
 * Reserved.
 */
#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <memory>

#define DEBUG_TYPE "gluon-accelerate-matmul"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::gluon {

#define GEN_PASS_DEF_GLUONACCELERATEMATMULPASS
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h.inc"

namespace {

// Consumed and erased by GluonToTritonGPUConversionPass.  This is an
// invocation-local lowering option, not a user-visible layout attribute.
constexpr StringLiteral kStoreCoalesceAttr = "ttg.gluon.store-coalesce";

// C0 has one purpose: choose one complete, target-representable M0 for every
// eligible dot.  It deliberately does not enumerate C1-C5 autotune neighbors,
// infer Shared layouts, lower BSM, or materialize Gluon register views.
struct MmaProfile {
  unsigned eM = 1;
  unsigned eN = 1;
  unsigned eK = 1;
  unsigned wM = 1;
  unsigned wN = 1;
  unsigned colMajor = 0;

  bool operator==(const MmaProfile &other) const {
    return eM == other.eM && eN == other.eN && eK == other.eK &&
           wM == other.wM && wN == other.wN &&
           colMajor == other.colMajor;
  }
};

struct Geometry {
  unsigned eM = 1;
  unsigned eN = 1;
  unsigned eK = 1;
  unsigned wM = 1;
  unsigned wN = 1;

  bool operator==(const Geometry &other) const {
    return eM == other.eM && eN == other.eN && eK == other.eK &&
           wM == other.wM && wN == other.wN;
  }
};

struct Candidate {
  MmaProfile profile;
  int64_t repM = 0;
  int64_t repN = 0;
  int64_t repK = 0;
  int64_t totalMmaInstructions = 0;
  int64_t registerSlots = 0;
  int64_t repetitions = 0;
};

struct DotFact {
  tt::DotOp dot;
  RankedTensorType aType;
  RankedTensorType bType;
  RankedTensorType resultType;
  unsigned numWarps = 0;
  unsigned nativeEK = 1;
  tt::InputPrecision inputPrecision;
  SmallVector<Candidate> candidates;
};

struct GroupPlan {
  SmallVector<unsigned> members;
  SmallVector<Candidate> candidates;
  bool resultFeedsDotA = false;
  Candidate selected;
};

static bool isStaticRank2(RankedTensorType type) {
  return type && type.getRank() == 2 &&
         llvm::all_of(type.getShape(),
                      [](int64_t dim) { return dim > 0; });
}

static Geometry getGeometry(const MmaProfile &profile) {
  return {profile.eM, profile.eN, profile.eK, profile.wM, profile.wN};
}

static int computeCapabilityToMMAVersion(int computeCapability) {
  // C500 currently has one MACA M0 implementation: v2.
  return computeCapability >= 80 ? 2 : 0;
}

static ttg::MACAMmaEncodingAttr
makeMmaEncoding(MLIRContext *context, int computeCapability,
                const MmaProfile &profile) {
  return ttg::MACAMmaEncodingAttr::get(
      context, /*versionMajor=*/2,
      /*versionMinor=*/computeCapability % 10,
      SmallVector<unsigned, 2>{profile.wM, profile.wN},
      SmallVector<unsigned, 3>{profile.eM, profile.eN, profile.eK},
      profile.colMajor,
      // C0 chooses M0 ownership only.  LDS transpose and stride stay in the
      // existing Shared/DotOperand lowering contracts.
      /*isATrans=*/false, /*isBTrans=*/false,
      SmallVector<unsigned, 2>{1, 1});
}

static Value findAggregateRoot(tt::DotOp dot) {
  auto extract = dot.getC().getDefiningOp<ExtractSliceOp>();
  if (!extract)
    return {};

  // The only relation needed here is logical accumulator ownership.  Physical
  // ctaIdx/elemIdx are intentionally deferred to GluonToTritonGPU lowering.
  Value result = dot.getResult();
  if (!result.hasOneUse())
    return {};
  auto insert = dyn_cast<InsertSliceOp>(*result.getUsers().begin());
  if (!insert || insert.getUpdate() != result ||
      insert.getOffsets() != extract.getOffsets())
    return {};
  return extract.getSource();
}

static bool hasSameRankedTensorShape(Value lhs, Value rhs) {
  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  return lhsType && rhsType && lhsType.getShape() == rhsType.getShape() &&
         lhsType.getElementType() == rhsType.getElementType();
}

// `colMajor` is selected after geometry.  Keep its evidence deliberately
// narrow: a direct dot result, optionally with one existing conversion, feeds
// the next dot A operand.  Layout propagation owns all longer SSA carriers.
static bool directlyFeedsDotA(tt::DotOp producer) {
  Value value = producer.getResult();
  for (unsigned depth = 0; depth != 2; ++depth) {
    if (!value.hasOneUse())
      return false;
    Operation *user = *value.getUsers().begin();
    if (auto consumer = dyn_cast<tt::DotOp>(user)) {
      return consumer->getParentRegion() == producer->getParentRegion() &&
             consumer.getA() == value &&
             hasSameRankedTensorShape(producer.getResult(), consumer.getA());
    }
    auto convert = dyn_cast<ttg::ConvertLayoutOp>(user);
    if (!convert || convert.getSrc() != value)
      return false;
    value = convert.getResult();
  }
  return false;
}

static SmallVector<unsigned>
enumerateKFactors(int64_t k, unsigned nativeEK, unsigned bitWidth,
                  unsigned kAtom) {
  SmallVector<unsigned> factors;
  if (nativeEK == 0 || bitWidth == 0)
    return factors;
  for (unsigned eK = nativeEK;
       eK <= 128 / bitWidth && k % (static_cast<int64_t>(kAtom) * eK) == 0;
       eK *= 2) {
    factors.push_back(eK);
    if (eK > (128 / bitWidth) / 2)
      break;
  }
  return factors;
}

// M/N grouping belongs to the legal M0 domain.  C0 ranks [1, 1] first, but
// must retain a valid grouped geometry when the native M/N atom cannot tile a
// dot exactly.
static SmallVector<unsigned> enumerateMNFactors(int64_t extent,
                                                 unsigned warps,
                                                 unsigned atom) {
  SmallVector<unsigned> factors;
  const int64_t base = static_cast<int64_t>(warps) * atom;
  if (base <= 0 || extent % base != 0)
    return factors;
  const int64_t quotient = extent / base;
  for (unsigned factor = 1; factor <= quotient; factor *= 2) {
    if (quotient % factor == 0)
      factors.push_back(factor);
    if (factor > quotient / 2)
      break;
  }
  return factors;
}

static bool isTargetRepresentable(const DotFact &fact,
                                  const MmaProfile &profile,
                                  int computeCapability,
                                  Candidate &candidate) {
  if (profile.wM == 0 || profile.wN == 0 || profile.eM == 0 ||
      profile.eN == 0 || profile.eK == 0 ||
      profile.wM * profile.wN != fact.numWarps)
    return false;

  const auto threadShape = ttg::getMmaThreadShape(/*needTrans=*/false);
  if (threadShape.size() != 3)
    return false;
  const auto aShape = fact.aType.getShape();
  const auto bShape = fact.bType.getShape();
  const int64_t m = aShape[0];
  const int64_t k = aShape[1];
  const int64_t n = bShape[1];
  const unsigned aBits = fact.aType.getElementType().getIntOrFloatBitWidth();
  const unsigned bBits = fact.bType.getElementType().getIntOrFloatBitWidth();
  const unsigned maxBits = std::max(aBits, bBits);

  const int64_t mTile = static_cast<int64_t>(profile.wM) * threadShape[0] *
                        profile.eM;
  const int64_t nTile = static_cast<int64_t>(profile.wN) * threadShape[1] *
                        profile.eN;
  const int64_t kTile = static_cast<int64_t>(threadShape[2]) * profile.eK;
  if (profile.eK % fact.nativeEK != 0 ||
      profile.eK * maxBits > 128 || m % mTile != 0 || n % nTile != 0 ||
      k % kTile != 0)
    return false;

  const int64_t repM = m / mTile;
  const int64_t repN = n / nTile;
  const int64_t repK = k / kTile;
  if (repM <= 0 || repN <= 0 || repK <= 0)
    return false;

  auto encoding =
      makeMmaEncoding(fact.resultType.getContext(), computeCapability, profile);
  auto resultType = RankedTensorType::get(
      fact.resultType.getShape(), fact.resultType.getElementType(), encoding);
  auto aEncoding = ttg::DotOperandEncodingAttr::get(
      fact.aType.getContext(), /*opIdx=*/0, encoding,
      fact.aType.getElementType());
  auto bEncoding = ttg::DotOperandEncodingAttr::get(
      fact.bType.getContext(), /*opIdx=*/1, encoding,
      fact.bType.getElementType());
  auto aType = RankedTensorType::get(fact.aType.getShape(),
                                     fact.aType.getElementType(), aEncoding);
  auto bType = RankedTensorType::get(fact.bType.getShape(),
                                     fact.bType.getElementType(), bEncoding);

  // These are the exact layout implementations consumed by the existing MACA
  // lowering.  The explicit repetition checks keep C0's exact geometry in
  // sync with their floor/max implementation.
  if (ttg::getNumRepM(m, profile.wM, profile.eM) != repM ||
      ttg::getNumRepN(n, profile.wN, profile.eN) != repN ||
      ttg::getNumRepK(k, profile.eK) != repK)
    return false;
  auto aReps = encoding.getRepForOperand(aShape, aBits, /*kWidth=*/0,
                                         /*opIdx=*/0);
  auto bReps = encoding.getRepForOperand(bShape, bBits, /*kWidth=*/0,
                                         /*opIdx=*/1);
  if (aReps.size() != 2 || bReps.size() != 2 || aReps[0] != repM ||
      aReps[1] != repK || bReps[0] != repK || bReps[1] != repN)
    return false;
  (void)ttg::toLinearLayout(resultType);
  (void)ttg::toLinearLayout(aType);
  (void)ttg::toLinearLayout(bType);

  candidate.profile = profile;
  candidate.repM = repM;
  candidate.repN = repN;
  candidate.repK = repK;
  candidate.totalMmaInstructions =
      repM * repN * repK * profile.eM * profile.eN *
      (profile.eK / fact.nativeEK);
  candidate.registerSlots =
      repM * repK * profile.eM * profile.eK +
      repN * repK * profile.eN * profile.eK +
      4 * repM * repN * profile.eM * profile.eN;
  candidate.repetitions = repM + repN + repK;
  return true;
}

static SmallVector<Candidate> buildCandidates(const DotFact &fact,
                                              int computeCapability) {
  SmallVector<Candidate> candidates;
  const auto threadShape = ttg::getMmaThreadShape(/*needTrans=*/false);
  if (threadShape.size() != 3)
    return candidates;

  const int64_t m = fact.aType.getShape()[0];
  const int64_t k = fact.aType.getShape()[1];
  const int64_t n = fact.bType.getShape()[1];
  const unsigned bitWidth = std::max(
      fact.aType.getElementType().getIntOrFloatBitWidth(),
      fact.bType.getElementType().getIntOrFloatBitWidth());
  const SmallVector<unsigned> kFactors =
      enumerateKFactors(k, fact.nativeEK, bitWidth, threadShape[2]);

  for (unsigned wM = 1; wM <= fact.numWarps; ++wM) {
    if (fact.numWarps % wM != 0)
      continue;
    const unsigned wN = fact.numWarps / wM;
    for (unsigned eM : enumerateMNFactors(m, wM, threadShape[0])) {
      for (unsigned eN : enumerateMNFactors(n, wN, threadShape[1])) {
        for (unsigned eK : kFactors) {
          // Orientation is selected only after geometry.  Both layouts are
          // checked through the same M0/DotOperand LinearLayout contracts.
          for (unsigned colMajor : {0u, 1u}) {
            Candidate candidate;
            MmaProfile profile{eM, eN, eK, wM, wN, colMajor};
            if (isTargetRepresentable(fact, profile, computeCapability,
                                      candidate))
              candidates.push_back(candidate);
          }
        }
      }
    }
  }
  return candidates;
}

static const Candidate *findCandidate(ArrayRef<Candidate> candidates,
                                      const MmaProfile &profile) {
  for (const Candidate &candidate : candidates)
    if (candidate.profile == profile)
      return &candidate;
  return nullptr;
}

static bool hasNativeMNGrouping(const Candidate &candidate) {
  return candidate.profile.eM == 1 && candidate.profile.eN == 1;
}

static bool betterGeometry(const Candidate &lhs, const Candidate &rhs) {
  // This is the C0 order from maca-layout-candidate-first-principles.md.
  // colMajor is intentionally absent; it is chosen only after the geometry.
  if (hasNativeMNGrouping(lhs) != hasNativeMNGrouping(rhs))
    return hasNativeMNGrouping(lhs);
  // A smaller K repetition wins only when it does not add MMA work.  This is
  // the document's "smaller rK without increasing total MMA" rule.
  if (lhs.repK != rhs.repK) {
    if (lhs.repK < rhs.repK &&
        lhs.totalMmaInstructions <= rhs.totalMmaInstructions)
      return true;
    if (rhs.repK < lhs.repK &&
        rhs.totalMmaInstructions <= lhs.totalMmaInstructions)
      return false;
  }
  if (lhs.totalMmaInstructions != rhs.totalMmaInstructions)
    return lhs.totalMmaInstructions < rhs.totalMmaInstructions;
  if (lhs.repK != rhs.repK)
    return lhs.repK < rhs.repK;
  if (lhs.registerSlots != rhs.registerSlots)
    return lhs.registerSlots < rhs.registerSlots;
  if (lhs.repetitions != rhs.repetitions)
    return lhs.repetitions < rhs.repetitions;
  const unsigned lhsBalance =
      std::abs(static_cast<int>(lhs.profile.wM) - static_cast<int>(lhs.profile.wN));
  const unsigned rhsBalance =
      std::abs(static_cast<int>(rhs.profile.wM) - static_cast<int>(rhs.profile.wN));
  if (lhsBalance != rhsBalance)
    return lhsBalance < rhsBalance;
  if ((lhs.profile.wM >= lhs.profile.wN) !=
      (rhs.profile.wM >= rhs.profile.wN))
    return lhs.profile.wM >= lhs.profile.wN;
  if (lhs.profile.eM != rhs.profile.eM)
    return lhs.profile.eM < rhs.profile.eM;
  if (lhs.profile.eN != rhs.profile.eN)
    return lhs.profile.eN < rhs.profile.eN;
  if (lhs.profile.eK != rhs.profile.eK)
    return lhs.profile.eK < rhs.profile.eK;
  if (lhs.profile.wM != rhs.profile.wM)
    return lhs.profile.wM < rhs.profile.wM;
  return lhs.profile.wN < rhs.profile.wN;
}

static Candidate selectC0(ArrayRef<Candidate> candidates,
                          bool resultFeedsDotA) {
  assert(!candidates.empty());
  const Candidate *bestGeometry = &candidates.front();
  for (const Candidate &candidate : llvm::drop_begin(candidates))
    if (betterGeometry(candidate, *bestGeometry))
      bestGeometry = &candidate;

  const Geometry geometry = getGeometry(bestGeometry->profile);
  const unsigned preferredColMajor = resultFeedsDotA ? 1 : 0;
  const Candidate *bestOrientation = nullptr;
  for (const Candidate &candidate : candidates) {
    if (!(getGeometry(candidate.profile) == geometry))
      continue;
    if (!bestOrientation ||
        (candidate.profile.colMajor == preferredColMajor &&
         bestOrientation->profile.colMajor != preferredColMajor) ||
        (candidate.profile.colMajor == bestOrientation->profile.colMajor &&
         betterGeometry(candidate, *bestOrientation)))
      bestOrientation = &candidate;
  }
  assert(bestOrientation && "selected C0 geometry must have an orientation");
  return *bestOrientation;
}

static bool hasSameC0Domain(const DotFact &lhs, const DotFact &rhs) {
  return lhs.aType.getShape() == rhs.aType.getShape() &&
         lhs.bType.getShape() == rhs.bType.getShape() &&
         lhs.resultType.getShape() == rhs.resultType.getShape() &&
         lhs.aType.getElementType() == rhs.aType.getElementType() &&
         lhs.bType.getElementType() == rhs.bType.getElementType() &&
         lhs.resultType.getElementType() == rhs.resultType.getElementType() &&
         lhs.numWarps == rhs.numWarps && lhs.nativeEK == rhs.nativeEK &&
         lhs.inputPrecision == rhs.inputPrecision;
}

static void rewriteDot(PatternRewriter &rewriter, tt::DotOp dot,
                       const MmaProfile &profile, int computeCapability) {
  auto oldResultType = cast<RankedTensorType>(dot.getResult().getType());
  auto oldAType = cast<RankedTensorType>(dot.getA().getType());
  auto oldBType = cast<RankedTensorType>(dot.getB().getType());
  auto encoding =
      makeMmaEncoding(oldResultType.getContext(), computeCapability, profile);
  auto resultType = RankedTensorType::get(oldResultType.getShape(),
                                           oldResultType.getElementType(),
                                           encoding);
  auto aEncoding = ttg::DotOperandEncodingAttr::get(
      oldAType.getContext(), /*opIdx=*/0, encoding, oldAType.getElementType());
  auto bEncoding = ttg::DotOperandEncodingAttr::get(
      oldBType.getContext(), /*opIdx=*/1, encoding, oldBType.getElementType());
  auto aType = RankedTensorType::get(oldAType.getShape(),
                                     oldAType.getElementType(), aEncoding);
  auto bType = RankedTensorType::get(oldBType.getShape(),
                                     oldBType.getElementType(), bEncoding);

  rewriter.setInsertionPoint(dot);
  Value accumulator = rewriter.create<ttg::ConvertLayoutOp>(
      dot.getLoc(), resultType, dot.getC());
  Value a = rewriter.create<ttg::ConvertLayoutOp>(dot.getA().getLoc(), aType,
                                                   dot.getA());
  Value b = rewriter.create<ttg::ConvertLayoutOp>(dot.getB().getLoc(), bType,
                                                   dot.getB());
  auto accelerated = rewriter.create<tt::DotOp>(
      dot.getLoc(), resultType, a, b, accumulator, dot.getInputPrecision());
  rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(dot, oldResultType,
                                                     accelerated.getResult());
}

class RewritePlannedDot final : public OpRewritePattern<tt::DotOp> {
public:
  RewritePlannedDot(MLIRContext *context,
                    const DenseMap<Operation *, MmaProfile> &plans,
                    int computeCapability)
      : OpRewritePattern<tt::DotOp>(context), plans(plans),
        computeCapability(computeCapability) {}

  LogicalResult matchAndRewrite(tt::DotOp dot,
                                PatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(dot.getResult().getType());
    if (!resultType || isa<ttg::MACAMmaEncodingAttr>(resultType.getEncoding()))
      return failure();
    auto it = plans.find(dot.getOperation());
    if (it == plans.end())
      return failure();
    rewriteDot(rewriter, dot, it->second, computeCapability);
    return success();
  }

private:
  const DenseMap<Operation *, MmaProfile> &plans;
  int computeCapability;
};

class GluonAccelerateMatmulPass
    : public impl::GluonAccelerateMatmulPassBase<
          GluonAccelerateMatmulPass> {
public:
  GluonAccelerateMatmulPass() = default;
  GluonAccelerateMatmulPass(int numStages, bool disablePrefetch,
                            bool storeCoalesce,
                            int computeCapability = 80) {
    this->computeCapability = computeCapability;
    this->numStages = numStages;
    this->disablePrefetch = disablePrefetch;
    this->storeCoalesce = storeCoalesce;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (storeCoalesce)
      module->setAttr(kStoreCoalesceAttr, UnitAttr::get(&getContext()));
    else
      module->removeAttr(kStoreCoalesceAttr);
    module->setAttr("use.opt.maca.mma",
                    IntegerAttr::get(IntegerType::get(&getContext(), 32), 0));
    if (computeCapabilityToMMAVersion(computeCapability) != 2)
      return;

    // Phase 1 is read-only.  C0 must not depend on greedy-rewrite order.
    SmallVector<DotFact> facts;
    module.walk([&](tt::DotOp dot) {
      auto aType = dyn_cast<RankedTensorType>(dot.getA().getType());
      auto bType = dyn_cast<RankedTensorType>(dot.getB().getType());
      auto resultType = dyn_cast<RankedTensorType>(dot.getResult().getType());
      if (!isStaticRank2(aType) || !isStaticRank2(bType) ||
          !isStaticRank2(resultType) ||
          aType.getShape()[1] != bType.getShape()[0] ||
          aType.getShape()[0] != resultType.getShape()[0] ||
          bType.getShape()[1] != resultType.getShape()[1] ||
          !resultType.getEncoding() ||
          isa<ttg::MACAMmaEncodingAttr>(resultType.getEncoding()) ||
          !supportMMA(dot, /*major=*/2, computeCapability % 10))
        return;

      auto nativeElements = getDefaultElemsPerThread(
          aType.getElementType(),
          dot.getInputPrecision() == tt::InputPrecision::TF32,
          computeCapability);
      if (nativeElements.size() != 3)
        return;

      DotFact fact{dot, aType, bType, resultType,
                   static_cast<unsigned>(ttg::lookupNumWarps(dot)),
                   nativeElements[2], dot.getInputPrecision()};
      if (fact.numWarps == 0)
        return;
      fact.candidates = buildCandidates(fact, computeCapability);
      // Exact geometry is a C0 admission condition.  A dot which has no
      // target-representable C0 remains available to the ordinary lowering.
      if (!fact.candidates.empty())
        facts.push_back(std::move(fact));
    });
    if (facts.empty())
      return;

    DenseMap<Operation *, unsigned> dotToIndex;
    for (auto [index, fact] : llvm::enumerate(facts))
      dotToIndex[fact.dot.getOperation()] = index;

    // Sequential K phases and logical aggregate leaves share one complete M0.
    // This is a same-M0 constraint only; physical extract/insert indices are
    // computed later from the finalized concrete types.
    llvm::EquivalenceClasses<Operation *> sameM0;
    for (DotFact &fact : facts)
      sameM0.unionSets(fact.dot.getOperation(), fact.dot.getOperation());

    DenseMap<Value, unsigned> aggregateRoots;
    SmallVector<bool> directDotA(facts.size(), false);
    for (auto [index, fact] : llvm::enumerate(facts)) {
      if (Value root = findAggregateRoot(fact.dot)) {
        auto [it, inserted] = aggregateRoots.try_emplace(root, index);
        if (!inserted)
          sameM0.unionSets(fact.dot.getOperation(),
                            facts[it->second].dot.getOperation());
      }

      if (auto producer = fact.dot.getC().getDefiningOp<tt::DotOp>()) {
        if (auto it = dotToIndex.find(producer.getOperation());
            it != dotToIndex.end())
          sameM0.unionSets(fact.dot.getOperation(),
                            facts[it->second].dot.getOperation());
      }
      directDotA[index] = directlyFeedsDotA(fact.dot);
    }

    DenseMap<Operation *, unsigned> rootToGroup;
    SmallVector<GroupPlan> groups;
    SmallVector<unsigned> dotToGroup(facts.size());
    for (unsigned index = 0; index < facts.size(); ++index) {
      auto leader = sameM0.findLeader(facts[index].dot.getOperation());
      assert(leader != sameM0.member_end());
      Operation *root = *leader;
      auto [it, inserted] = rootToGroup.try_emplace(root, groups.size());
      if (inserted)
        groups.push_back({});
      groups[it->second].members.push_back(index);
      dotToGroup[index] = it->second;
    }

    for (GroupPlan &group : groups) {
      const DotFact &leader = facts[group.members.front()];
      if (!llvm::all_of(llvm::drop_begin(group.members), [&](unsigned member) {
            return hasSameC0Domain(leader, facts[member]);
          })) {
        // This pass has no cross-shape cost model.  Keeping the component
        // unchanged is preferable to selecting a profile by leader order.
        for (unsigned member : group.members)
          facts[member].candidates.clear();
        continue;
      }
      for (const Candidate &candidate : leader.candidates) {
        bool common = true;
        for (unsigned member : llvm::drop_begin(group.members)) {
          const Candidate *same =
              findCandidate(facts[member].candidates, candidate.profile);
          if (!same) {
            common = false;
            break;
          }
        }
        if (common)
          group.candidates.push_back(candidate);
      }
      if (group.candidates.empty()) {
        // Do not make a partial component: ordinary lowering remains correct.
        // All of its members are removed from the C0 rewrite plan below.
        for (unsigned member : group.members)
          facts[member].candidates.clear();
      }
    }

    // C->Dot-A is the sole positive orientation preference.  Dot-B and no
    // identifiable consumer both retain the stable colMajor=0 default.
    for (unsigned index = 0; index < facts.size(); ++index)
      if (directDotA[index])
        groups[dotToGroup[index]].resultFeedsDotA = true;

    DenseMap<Operation *, MmaProfile> plans;
    for (GroupPlan &group : groups) {
      if (group.candidates.empty())
        continue;
      group.selected = selectC0(group.candidates, group.resultFeedsDotA);
      for (unsigned member : group.members) {
        if (facts[member].candidates.empty())
          continue;
        const MmaProfile &profile = group.selected.profile;
        LDBG("C0 " << facts[member].dot.getLoc() << " -> e=[" << profile.eM
                    << "," << profile.eN << "," << profile.eK << "], w=["
                    << profile.wM << "," << profile.wN
                    << "], col=" << profile.colMajor);
        plans[facts[member].dot.getOperation()] = profile;
      }
    }
    if (plans.empty())
      return;

    // Phase 2 reuses the regular MetaX rewrite-pattern shape.  The pattern is
    // intentionally dumb: every decision was made above before the first dot
    // changed type.
    RewritePatternSet patterns(&getContext());
    patterns.add<RewritePlannedDot>(&getContext(), plans, computeCapability);
    if (applyPatternsGreedily(module, std::move(patterns)).failed()) {
      signalPassFailure();
      return;
    }
    module->setAttr("use.opt.maca.mma",
                    IntegerAttr::get(IntegerType::get(&getContext(), 32), 1));
  }
};

} // namespace

std::unique_ptr<Pass>
createGluonAccelerateMatmulPass(int numStages, bool disablePrefetch,
                                bool storeCoalesce, int computeCapability) {
  return std::make_unique<GluonAccelerateMatmulPass>(
      numStages, disablePrefetch, storeCoalesce, computeCapability);
}

} // namespace mlir::triton::gluon

#undef LDBG
#undef DBGS
#undef DEBUG_TYPE
