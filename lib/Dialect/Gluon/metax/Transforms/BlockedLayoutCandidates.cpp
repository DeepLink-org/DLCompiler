#include "triton/Dialect/Gluon/metax/Transforms/LayoutCandidates.h"

#include "mlir/IR/Diagnostics.h"
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <limits>
#include <optional>

#define DEBUG_TYPE "gluon-blocked-layout-candidates"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[gluon-layout-autotune][blocked] " << X << "\n")

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;
namespace candidate = mlir::triton::gluon::layout_autotune;

namespace mlir::triton::gluon {

#define GEN_PASS_DEF_GLUONBLOCKEDLAYOUTCANDIDATEPASS
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h.inc"

namespace {

using BlockedTypes = llvm::MapVector<ttg::BlockedEncodingAttr,
                                     SmallVector<RankedTensorType>>;

struct BlockedNeighbor {
  StringLiteral name;
  FailureOr<ttg::BlockedEncodingAttr> (*build)(
      FuncOp, ttg::BlockedEncodingAttr, unsigned, unsigned);
};

std::optional<uint64_t> getProduct(ArrayRef<unsigned> values) {
  uint64_t product = 1;
  for (unsigned value : values) {
    if (value != 0 &&
        product > std::numeric_limits<uint64_t>::max() / value)
      return std::nullopt;
    product *= value;
  }
  return product;
}

void collectBlockedTypes(FuncOp function, BlockedTypes &types) {
  candidate::walkValueTypes(function, [&](Type type) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    auto blocked = tensor ? dyn_cast<ttg::BlockedEncodingAttr>(
                                tensor.getEncoding())
                          : ttg::BlockedEncodingAttr();
    if (!blocked)
      return;
    auto &concreteTypes = types[blocked];
    if (!llvm::is_contained(concreteTypes, tensor))
      concreteTypes.push_back(tensor);
  });
}

FailureOr<ttg::BlockedEncodingAttr>
buildChecked(FuncOp function, ttg::BlockedEncodingAttr baseline,
             ArrayRef<unsigned> sizePerThread,
             ArrayRef<unsigned> threadsPerWarp,
             ArrayRef<unsigned> warpsPerCTA, unsigned numWarps,
             unsigned targetThreadsPerWarp) {
  std::optional<uint64_t> threadCount = getProduct(threadsPerWarp);
  std::optional<uint64_t> warpCount = getProduct(warpsPerCTA);
  if (!threadCount || *threadCount != targetThreadsPerWarp || !warpCount ||
      *warpCount != numWarps)
    return failure();
  auto emitError = [&]() -> InFlightDiagnostic {
    return function.emitError() << "cannot construct Blocked candidate: ";
  };
  auto encoding = ttg::BlockedEncodingAttr::getChecked(
      emitError, function.getContext(), sizePerThread, threadsPerWarp,
      warpsPerCTA, baseline.getOrder(), baseline.getCTALayout());
  if (!encoding)
    return failure();
  return encoding;
}

FailureOr<ttg::BlockedEncodingAttr>
buildB1(FuncOp function, ttg::BlockedEncodingAttr baseline,
        unsigned numWarps, unsigned threadsPerWarp) {
  if (baseline.getOrder().size() != 2 ||
      baseline.getSizePerThread().size() != 2 ||
      baseline.getThreadsPerWarp().size() != 2 ||
      baseline.getWarpsPerCTA().size() != 2)
    return failure();
  const unsigned contiguous = baseline.getOrder()[0];
  const unsigned other = baseline.getOrder()[1];
  SmallVector<unsigned> tpw(baseline.getThreadsPerWarp());
  if (tpw[contiguous] < 2 || tpw[other] >
                                  std::numeric_limits<unsigned>::max() / 2)
    return failure();
  tpw[contiguous] /= 2;
  tpw[other] *= 2;

  SmallVector<unsigned> wpc(baseline.getWarpsPerCTA());
  for (unsigned dim : {contiguous, other}) {
    SmallVector<unsigned, 3> tileFactors{
        baseline.getSizePerThread()[dim],
        baseline.getThreadsPerWarp()[dim],
        baseline.getWarpsPerCTA()[dim]};
    SmallVector<unsigned, 2> denominatorFactors{
        baseline.getSizePerThread()[dim], tpw[dim]};
    std::optional<uint64_t> tile = getProduct(tileFactors);
    std::optional<uint64_t> denominator = getProduct(denominatorFactors);
    if (!tile || !denominator || *denominator == 0 ||
        *tile % *denominator != 0)
      return failure();
    uint64_t quotient = *tile / *denominator;
    if (!llvm::isPowerOf2_64(quotient) ||
        quotient > std::numeric_limits<unsigned>::max())
      return failure();
    wpc[dim] = static_cast<unsigned>(quotient);
  }
  return buildChecked(function, baseline, baseline.getSizePerThread(), tpw,
                      wpc, numWarps, threadsPerWarp);
}

FailureOr<ttg::BlockedEncodingAttr>
buildB3(FuncOp function, ttg::BlockedEncodingAttr baseline,
        unsigned numWarps, unsigned threadsPerWarp) {
  if (baseline.getOrder().size() != 2 ||
      baseline.getSizePerThread().size() != 2 ||
      baseline.getThreadsPerWarp().size() != 2)
    return failure();
  const unsigned contiguous = baseline.getOrder()[0];
  const unsigned other = baseline.getOrder()[1];
  SmallVector<unsigned> spt(baseline.getSizePerThread());
  SmallVector<unsigned> tpw(baseline.getThreadsPerWarp());
  if (spt[contiguous] < 2 || tpw[other] < 2 ||
      tpw[contiguous] > std::numeric_limits<unsigned>::max() / 2)
    return failure();
  spt[contiguous] /= 2;
  tpw[contiguous] *= 2;
  tpw[other] /= 2;
  return buildChecked(function, baseline, spt, tpw,
                      baseline.getWarpsPerCTA(), numWarps, threadsPerWarp);
}

bool supportsAllConcreteTypes(ArrayRef<RankedTensorType> types,
                              ttg::BlockedEncodingAttr encoding) {
  return llvm::all_of(types, [&](RankedTensorType type) {
    if (type.getRank() != static_cast<int64_t>(encoding.getOrder().size()) ||
        llvm::any_of(type.getShape(), [](int64_t extent) {
          return extent <= 0 || !llvm::isPowerOf2_64(extent);
        }))
      return false;
    (void)ttg::toLinearLayout(type.cloneWithEncoding(encoding));
    return true;
  });
}

LogicalResult verifyDirectAsyncCopyUses(
    FuncOp function, ttg::BlockedEncodingAttr baseline,
    ttg::BlockedEncodingAttr evolved, unsigned &useCount) {
  MLIRContext *context = function.getContext();
  StringAttr reg = StringAttr::get(context, "register");
  StringAttr lane = StringAttr::get(context, "lane");
  StringAttr warp = StringAttr::get(context, "warp");
  StringAttr block = StringAttr::get(context, "block");
  StringAttr offset = StringAttr::get(context, "offset");
  LogicalResult status = success();
  useCount = 0;

  auto reject = [&](ttg::AsyncCopyGlobalToLocalOp copy, StringRef reason) {
    LDBG("function=" << function.getSymName() << " op=" << copy.getOperation()
                      << " source=" << baseline << " evolved=" << evolved
                      << " action=reject reason=async-copy-" << reason);
    status = failure();
  };

  function.walk([&](ttg::AsyncCopyGlobalToLocalOp copy) {
    if (failed(status))
      return;
    auto sourceType = dyn_cast<RankedTensorType>(copy.getSrc().getType());
    if (!sourceType || sourceType.getEncoding() != baseline)
      return;
    ++useCount;

    auto destinationType =
        dyn_cast<ttg::MemDescType>(copy.getResult().getType());
    auto shared = destinationType
                      ? dyn_cast<ttg::SwizzledSharedEncodingAttr>(
                            destinationType.getEncoding())
                      : ttg::SwizzledSharedEncodingAttr();
    ArrayRef<unsigned> sourceOrder = evolved.getOrder();
    if (!destinationType || !shared || sourceType.getRank() != 2 ||
        destinationType.getRank() != 2 || sourceOrder.size() != 2 ||
        shared.getOrder().size() != 2 ||
        sourceOrder.front() != shared.getOrder().front()) {
      reject(copy, "rank-or-order");
      return;
    }

    const unsigned contiguousDim = sourceOrder.front();
    const unsigned copyElements = evolved.getSizePerThread()[contiguousDim];
    const unsigned elementBits = tt::getPointeeBitWidth(sourceType);
    const unsigned copyBytes = copyElements * std::max(1u, elementBits / 8);
    constexpr unsigned legalCopyBytes[] = {4, 8, 16};
    if (copyElements == 0 ||
        llvm::none_of(legalCopyBytes,
                      [&](unsigned width) { return copyBytes == width; }) ||
        copyElements > static_cast<unsigned>(copy.getContiguity()) ||
        (shared.getMaxPhase() > 1 && copyElements > shared.getVec())) {
      reject(copy, "transaction-width");
      return;
    }

    auto plannedSourceType = sourceType.cloneWithEncoding(evolved);
    LinearLayout sourceLayout = ttg::toLinearLayout(plannedSourceType);
    auto freeVariables = sourceLayout.getFreeVariableMasks();
    if (freeVariables.lookup(lane) != 0 || freeVariables.lookup(warp) != 0 ||
        freeVariables.lookup(block) != 0) {
      reject(copy, "redundant-issuer");
      return;
    }
    sourceLayout =
        tt::actionRemoveBroadcastedRegs(sourceLayout).apply(sourceLayout);

    auto noSwizzle = ttg::SwizzledSharedEncodingAttr::get(
        context, shared.getVec(), /*perPhase=*/1, /*maxPhase=*/1,
        shared.getOrder(), shared.getCTALayout());
    auto plannedDestinationType = ttg::MemDescType::get(
        destinationType.getShape(), destinationType.getElementType(),
        noSwizzle, destinationType.getMemorySpace(),
        destinationType.getMutableMemory(), destinationType.getAllocShape());
    LinearLayout destinationLayout =
        ttg::toLinearLayout(plannedDestinationType);
    LinearLayout conversion =
        sourceLayout.invertAndCompose(destinationLayout);
    if (!conversion.isTrivialOver({block})) {
      reject(copy, "block-mapping");
      return;
    }
    conversion = conversion.sublayout({reg, lane, warp}, {offset});
    if (!conversion.isInjective() || !conversion.isSurjective() ||
        conversion.getNumConsecutiveInOut() <
            static_cast<int32_t>(copyElements) ||
        sourceLayout.getInDimSize(reg) % copyElements != 0)
      reject(copy, "non-bijective-mapping");
  });
  return status;
}

class GluonBlockedLayoutCandidatePass final
    : public impl::GluonBlockedLayoutCandidatePassBase<
          GluonBlockedLayoutCandidatePass> {
public:
  void runOnOperation() override {
    FuncOp function = getOperation();
    ModuleOp module = function->getParentOfType<ModuleOp>();
    std::optional<int> numWarps = ttg::maybeLookupNumWarps(function);
    int threadsPerWarp =
        module ? ttg::TritonGPUDialect::getThreadsPerWarp(module) : 0;
    if (!numWarps || *numWarps <= 0 || threadsPerWarp <= 0) {
      function.emitError() << "Blocked candidates require fixed launch topology";
      signalPassFailure();
      return;
    }

    BlockedTypes types;
    collectBlockedTypes(function, types);
    SmallVector<ArrayAttr> lanes{ArrayAttr::get(function.getContext(), {})};
    const BlockedNeighbor neighbors[] = {
        {"B1", buildB1},
        {"B3", buildB3},
    };
    for (const auto &[baseline, concreteTypes] : types) {
      for (const BlockedNeighbor &neighbor : neighbors) {
        FailureOr<ttg::BlockedEncodingAttr> evolved = [&] {
          ScopedDiagnosticHandler suppress(
              function.getContext(), [](Diagnostic &) { return success(); });
          return neighbor.build(function, baseline, *numWarps,
                                threadsPerWarp);
        }();
        unsigned directAsyncCopyUses = 0;
        if (failed(evolved) || *evolved == baseline ||
            !supportsAllConcreteTypes(concreteTypes, *evolved) ||
            failed(verifyDirectAsyncCopyUses(
                function, baseline, *evolved, directAsyncCopyUses))) {
          LDBG("function=" << function.getSymName() << " source=" << baseline
                            << " neighbor=" << neighbor.name
                            << " concrete_types=" << concreteTypes.size()
                            << " action=reject reason=attr-or-linear-layout");
          continue;
        }
        auto replacement = candidate::getEncodingReplacement(
            function.getContext(), baseline, *evolved);
        ArrayAttr lane = ArrayAttr::get(function.getContext(), {replacement});
        if (!llvm::is_contained(lanes, lane)) {
          lanes.push_back(lane);
          LDBG("function=" << function.getSymName() << " source=" << baseline
                            << " neighbor=" << neighbor.name
                            << " evolved=" << *evolved
                            << " concrete_types=" << concreteTypes.size()
                            << " async_copy_uses=" << directAsyncCopyUses
                            << " action=accept");
        }
      }
    }
    candidate::setCandidates(function, candidate::CandidateStage::Blocked,
                             lanes);
    LDBG("function=" << function.getSymName() << " owners=" << types.size()
                      << " lanes=" << lanes.size());
  }
};

} // namespace

std::unique_ptr<Pass> createGluonBlockedLayoutCandidatePass() {
  return std::make_unique<GluonBlockedLayoutCandidatePass>();
}

} // namespace mlir::triton::gluon

#undef LDBG
#undef DEBUG_TYPE
