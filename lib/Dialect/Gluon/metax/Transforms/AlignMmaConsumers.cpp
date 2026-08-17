/*
 * 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
 * Reserved.
 */
#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gluon-align-mma-consumers"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::gluon {

#define GEN_PASS_DEF_GLUONALIGNMMACONSUMERSPASS
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h.inc"

namespace {

static ttg::MACAMmaEncodingAttr getMmaEncoding(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type ? dyn_cast_or_null<ttg::MACAMmaEncodingAttr>(type.getEncoding())
              : ttg::MACAMmaEncodingAttr();
}

static LogicalResult verifyMmaCorridor(
    OpOperand &root, ttg::MACAMmaEncodingAttr encoding) {
  SetVector<Value> slice;
  DenseMap<Value, Attribute> layouts;
  auto stopAtOwnershipBoundary = [](Operation *operation) {
    return isa<tt::DotOp, tt::AtomicRMWOp, tt::AtomicCASOp>(operation) ||
           (!isMemoryEffectFree(operation) &&
            !isa<ttg::ConvertLayoutOp>(operation));
  };
  return mlir::getConvertBackwardSlice(root, slice, encoding, layouts,
                                       stopAtOwnershipBoundary);
}

/// Find the single concrete MMA layout that can legally own a register-only
/// backward slice. The existing rematerialization utility is authoritative;
/// candidate discovery only keeps the search finite and deterministic.
static FailureOr<ttg::MACAMmaEncodingAttr>
findMmaCorridor(OpOperand &root) {
  SetVector<Value> worklist;
  DenseSet<Value> visited;
  SetVector<ttg::MACAMmaEncodingAttr> candidates;
  worklist.insert(root.get());

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    if (auto mma = getMmaEncoding(value)) {
      candidates.insert(mma);
      continue;
    }

    Operation *definition = value.getDefiningOp();
    if (!definition || isa<tt::DotOp>(definition) ||
        !isMemoryEffectFree(definition) || definition->getNumRegions() != 0)
      continue;

    for (Value operand : definition->getOperands())
      if (isa<RankedTensorType>(operand.getType()))
        worklist.insert(operand);
  }
  if (candidates.size() != 1 ||
      failed(verifyMmaCorridor(root, candidates.front())))
    return failure();
  return candidates.front();
}

static bool isDivisibleBy(int64_t extent, ArrayRef<unsigned> factors) {
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

/// Check the exact geometry consumed by the existing MACA dot lowering. The
/// encoding and DotOperand relation themselves remain verified by the dialect.
static bool isLegalForDot(tt::DotOp dot, ttg::MACAMmaEncodingAttr mma) {
  auto a = dyn_cast<RankedTensorType>(dot.getA().getType());
  auto b = dyn_cast<RankedTensorType>(dot.getB().getType());
  auto d = dyn_cast<RankedTensorType>(dot.getD().getType());
  ArrayRef<unsigned> warps = mma.getWarpsPerCTA();
  ArrayRef<unsigned> elements = mma.getElementsMNK();
  SmallVector<int> atom = ttg::getMmaThreadShape(/*needTrans=*/false);
  if (!a || !b || !d || a.getRank() != 2 || b.getRank() != 2 ||
      d.getRank() != 2 || warps.size() != 2 || elements.size() != 3 ||
      atom.size() != 3 ||
      llvm::any_of(atom, [](int value) { return value <= 0; }) ||
      static_cast<uint64_t>(warps[0]) * warps[1] !=
          static_cast<uint64_t>(ttg::lookupNumWarps(dot)))
    return false;

  return isDivisibleBy(
             d.getShape()[0],
             {warps[0], static_cast<unsigned>(atom[0]), elements[0]}) &&
         isDivisibleBy(
             d.getShape()[1],
             {warps[1], static_cast<unsigned>(atom[1]), elements[1]}) &&
         isDivisibleBy(a.getShape()[1],
                       {static_cast<unsigned>(atom[2]), elements[2]}) &&
         isDivisibleBy(b.getShape()[0],
                       {static_cast<unsigned>(atom[2]), elements[2]});
}

static Value convertTo(IRRewriter &rewriter, Location location, Value value,
                       Attribute encoding) {
  auto type = cast<RankedTensorType>(value.getType());
  auto targetType = type.cloneWithEncoding(encoding);
  if (targetType == type)
    return value;
  return rewriter.create<ttg::ConvertLayoutOp>(location, targetType, value);
}

static Value stripOneConversion(Value value) {
  if (auto conversion = value.getDefiningOp<ttg::ConvertLayoutOp>())
    return conversion.getSrc();
  return value;
}

static void alignDot(IRRewriter &rewriter, tt::DotOp dot,
                     ttg::MACAMmaEncodingAttr mma) {
  auto oldResultType = cast<RankedTensorType>(dot.getType());
  auto aType = cast<RankedTensorType>(dot.getA().getType());
  auto bType = cast<RankedTensorType>(dot.getB().getType());
  auto aEncoding = ttg::DotOperandEncodingAttr::get(
      dot.getContext(), /*opIdx=*/0, mma, aType.getElementType());
  auto bEncoding = ttg::DotOperandEncodingAttr::get(
      dot.getContext(), /*opIdx=*/1, mma, bType.getElementType());

  rewriter.setInsertionPoint(dot);
  Value a = convertTo(rewriter, dot.getA().getLoc(),
                      stripOneConversion(dot.getA()), aEncoding);
  Value b = convertTo(rewriter, dot.getB().getLoc(),
                      stripOneConversion(dot.getB()), bEncoding);
  Value accumulator = convertTo(rewriter, dot.getC().getLoc(),
                                stripOneConversion(dot.getC()), mma);

  IRMapping mapping;
  mapping.map(dot.getA(), a);
  mapping.map(dot.getB(), b);
  mapping.map(dot.getC(), accumulator);
  Operation *aligned = rewriter.clone(*dot, mapping);
  aligned->getResult(0).setType(oldResultType.cloneWithEncoding(mma));
  Value result = convertTo(rewriter, dot.getLoc(), aligned->getResult(0),
                           oldResultType.getEncoding());
  rewriter.replaceOp(dot, result);
}

static void alignAtomic(IRRewriter &rewriter, tt::AtomicRMWOp atomic,
                        ttg::MACAMmaEncodingAttr mma) {
  auto oldResultType = cast<RankedTensorType>(atomic.getType());
  rewriter.setInsertionPoint(atomic);

  IRMapping mapping;
  for (Value operand : atomic->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType()))
      continue;
    mapping.map(operand, convertTo(rewriter, operand.getLoc(), operand, mma));
  }
  Operation *aligned = rewriter.clone(*atomic, mapping);
  aligned->getResult(0).setType(oldResultType.cloneWithEncoding(mma));
  Value result = convertTo(rewriter, atomic.getLoc(), aligned->getResult(0),
                           oldResultType.getEncoding());
  rewriter.replaceOp(atomic, result);
}

struct DotPlan {
  tt::DotOp consumer;
  ttg::MACAMmaEncodingAttr encoding;
};

struct AtomicPlan {
  tt::AtomicRMWOp consumer;
  ttg::MACAMmaEncodingAttr encoding;
};

class GluonAlignMmaConsumersPass final
    : public impl::GluonAlignMmaConsumersPassBase<
          GluonAlignMmaConsumersPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<DotPlan> dotPlans;
    SmallVector<AtomicPlan> atomicPlans;

    // A register-resident producer may feed either dot role. The DotOperand
    // relation and MACA geometry, rather than a kernel-specific role name,
    // decide whether both operands and the result can share its parent M0.
    module.walk([&](tt::DotOp dot) {
      ttg::MACAMmaEncodingAttr selected;
      bool conflict = false;
      for (OpOperand &operand : dot->getOpOperands().take_front(2)) {
        auto conversion = operand.get().getDefiningOp<ttg::ConvertLayoutOp>();
        if (!conversion)
          continue;
        FailureOr<ttg::MACAMmaEncodingAttr> source =
            findMmaCorridor(conversion.getSrcMutable());
        if (failed(source))
          continue;
        if (selected && selected != *source) {
          conflict = true;
          break;
        }
        selected = *source;
      }
      if (!selected || conflict || !isLegalForDot(dot, selected)) {
        LDBG("skip dot at " << dot.getLoc()
                            << " reason=no-unique-legal-mma-corridor");
        return;
      }
      auto current = dyn_cast<ttg::DotOperandEncodingAttr>(
          cast<RankedTensorType>(dot.getA().getType()).getEncoding());
      if (current && current.getParent() == selected)
        return;
      dotPlans.push_back({dot, selected});
    });

    // AtomicRMW's ODS contract requires pointer, value, mask, and result to
    // share one encoding. Retarget that complete contract; standard layout
    // rematerialization rebuilds addresses and masks afterwards.
    module.walk([&](tt::AtomicRMWOp atomic) {
      if (!isa<RankedTensorType>(atomic.getVal().getType()))
        return;
      FailureOr<ttg::MACAMmaEncodingAttr> source =
          findMmaCorridor(atomic.getValMutable());
      if (failed(source) ||
          llvm::any_of(atomic->getOpOperands(), [&](OpOperand &operand) {
            return isa<RankedTensorType>(operand.get().getType()) &&
                   failed(verifyMmaCorridor(operand, *source));
          })) {
        LDBG("skip atomic-rmw at " << atomic.getLoc()
                                   << " reason=no-unique-mma-corridor");
        return;
      }
      auto current = cast<RankedTensorType>(atomic.getVal().getType());
      if (current.getEncoding() == *source)
        return;
      atomicPlans.push_back({atomic, *source});
    });

    IRRewriter rewriter(&getContext());
    for (DotPlan &plan : dotPlans) {
      LDBG("align dot at " << plan.consumer.getLoc() << " with "
                           << plan.encoding);
      alignDot(rewriter, plan.consumer, plan.encoding);
    }
    for (AtomicPlan &plan : atomicPlans) {
      LDBG("align atomic-rmw at " << plan.consumer.getLoc() << " with "
                                  << plan.encoding);
      alignAtomic(rewriter, plan.consumer, plan.encoding);
    }
  }
};

} // namespace

std::unique_ptr<Pass> createGluonAlignMmaConsumersPass() {
  return std::make_unique<GluonAlignMmaConsumersPass>();
}

} // namespace mlir::triton::gluon

#undef LDBG
#undef DBGS
#undef DEBUG_TYPE
