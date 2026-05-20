

#include "dicp/TritonToLinalg/MarkTensorKindPass.h"
#include "dicp/Dialect/TritonDicp/IR/TritonDicpDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mark-tensor-kind"

using namespace mlir;
using namespace triton;
const unsigned INT_BIT_WIDTH = 32;
const unsigned SET_INIT_SIZE = 16;

template <typename T, typename = void> struct has_getPtr : std::false_type {};
template <typename T>
struct has_getPtr<T, std::void_t<decltype(std::declval<T>().getPtr())>>
    : std::true_type {};

template <typename T, typename = void> struct has_getSrc : std::false_type {};
template <typename T>
struct has_getSrc<T, std::void_t<decltype(std::declval<T>().getSrc())>>
    : std::true_type {};

template <typename T, typename = void> struct has_getBase : std::false_type {};
template <typename T>
struct has_getBase<T, std::void_t<decltype(std::declval<T>().getBase())>>
    : std::true_type {};

template <typename OpTy> static Value extractPointer(OpTy op) {
  if constexpr (has_getPtr<OpTy>::value)
    return op.getPtr();
  else if constexpr (has_getSrc<OpTy>::value)
    return op.getSrc();
  else if constexpr (has_getBase<OpTy>::value)
    return op.getBase();
  else {
    Operation *raw = op.getOperation();
    if (!raw || raw->getNumOperands() == 0)
      return Value();
    return raw->getOperand(0);
  }
}

static void setBlockArgumentAttr(BlockArgument blockArg, triton::FuncOp func,
                                 TensorKind tensorKind) {
  unsigned argIdx = blockArg.getArgNumber();
  auto existingAttr =
      func.getArgAttrOfType<IntegerAttr>(argIdx, "tt.tensor_kind");
  TensorKind oldVal = existingAttr
                          ? static_cast<TensorKind>(existingAttr.getInt())
                          : TensorKind::NONE;

  TensorKind finalVal = tensorKind;
  if ((oldVal == TensorKind::INPUT && tensorKind == TensorKind::OUTPUT) ||
      (oldVal == TensorKind::OUTPUT && tensorKind == TensorKind::INPUT)) {
    finalVal = TensorKind::INPUT_OUTPUT;
  } else if (oldVal == TensorKind::INPUT_OUTPUT) {
    finalVal = oldVal;
  }

  LLVM_DEBUG(llvm::dbgs() << "Setting tensor_kind for argument " << argIdx
                          << ": " << finalVal << "\n";);

  func.setArgAttr(
      argIdx, "tt.tensor_kind",
      IntegerAttr::get(IntegerType::get(func.getContext(), INT_BIT_WIDTH),
                       static_cast<int>(finalVal)));
}

template <typename OpTy>
static void addTensorKindToArguments(OpTy op, TensorKind tensorKind) {
  Value ptr = extractPointer(op);
  if (!ptr)
    return;

  LLVM_DEBUG(llvm::dbgs() << "Processing op: " << *op.getOperation() << "\n";);

  Value cur = ptr;
  llvm::SmallPtrSet<Value, SET_INIT_SIZE> visited;
  while (visited.insert(cur).second) {
    if (auto blockArg = dyn_cast<BlockArgument>(cur)) {
      if (auto func = dyn_cast_or_null<triton::FuncOp>(
              blockArg.getOwner()->getParentOp())) {
        if (blockArg.getOwner() == &func.getBody().front() &&
            isa<triton::PointerType>(blockArg.getType())) {
          setBlockArgumentAttr(blockArg, func, tensorKind);
          break;
        }
      }
    }

    Operation *defOp = cur.getDefiningOp();
    if (!defOp || defOp->getNumOperands() == 0)
      break;
    cur = defOp->getOperand(0);
  }
}

template <TensorKind Kind, typename OpTy>
struct MarkTensorKindPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    addTensorKindToArguments(op, Kind);
    return success();
  }
};

void MarkTensorKindPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());

  // INPUT tensors
  patterns.add<
      MarkTensorKindPattern<TensorKind::INPUT, triton::LoadOp>,
      MarkTensorKindPattern<TensorKind::INPUT, triton::dicp::IndexSelectSimdOp>,
      MarkTensorKindPattern<TensorKind::INPUT, triton::dicp::GatherOutToUbOp>,
      MarkTensorKindPattern<TensorKind::INPUT, triton::dicp::IndirectLoadOp>>(
      &getContext());

  // OUTPUT tensors
  patterns.add<
      MarkTensorKindPattern<TensorKind::OUTPUT, triton::StoreOp>,
      MarkTensorKindPattern<TensorKind::OUTPUT, triton::dicp::IndexPutOp>,
      MarkTensorKindPattern<TensorKind::OUTPUT, triton::dicp::ScatterUbToOutOp>,
      MarkTensorKindPattern<TensorKind::OUTPUT, triton::dicp::IndirectStoreOp>>(
      &getContext());

  // INPUT_OUTPUT tensors
  patterns.add<
      MarkTensorKindPattern<TensorKind::INPUT_OUTPUT, triton::AtomicRMWOp>,
      MarkTensorKindPattern<TensorKind::INPUT_OUTPUT, triton::AtomicCASOp>>(
      &getContext());

  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<OperationPass<ModuleOp>> triton::createMarkTensorKindPass() {
  return std::make_unique<MarkTensorKindPass>();
}