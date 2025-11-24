#include "dicp/Dialect/TritonExt/Transforms/CanonicalizerPattern.h"
#include "dicp/Dialect/TritonExt/Transforms/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "canonicalize-triton-ir-ascend"

using namespace mlir;
using namespace dicp;
using namespace trtion_ext;
using namespace triton;

namespace mlir::dicp::trtion_ext {
#define GEN_PASS_DEF_CANONICALIZETRITONIRASCEND
#include "dicp/Dialect/TritonExt/Transforms/Passes.h.inc"
} // namespace mlir::dicp::trtion_ext

namespace {

class BitcastConverter : public OpRewritePattern<triton::BitcastOp> {
public:
  using OpRewritePattern<triton::BitcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::BitcastOp op,
                                PatternRewriter &rewriter) const {
    if (op->hasAttr("Input_Arg_i1_Bitcast_To_i8")) {
      return failure();
    }

    Value result;
    if (auto resPointerType = dyn_cast<triton::PointerType>(op.getType())) {
      // TODO: use typeconverter
      auto srcPointerType = cast<triton::PointerType>(op.getSrc().getType());
      auto resType = MemRefType::get({ShapedType::kDynamic},
                                     resPointerType.getPointeeType());
      // Handling special case
      // %0 = tt.bitcast %arg0 {MixUse} : !tt.ptr<i1> -> !tt.ptr<i8>
      if (isa<BlockArgument>(op.getSrc()) &&
          srcPointerType.getPointeeType() == rewriter.getIntegerType(1) &&
          resPointerType.getPointeeType() == rewriter.getIntegerType(8)) {
        rewriter.modifyOpInPlace(op, [&]() {
          op->setAttr("Input_Arg_i1_Bitcast_To_i8", rewriter.getUnitAttr());
        });
        return success();
      }
      result =
          rewriter.create<arith::BitcastOp>(op.getLoc(), resType, op.getSrc());
    } else {
      result = rewriter.create<arith::BitcastOp>(op.getLoc(), op.getType(),
                                                 op.getSrc());
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

static void converti1ArgInTTFunc(triton::FuncOp func) {
  OpBuilder builder(func);

  auto name = func.getName();
  auto type = func.getFunctionType();

  SmallVector<DictionaryAttr> argAttrs, resAttrs;
  func.getAllArgAttrs(argAttrs);
  func.getAllResultAttrs(resAttrs);

  // bit-casted tt.ptr的特殊处理
  SmallVector<Type> inputTypes{type.getInputs()};
  SmallVector<Type> retTypes{type.getResults()};
  if (func.getSymVisibility() == "public" && !func.isDeclaration()) {
    for (size_t i = 0; i < func.getNumArguments(); ++i) {
      auto arg = func.getArgument(i);
      // Special method for i1 arg
      if (!isa<triton::PointerType>(arg.getType()) ||
          dyn_cast<triton::PointerType>(arg.getType()).getPointeeType() !=
              builder.getIntegerType(1)) {
        continue;
      }

      // 收集当前参数的 users（拷贝到 vector，方便之后 erase）
      SmallVector<Operation *> argVaildUser{arg.getUsers()};
      bool isAllBitcastI1ToI8 =
          llvm::all_of(argVaildUser, [](Operation *userOp) {
            return userOp->hasAttr("Input_Arg_i1_Bitcast_To_i8");
          });

      if (!isAllBitcastI1ToI8) {
        LLVM_DEBUG({
          auto &os = llvm::dbgs();
          os << arg << " has users:\n";
          int cnt = 0;
          for (auto it : argVaildUser) {
            os << "users[" << cnt++ << "] = " << *it;
          }
        });
        func->emitError("The parameters of type i1 input to the function "
                        "cannot be processed.");
        return;
      }

      // 创建 new pointer type (i8 pointer, 复用原 address space)
      auto ttPtrType = triton::PointerType::get(
          builder.getI8Type(),
          mlir::dyn_cast<triton::PointerType>(arg.getType()).getAddressSpace());

      // 先把参数类型改成 i8 pointer（这样替换时类型一致）
      arg.setType(ttPtrType);
      inputTypes[i] = arg.getType();

      // 把 argVaildUser 的结果的 use 全部替换为 arg，然后把这些 op erase 掉
      for (Operation *userOp : argVaildUser) {
        // userOp 可能已经被移除或不在模块中，检查父操作存在性
        if (!userOp || !userOp->getParentOp())
          continue;

        // 对 userOp 的每个 result 做替换（如果类型一致），若不一致则报错并返回
        for (Value res : userOp->getResults()) {
          // 如果类型不一致，直接报错并返回（不做自动 cast）
          if (res.getType() != arg.getType()) {
            userOp->emitError("Result type of this operation is incompatible "
                              "with the new parameter type; cannot replace "
                              "uses safely.");
            return;
          }

          // 类型一致则替换所有 uses
          if (!res.use_empty())
            res.replaceAllUsesWith(arg);
        }

        // 删除原来的中间操作（例如原来的 tt.bitcast）
        userOp->erase();
      }
    }
  }

  auto castType = FunctionType::get(func.getContext(), inputTypes, retTypes);

  auto funcFunc = builder.create<triton::FuncOp>(func.getLoc(), name, castType);
  funcFunc.setAllArgAttrs(argAttrs);
  funcFunc.setAllResultAttrs(resAttrs);

  auto &funcFuncBody = funcFunc.getBody();
  auto &funcBody = func.getBody();

  IRMapping map;
  funcBody.cloneInto(&funcFuncBody, map);

  func.erase();
}

struct CanonicalizeTritonIRAscendPass
    : mlir::dicp::trtion_ext::impl::CanonicalizeTritonIRAscendBase<
          CanonicalizeTritonIRAscendPass> {
  void runOnOperation() override;

  void
  populateTritonToLinalgCanonicalizationPatterns(RewritePatternSet &patterns);
  template <typename OpTy>
  void addTensorKindToArguments(OpTy op, triton::FuncOp func,
                                TensorKind tensorKind);
};

} // namespace

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

  func.setArgAttr(
      argIdx, "tt.tensor_kind",
      IntegerAttr::get(IntegerType::get(func.getContext(), INT_BIT_WIDTH),
                       static_cast<int>(finalVal)));
}

template <typename OpTy>
void CanonicalizeTritonIRAscendPass::addTensorKindToArguments(
    OpTy op, triton::FuncOp func, TensorKind tensorKind) {
  Value ptr = op.getPtr();
  if (!ptr)
    return;

  Value cur = ptr;
  llvm::SmallPtrSet<Value, SET_INIT_SIZE> visited;
  // 回溯 def-use 链，找到起源 BlockArgument
  while (visited.insert(cur).second) {
    // 如果是 BlockArgument，则尝试设置属性
    if (auto blockArg = dyn_cast<BlockArgument>(cur)) {
      if (blockArg.getOwner() == &func.getBody().front()) {
        auto type = blockArg.getType();
        // 检查是否是 triton::PointerType
        if (!isa<triton::PointerType>(type))
          break;
        setBlockArgumentAttr(blockArg, func, tensorKind);
        break;
      }
    }

    Operation *defOp = cur.getDefiningOp();
    if (!defOp)
      break;
    cur = defOp->getOperand(0);
  }
}

void CanonicalizeTritonIRAscendPass::
    populateTritonToLinalgCanonicalizationPatterns(
        RewritePatternSet &patterns) {

  patterns.add<RemfToBasicArithmetic, RemSIToBasicArithmetic>(
      patterns.getContext());
  patterns.add<BitcastCanonicalizer>(patterns.getContext());
  patterns.add<ScalarStoreCanonicalizer>(patterns.getContext());
  patterns.add<ScalarAtomicRMWCanonicalizer>(patterns.getContext());
  patterns.add<ScalarAtomicCASCanonicalizer>(patterns.getContext());
  patterns.add<AtomicMaxMinCanonicalizer>(patterns.getContext());
  patterns.add<ScalarMathCanonicalizer<math::AbsFOp>,
               ScalarMathCanonicalizer<math::CeilOp>,
               ScalarMathCanonicalizer<math::CosOp>,
               ScalarMathCanonicalizer<math::ErfOp>,
               ScalarMathCanonicalizer<math::ExpOp>,
               ScalarMathCanonicalizer<math::Exp2Op>,
               ScalarMathCanonicalizer<math::FloorOp>,
               ScalarMathCanonicalizer<math::LogOp>,
               ScalarMathCanonicalizer<math::Log2Op>,
               ScalarMathCanonicalizer<math::RsqrtOp>,
               ScalarMathCanonicalizer<math::SinOp>,
               ScalarMathCanonicalizer<math::SqrtOp>,
               ScalarMathCanonicalizer<math::TanhOp>,
               ScalarMathCanonicalizer<arith::AddFOp>,
               ScalarMathCanonicalizer<arith::SubFOp>,
               ScalarMathCanonicalizer<arith::MulFOp>,
               ScalarMathCanonicalizer<arith::DivFOp>,
               ScalarMathCanonicalizer<arith::NegFOp>,
               ScalarMathCanonicalizer<arith::RemFOp>,
               ScalarMathCanonicalizer<arith::MaxNumFOp>,
               ScalarMathCanonicalizer<arith::MaximumFOp>,
               ScalarMathCanonicalizer<arith::MinNumFOp>,
               ScalarMathCanonicalizer<arith::MinimumFOp>>(
      patterns.getContext());
  patterns.add<MakeTensorPtrCanonicalizer>(patterns.getContext());
  patterns.add<ReduceSingleCanonicalizer>(patterns.getContext());
}

void CanonicalizeTritonIRAscendPass::runOnOperation() {
  auto moduleOp = getOperation();
  auto ctx = &getContext();

  // Check if the kernel contains tl.dot. Without tl.dot,
  // the kernel would be pure AIV kernel.
  bool existDot = false;
  moduleOp.walk([&](triton::DotOp dotOp) {
    existDot = true;
    return WalkResult::interrupt();
  });
  moduleOp.walk([&](triton::DotScaledOp dotScaledOp) {
    existDot = true;
    return WalkResult::interrupt();
  });

  // Traverse all the triton::FuncOp to add tensor_kind attribute
  moduleOp.walk([&](triton::FuncOp func) {
    func.walk([&](triton::LoadOp loadOp) {
      addTensorKindToArguments(loadOp, func, TensorKind::INPUT);
    });
    func.walk([&](triton::StoreOp storeOp) {
      addTensorKindToArguments(storeOp, func, TensorKind::OUTPUT);
    });
    func.walk([&](triton::AtomicRMWOp atomicOp) {
      addTensorKindToArguments(atomicOp, func, TensorKind::INPUT_OUTPUT);
    });
    func.walk([&](triton::AtomicCASOp atomicOp) {
      addTensorKindToArguments(atomicOp, func, TensorKind::INPUT_OUTPUT);
    });
  });

  {
    RewritePatternSet canonicalizerPatterns(&getContext());
    populateTritonToLinalgCanonicalizationPatterns(canonicalizerPatterns);
    if (failed(applyPatternsGreedily(moduleOp,
                                     std::move(canonicalizerPatterns)))) {
      moduleOp->emitError("failed to apply Canonicalizer Patterns");
      signalPassFailure();
    }
  }

  {
    RewritePatternSet patterns(ctx);
    patterns.add<BitcastConverter>(ctx);
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      moduleOp->emitError("failed to apply Canonicalizer Patterns");
      signalPassFailure();
    }
  }
  moduleOp.walk([&](triton::FuncOp func) { converti1ArgInTTFunc(func); });
}

std::unique_ptr<OperationPass<ModuleOp>>
trtion_ext::createCanonicalizeTritonIRAscendPass() {
  return std::make_unique<CanonicalizeTritonIRAscendPass>();
}
