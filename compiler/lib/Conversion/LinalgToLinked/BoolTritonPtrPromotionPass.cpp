#include "dicp/Conversion/LinalgToLinked/LinalgToLinked.h"

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
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "bool-triton-ptr-promotion"

using namespace mlir;
using namespace dicp;
using namespace linked;
using namespace triton;

namespace mlir::dicp::linked {
#define GEN_PASS_DEF_BOOLTRITONPTRPROMOTION
#include "dicp/Conversion/LinalgToLinked/Passes.h.inc"
} // namespace mlir::dicp::linked

namespace {

/*
 * Move tt.bitcast to a previous location if tt.bitcast is not directly applied
 * on function arguments
 */
class BitcastCanonicalizer : public OpRewritePattern<triton::BitcastOp> {
public:
  using OpRewritePattern<triton::BitcastOp>::OpRewritePattern;
  /*
   * Move tt.bitcast to a previous location if tt.bitcast is not directly
   * applied on function arguments
   */
  LogicalResult matchAndRewrite(triton::BitcastOp bitcastOp,
                                PatternRewriter &rewriter) const {
    Value castSrc = bitcastOp.getSrc();
    Value castRes = bitcastOp.getResult();
    Type castSrcTy = castSrc.getType();
    Type castSrcPtrTy = isa<ShapedType>(castSrcTy)
                            ? cast<ShapedType>(castSrcTy).getElementType()
                            : castSrcTy;
    if (!isa<triton::PointerType>(castSrcPtrTy))
      return failure();

    auto origBitwidth = getPointeeBitWidth(castSrc.getType());
    auto castBitwidth = getPointeeBitWidth(castRes.getType());

    if (origBitwidth == 1)
      origBitwidth = 8;
    if (castBitwidth == 1)
      castBitwidth = 8;
    if (origBitwidth != castBitwidth) {
      bitcastOp.emitError() << "Casting pointers with unmatched bitwidth!\n";
      return failure();
    }

    Operation *beforeCastOp = castSrc.getDefiningOp();
    if (beforeCastOp == nullptr) {
      return failure();
    }

    auto newRes =
        TypeSwitch<Operation *, FailureOr<Operation *>>(beforeCastOp)
            // before: addptr - bitcast - load/store
            // after: bitcast - addptr - load/store
            .Case<triton::AddPtrOp>([&](triton::AddPtrOp addptrOp) {
              auto newCastOp = rewriter.create<triton::BitcastOp>(
                  bitcastOp.getLoc(), castRes.getType(), addptrOp.getPtr());
              return rewriter.create<triton::AddPtrOp>(
                  bitcastOp.getLoc(), castRes.getType(), newCastOp.getResult(),
                  addptrOp.getOffset());
            })
            .Case<triton::SplatOp>([&](triton::SplatOp splatOp) {
              Type newCastSrcTy =
                  cast<RankedTensorType>(castRes.getType()).getElementType();

              Value splatSrc = splatOp.getSrc();
              Type splatSrcTy = splatSrc.getType();
              if (auto splatSrcTensorTy =
                      dyn_cast<RankedTensorType>(splatSrcTy))
                newCastSrcTy =
                    splatSrcTensorTy.cloneWith(std::nullopt, newCastSrcTy);
              auto newCastOp = rewriter.create<triton::BitcastOp>(
                  bitcastOp.getLoc(), newCastSrcTy, splatSrc);
              return rewriter.create<triton::SplatOp>(
                  bitcastOp.getLoc(), castRes.getType(), newCastOp);
            })
            // before: bitcast - bitcast
            // after(fusion optimization): bitcast
            .Case<triton::BitcastOp>([&](triton::BitcastOp prevCastOp) {
              return rewriter.create<triton::BitcastOp>(
                  bitcastOp.getLoc(), castRes.getType(), prevCastOp.getSrc());
            })
            .Default([&](Operation *op) {
              return rewriter.notifyMatchFailure(bitcastOp,
                                                 "Unknown bitcast pattern");
            });
    if (succeeded(newRes)) {
      rewriter.replaceOp(bitcastOp, newRes.value());
      if (beforeCastOp->use_empty()) {
        rewriter.eraseOp(beforeCastOp);
      }
      LLVM_DEBUG({
        auto &os = llvm::dbgs();
        os << "BitcastCanonicalizer  s has users:\n";
      });
      return success();
    }
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "BitcastCanonicalizer f has users:\n";
    });
    return failure();
  }
};

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

struct BoolTritonPtrPromotionPass
    : mlir::dicp::linked::impl::BoolTritonPtrPromotionBase<
          BoolTritonPtrPromotionPass> {
  void runOnOperation() override;
};
} // namespace

void BoolTritonPtrPromotionPass::runOnOperation() {
  auto moduleOp = getOperation();
  auto ctx = &getContext();
  {
    RewritePatternSet patterns(ctx);
    patterns.add<BitcastCanonicalizer>(ctx);
    patterns.add<BitcastConverter>(ctx);
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      moduleOp->emitError("failed to apply Canonicalizer Patterns");
      signalPassFailure();
    }
  }
  moduleOp.walk([&](triton::FuncOp func) { converti1ArgInTTFunc(func); });
}

std::unique_ptr<OperationPass<ModuleOp>>
linked::createBoolTritonPtrPromotionPass() {
  return std::make_unique<BoolTritonPtrPromotionPass>();
}
