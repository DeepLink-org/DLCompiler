#include "dicp/Dialect/TritonExt/Transforms/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "canonical-cmp"

using namespace mlir;
using namespace dicp;
using namespace trtion_ext;
using namespace triton;

namespace mlir::dicp::trtion_ext {
#define GEN_PASS_DEF_CANONICALIZECMPI
#include "dicp/Dialect/TritonExt/Transforms/Passes.h.inc"
} // namespace mlir::dicp::trtion_ext

namespace {

/// Create a constant of value `1` of the given type
/// `tyNewElem`: return an IntegerAttr constant (arith.constant)
static Value buildConstOne(OpBuilder &b, Location loc, IntegerType intTy) {
  MLIRContext *ctx = b.getContext();
  APInt one(intTy.getWidth(), 1);
  auto intAttr = IntegerAttr::get(intTy, one);
  return b.create<arith::ConstantOp>(loc, intTy, intAttr);
}

/// The rewrite pattern for arith::CmpIOp
struct CmpISemanticRewritePattern : public OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp cmp,
                                PatternRewriter &rewriter) const final {
    auto pred = cmp.getPredicate();

    // ttshared只支持slt,ult,sge，因此将sle,sgt,ule转化为slt,ult,sge
    bool isSLE = (pred == arith::CmpIPredicate::sle);
    bool isSGT = (pred == arith::CmpIPredicate::sgt);
    bool isULE = (pred == arith::CmpIPredicate::ule);
    if (!isSLE && !isULE)
      return failure();
    // 过滤：保留那些是 triton.load/triton.store 且 cmp.result() 被用作它们的
    // mask 的 users
    SmallVector<Operation *> cmpUsers{cmp->getUsers()};
    llvm::erase_if(cmpUsers, [&](Operation *userOp) {
      if (auto loadOp = dyn_cast<triton::LoadOp>(userOp)) {
        // 请根据你工程中生成的 accessor 名称调整下面这一行
        // mask 可能为 nullptr / null Value，先检查再比较
        if (loadOp.getMask() && loadOp.getMask() == cmp.getResult())
          return false; // keep
      } else if (auto storeOp = dyn_cast<triton::StoreOp>(userOp)) {
        if (storeOp.getMask() && storeOp.getMask() == cmp.getResult())
          return false; // keep
      }
      return true; // erase everything else
    });
    // 不存在则，不处理
    if (cmpUsers.empty()) {
      LLVM_DEBUG(llvm::dbgs()
                     << "[CmpISemanticRewritePattern] Cmp op: " << cmp << "\n";
                 llvm::dbgs() << "  Predicate: "
                              << static_cast<int>(cmp.getPredicate()) << "\n";
                 llvm::dbgs() << "  Users of cmp result:\n";

                 // 遍历 cmp.getResult() 的所有 users（直接用 getResult() 而非
                 // getUsers()）
                 for (Operation *userOp
                      : cmp.getResult().getUsers()) {
                   llvm::dbgs() << "    -> " << userOp->getName() << " (";
                   userOp->print(llvm::dbgs());
                   llvm::dbgs() << ")\n";
                 });
      return failure();
    }
    Value rhs = cmp.getRhs();
    auto splatOp = rhs.getDefiningOp<triton::SplatOp>();
    if (splatOp == nullptr)
      return failure();

    Value scrVal = splatOp.getSrc();
    IntegerType scrValType = mlir::dyn_cast<IntegerType>(scrVal.getType());
    if (scrValType == nullptr)
      return failure();

    Location loc = cmp.getLoc();
    // Build constant 1 of extType
    Value cstOne = buildConstOne(rewriter, loc, scrValType);

    Value newSplatSrcVal = nullptr;
    // Map predicate
    arith::CmpIPredicate newPred;
    rewriter.setInsertionPoint(splatOp);
    if (isSLE) {
      newSplatSrcVal =
          rewriter.create<arith::AddIOp>(loc, scrValType, scrVal, cstOne);
      newPred = arith::CmpIPredicate::slt;

    } else if (isSGT) {
      newSplatSrcVal =
          rewriter.create<arith::SubIOp>(loc, scrValType, scrVal, cstOne);
      newPred = arith::CmpIPredicate::sge;

    } else /* isULE */ {
      newSplatSrcVal =
          rewriter.create<arith::AddIOp>(loc, scrValType, scrVal, cstOne);
      newPred = arith::CmpIPredicate::ult;
    }

    triton::SplatOp newSplatOp = rewriter.create<triton::SplatOp>(
        loc, splatOp.getType(), newSplatSrcVal);
    rewriter.replaceOp(splatOp, newSplatOp);

    rewriter.setInsertionPoint(cmp);
    Value lhs = cmp.getLhs();
    // Create new cmp on expanded types
    Value newCmp = rewriter.create<arith::CmpIOp>(loc, newPred, lhs,
                                                  newSplatOp.getResult());

    // If the original result type is the same, just replace.
    // NOTE: arith.cmpi returns an integer or vector of i1; our newCmp has same
    // result shape.
    rewriter.replaceOp(cmp, newCmp);
    return success();
  }
};

/// Reverse a CmpIPredicate for operand swapping:
/// a < b  <=>  b > a, etc.
/// eq/ne are symmetric.
static arith::CmpIPredicate reversePredicate(arith::CmpIPredicate p) {
  switch (p) {
  case arith::CmpIPredicate::eq:
    return arith::CmpIPredicate::eq;
  case arith::CmpIPredicate::ne:
    return arith::CmpIPredicate::ne;
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::sgt;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::sle;
  case arith::CmpIPredicate::ult:
    return arith::CmpIPredicate::ugt;
  case arith::CmpIPredicate::ule:
    return arith::CmpIPredicate::uge;
  case arith::CmpIPredicate::ugt:
    return arith::CmpIPredicate::ult;
  case arith::CmpIPredicate::uge:
    return arith::CmpIPredicate::ule;
  default:
    // For safety, return the original if unknown
    return p;
  }
}

/// If cmp.lhs is defined by triton::SplatOp and cmp.rhs is NOT a
/// triton::SplatOp, swap lhs/rhs and reverse the predicate.
struct SwapSplatLhsCmpPattern : public OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp cmp,
                                PatternRewriter &rewriter) const final {
    // Check lhs is splat
    Value lhs = cmp.getLhs();
    Value rhs = cmp.getRhs();

    auto lhsSplat = lhs.getDefiningOp<triton::SplatOp>();
    auto rhsSplat = rhs.getDefiningOp<triton::SplatOp>();
    if (!lhsSplat || rhsSplat)
      return failure();

    Value lhsSplatScrVal = lhsSplat.getSrc();
    if (!mlir::isa<IntegerType>(lhsSplatScrVal.getType()))
      return failure();

    // Swap operands and reverse predicate
    Location loc = cmp.getLoc();
    arith::CmpIPredicate oldPred = cmp.getPredicate();
    arith::CmpIPredicate newPred = reversePredicate(oldPred);

    // Create the new cmp with swapped operands
    rewriter.setInsertionPoint(cmp);
    Value newCmp = rewriter.create<arith::CmpIOp>(loc, newPred, rhs, lhs);

    rewriter.replaceOp(cmp, newCmp);
    return success();
  }
};

/// The pass that applies the pattern
struct CanonicalizeCmpiPass
    : mlir::dicp::trtion_ext::impl::CanonicalizeCmpiBase<CanonicalizeCmpiPass> {

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto ctx = &getContext();
    {
      RewritePatternSet patterns(ctx);
      patterns.add<SwapSplatLhsCmpPattern>(ctx);
      patterns.add<CmpISemanticRewritePattern>(ctx);
      if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
        moduleOp->emitError("failed to apply Canonicalizer Patterns");
        signalPassFailure();
      }
    }
    PassManager pm(&getContext(), moduleOp.getOperationName());
    // Erase dead code and fold constants created during lowering
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::dicp::trtion_ext::createCanonicalizeCmpiPass() {
  return std::make_unique<CanonicalizeCmpiPass>();
}
