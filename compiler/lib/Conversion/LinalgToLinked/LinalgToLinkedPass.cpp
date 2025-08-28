#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "dicp/Conversion/LinalgToLinked/LinalgToLinked.h"
#include "dicp/Conversion/LinalgToLinked/VerifyNoLinalgGenericPass.hpp"
#include "dicp/Dialect/NPU/IR/NPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Location.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "triton/Dialect/Triton/IR/Dialect.h"


#define DEBUG_TYPE "linalg-to-linked"
#include "llvm/Support/Debug.h"
#include <iostream>

using namespace mlir;
using namespace dicp;
using namespace linked;

#define GEN_PASS_CLASSES
#include "dicp/Conversion/LinalgToLinked/Passes.h.inc"

namespace {

const std::string globalKernelAttr = "global_kernel";
const std::string kernelMixModeName = "mix_mode";
inline constexpr unsigned getMaxEnumValForProgramIDDim() {
  return 2;
}
static auto constexpr LAUNCH_GRID_RANK = getMaxEnumValForProgramIDDim() + 1;
static unsigned int constexpr TRITON_PROGRAM_INFO_ARG_COUNT =
    LAUNCH_GRID_RANK * 2;


// 处理嵌套的if/else
void transformNestedIfElse(Operation &op, OpBuilder &builder) {
    auto nestedBranch = dyn_cast<cf::CondBranchOp>(&op);
    SmallVector<Operation*> nestedTrueOps;
    SmallVector<Operation*> nestedFalseOps;
    bool nestedTrueHasReturn = false;
    bool nestedFalseHasReturn = false;

    for (Operation &op : nestedBranch.getTrueDest()->without_terminator()) {
        if (dyn_cast<cf::CondBranchOp>(&op)) {
            transformNestedIfElse(op, builder);
        }
        nestedTrueOps.push_back(&op);
        if (isa<func::ReturnOp>(op)) {
            nestedTrueHasReturn = true;
        }
    }
    for (Operation &op : nestedBranch.getFalseDest()->without_terminator()) {
        if (dyn_cast<cf::CondBranchOp>(&op)) {
            transformNestedIfElse(op, builder);
        }
        nestedFalseOps.push_back(&op);
        if (isa<func::ReturnOp>(op)) {
            nestedFalseHasReturn = true;
        }
    }
    builder.setInsertionPoint(nestedBranch);
    auto nestedIfOp = builder.create<scf::IfOp>(
        nestedBranch.getLoc(),
        nestedBranch.getCondition(),
        [&](OpBuilder &thenBuilder, Location loc) {
            for (Operation *op : nestedTrueOps) {
                op->moveBefore(thenBuilder.getInsertionBlock(), thenBuilder.getInsertionPoint());
            }
            if (!nestedTrueHasReturn) {
                thenBuilder.create<scf::YieldOp>(loc);
            }
        },
        [&](OpBuilder &elseBuilder, Location loc) {
            for (Operation *op : nestedFalseOps) {
                op->moveBefore(elseBuilder.getInsertionBlock(), elseBuilder.getInsertionPoint());
            }
            if (!nestedTrueHasReturn) {
                elseBuilder.create<scf::YieldOp>(loc);
            }
        }
    );
    nestedBranch.erase();
    nestedBranch.getTrueDest()->erase();
    nestedBranch.getFalseDest()->erase();
}
void convertTTFunc(func::FuncOp func, const bool existDot) {
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
      if (!isa<BaseMemRefType>(arg.getType()) ||
          dyn_cast<BaseMemRefType>(arg.getType()).getElementTypeBitWidth() !=
              1) {
        continue;
      }

      SmallVector<Operation *> argVaildUser{arg.getUsers()};
      llvm::erase_if(argVaildUser, [](Operation *op) -> bool {
        return isOpTriviallyDead(op);
      });

      if (!argVaildUser.empty()) {
        LLVM_DEBUG({
          auto &os = llvm::dbgs();
          os << arg << " has users:\n";
          int cnt = 0;
          for (auto it : argVaildUser) {
            os << "users[" << cnt++ << "] = " << *it;
          }
        });
        if (llvm::all_of(argVaildUser, [](Operation *userOp) {
              return isa<UnrealizedConversionCastOp>(userOp);
            })) {
          auto castOp = cast<UnrealizedConversionCastOp>(*argVaildUser.begin());
          if (castOp.getInputs().size() == 1 &&
              castOp.getOutputs().size() == 1) {
            arg.setType(castOp.getOutputs()[0].getType());
            inputTypes[i] = arg.getType();
          }
        } else {
          func->emitError(Twine("Unsupported use of func arg at index ") +
                          Twine(i));
        }
      } else {
        // Process unused bool ptr type specially, which guarantees bool pointer
        // argument's type is realistic and don't mislead backend compiler.
        // realistic memory layout of bool pointer is 8 bit width
        auto memType = dyn_cast<BaseMemRefType>(arg.getType())
                           .cloneWith(std::nullopt, builder.getI8Type());
        arg.setType(memType);
        inputTypes[i] = arg.getType();
      }
    }
  }
  auto castType = FunctionType::get(func.getContext(), inputTypes, retTypes);

  auto funcFunc = builder.create<func::FuncOp>(func.getLoc(), name, castType);
  funcFunc.setAllArgAttrs(argAttrs);
  funcFunc.setAllResultAttrs(resAttrs);
  auto kernelAttr = func->getAttr(globalKernelAttr);
  if (kernelAttr) {
    funcFunc->setAttr(globalKernelAttr, kernelAttr);
  }
  std::string kernelMixMode = "aiv";
  if (existDot) {
    // mix also works for pure cube kernel by using the same MAGIC_ELF keyword
    kernelMixMode = "mix";
  }
  // Set mix_mode in the func attrs so that the backend could know
  // the mix_mode by parse the func attrs.
  // The backend needs to know the mix_mode because the host wrapper
  // needs to set the devbin.magic. Check npu_utils.cpp.
  funcFunc->setAttr(kernelMixModeName, builder.getStringAttr(kernelMixMode));

  auto &funcFuncBody = funcFunc.getBody();
  auto &funcBody = func.getBody();

  IRMapping map;
  funcBody.cloneInto(&funcFuncBody, map);

  for (Block &block : funcFuncBody.getBlocks()) {
    auto term = block.getTerminator();
    if (auto condBranch = dyn_cast<cf::CondBranchOp>(term)) {
        SmallVector<Operation*> trueOps;
        SmallVector<Operation*> falseOps;
        bool trueHasReturn = false;
        bool falseHasReturn = false;
        for (Operation &op : condBranch.getTrueDest()->without_terminator()) {
            if (dyn_cast<cf::CondBranchOp>(&op)) {
                transformNestedIfElse(op, builder);
            }
            trueOps.push_back(&op);
            if (isa<func::ReturnOp>(op)) {
                trueHasReturn = true;
            }
        }
        for (Operation &op : condBranch.getFalseDest()->without_terminator()) {
            if (dyn_cast<cf::CondBranchOp>(&op)) {
                transformNestedIfElse(op, builder);
            }
            falseOps.push_back(&op);
            if (isa<func::ReturnOp>(op)) {
                falseHasReturn = true;
            }
        }
        builder.setInsertionPoint(condBranch);
        auto ifOp = builder.create<scf::IfOp> (
            condBranch.getLoc(),
            condBranch.getCondition(),
            [&](OpBuilder &thenBuilder, Location loc) {
                for (Operation *op : trueOps) {
                    op->moveBefore(thenBuilder.getInsertionBlock(), thenBuilder.getInsertionPoint());
                }
                if (!trueHasReturn) {
                    thenBuilder.create<scf::YieldOp>(loc);
                }
            },
            [&](OpBuilder &elseBuilder, Location loc) {
                for (Operation *op : falseOps) {
                    op->moveBefore(elseBuilder.getInsertionBlock(), elseBuilder.getInsertionPoint());
                }
                if (!falseHasReturn) {
                    elseBuilder.create<scf::YieldOp>(loc);
                }
            }
        );
        if (!trueHasReturn && !falseHasReturn) {
            Block *afterBlock = condBranch->getBlock();
            if (!afterBlock->empty()) {
                builder.setInsertionPointToEnd(afterBlock);
                builder.create<func::ReturnOp>(condBranch.getLoc());
            }
        }
        condBranch.erase();
        condBranch.getTrueDest()->erase();
        condBranch.getFalseDest()->erase();
      } else {
        builder.setInsertionPoint(term);
        builder.create<func::ReturnOp>(func.getLoc(), term->getOperands());
        term->erase();
      }
  }
  func.erase();
}


void addProgramInfo(func::FuncOp func, bool globalKernel) {
  OpBuilder b(func);

  auto origFuncType = func.getFunctionType();
  auto origInputTypes = origFuncType.getInputs();
  SmallVector<Type> newInputTypes(origInputTypes);

  if (globalKernel) {
    func->setAttr(globalKernelAttr, b.getStringAttr(""));
  } else {
    func->setAttr(globalKernelAttr, b.getStringAttr("local"));
  }
}


struct TritonAnnotationConverter
    : OpRewritePattern<triton::AnnotationOp> {
  using OpRewritePattern<triton::AnnotationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::AnnotationOp op,
                                PatternRewriter &rewriter) const final {
    auto markOp = rewriter.create<annotation::MarkOp>(op.getLoc(), op.getSrc());
    // Forward all annotations.
    markOp->setAttrs(op->getAttrs());
    rewriter.eraseOp(op);
    return success();
  }
};



class LinalgToLinkedPass : public LinalgToLinkedBase<LinalgToLinkedPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect,
        scf::SCFDialect, tensor::TensorDialect, annotation::AnnotationDialect,
        bufferization::BufferizationDialect, memref::MemRefDialect>();
  }

  void populateLinalgToLinkedConversionPatterns(RewritePatternSet &patterns) {
    patterns.add<TritonAnnotationConverter>(patterns.getContext());
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    // Check if the kernel contains tl.dot. Without tl.dot,
    // the kernel would be pure AIV kernel.
    bool existDot = false;
    moduleOp.walk([&](linalg::MatmulOp dotOp) {
      existDot = true;
      return WalkResult::interrupt();
    });
    this->populateLinalgToLinkedConversionPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
      moduleOp.emitError("Pattern application failed");
      signalPassFailure();
    }
    size_t tritonFuncCount = 0;
    for (auto func : getOperation().getOps<triton::FuncOp>()) {
      ++tritonFuncCount;
    }

    size_t funcOpCount = 0;
    // 遍历 func::FuncOp 操作
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      ++funcOpCount;
    }

    // 遍历kernel中的function，修改program id、number of programs参数
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      addProgramInfo(func, globalKernel);
    }

    // 函数头尾转换
    moduleOp.walk(
        [&](func::FuncOp func) { convertTTFunc(func, existDot); });

    PassManager pm(context);
    if (failed(pm.run(moduleOp))) {
        signalPassFailure();
    }

    // 强制在函数参数开头添加一个参数，代表工作空间的占位参数
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      if (!func->hasAttr("global_kernel"))
        continue;

      auto context = func.getContext();
      constexpr int64_t syncBlockLockArgIdx = 0;
      NamedAttribute syncBlockLockArgAttr(StringAttr::get(context, "syncBlockLock"),
                                      UnitAttr::get(context));
      MemRefType syncBlockLockArgType =
          MemRefType::get(SmallVector<int64_t>(1, ShapedType::kDynamic),
                          IntegerType::get(context, 8));
      func.insertArgument(syncBlockLockArgIdx, // argIndex
                          syncBlockLockArgType, // argType
                          nullptr, func->getLoc()); // dicAttr
      func->setAttr("SyncBlockLockArgIdx",
                    IntegerAttr::get(IntegerType::get(&getContext(), 64), 0));  // 64: 64位整型

      constexpr int64_t workspaceArgIdx = 1;
      MemRefType workspaceArgType =
          MemRefType::get(SmallVector<int64_t>(1, ShapedType::kDynamic),
                          IntegerType::get(context, 8));
      NamedAttribute workspaceArgAttr(StringAttr::get(context, "workspace"),
                                      UnitAttr::get(context));

      func.insertArgument(/*argIndex*/ workspaceArgIdx,
                          /*argType*/ workspaceArgType,
                          /*dicAttr*/ nullptr, func->getLoc());
      func->setAttr("WorkspaceArgIdx",
                    IntegerAttr::get(IntegerType::get(&getContext(), 64), 1));
    }

  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> linked::createLinalgToLinkedPass() {
  return std::make_unique<LinalgToLinkedPass>();
}
