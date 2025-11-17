// #include "bishengir/Annotation/Annotation.h"
// #include "bishengir/Dialect/Annotation/IR/Annotation.h"
// #include "compiler/include/bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
// #include "bishengir/Dialect/Annotation/IR/AnnotationOps.cpp.inc"
// #include "bishengir/Dialect/Annotation/IR/AnnotationOpsDialect.h.inc"
// #include "bishengir/Dialect/Annotation/IR/AnnotationOpsDialect.cpp.inc"

#include "dicp/Conversion/LinalgToLinked/LinalgToLinked.h"
#include "dicp/Conversion/LinalgToLinked/VerifyNoLinalgGenericPass.hpp"
#include "dicp/Dialect/NPU/IR/NPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
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
inline constexpr unsigned getMaxEnumValForProgramIDDim() { return 2; }
static auto constexpr LAUNCH_GRID_RANK = getMaxEnumValForProgramIDDim() + 1;
static unsigned int constexpr TRITON_PROGRAM_INFO_ARG_COUNT =
    LAUNCH_GRID_RANK * 2;

class TritonTypeConverter : public mlir::TypeConverter {
public:
  explicit TritonTypeConverter() {
    addConversion([](Type type) { return type; });

    addConversion([](triton::PointerType ptrType) {
      return MemRefType::get({ShapedType::kDynamic}, ptrType.getPointeeType());
    });

    addConversion([](TensorType tensorType) -> Type {
      auto elemType = tensorType.getElementType();
      if (auto ptrType = dyn_cast<triton::PointerType>(elemType)) {
        elemType = ptrType.getPointeeType();
      }
      return MemRefType::get(tensorType.getShape(), elemType);
    });
  }
};

// TritonTypeConverter::TritonTypeConverter() {
//   addConversion([](Type type) { return type; });

//   addConversion([](triton::PointerType ptrType) {
//     return MemRefType::get({ShapedType::kDynamic}, ptrType.getPointeeType());
//   });

//   addConversion([](TensorType tensorType) -> Type {
//     auto elemType = tensorType.getElementType();
//     if (auto ptrType = dyn_cast<triton::PointerType>(elemType)) {
//       elemType = ptrType.getPointeeType();
//     }
//     return MemRefType::get(tensorType.getShape(), elemType);
//   });
// }

static LogicalResult convertMultipleBlockControlFlow(Operation *funcOp,
                                                     OpBuilder &builder) {
  if (!isa<func::FuncOp>(funcOp)) {
    funcOp->emitError(
        "convertMultipleBlockControlFlow can only process func::FuncOp!");
    return failure();
  }

  SmallVector<Operation *> candidate;
  SmallVector<Block *> eraseBlocks;
  for (Block &block : dyn_cast<func::FuncOp>(funcOp).getBody()) {
    auto curTerminator = block.getTerminator();
    if (isa<cf::CondBranchOp>(curTerminator))
      candidate.push_back(curTerminator);
    else if (isa<func::ReturnOp>(curTerminator)) {
      if (candidate.empty()) {
        curTerminator->emitError(
            "funcOp has more than one Block but got a early 'tt.return' Op.");
        return failure();
      }
    } else
      return failure();

    if (!block.isEntryBlock())
      eraseBlocks.push_back(&block);
  }

  if (candidate.empty()) {
    funcOp->emitError("funcOp has more than one Block but no candidate "
                      "Terminator was found!");
    return failure();
  }

  llvm::BitVector visitFlag(candidate.size(), false);

  // Recursive function to convert all cf::CondBranchOp to scf::IfOp
  std::function<void(Operation *, Operation *)> convertToSCF =
      [&](Operation *op, Operation *insertPosOp) -> void {
    auto condBranchOp = dyn_cast_if_present<cf::CondBranchOp>(op);
    auto iter = llvm::find(candidate, condBranchOp);
    if (!(condBranchOp && iter != candidate.end())) {
      op->emitError(
          "convertToSCF must process with condBranchOp in candidates!");
      return;
    }
    visitFlag.set(iter - candidate.begin());

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(insertPosOp);

    // Well, here force to destory original control flow
    builder.create<scf::IfOp>(
        condBranchOp->getLoc(), condBranchOp.getCondition(),
        /*thenBuilder=*/
        [&](OpBuilder &builder, Location loc) {
          SmallVector<Operation *> movedOps = llvm::map_to_vector(
              condBranchOp.getTrueDest()->without_terminator(),
              [](Operation &op) { return &op; });
          for (auto *innerOp : movedOps) {
            innerOp->moveBefore(builder.getInsertionBlock(),
                                builder.getInsertionPoint());
          }

          auto blockTerm = condBranchOp.getTrueDest()->getTerminator();
          if (isa<cf::CondBranchOp>(blockTerm)) {
            if (movedOps.empty()) {
              blockTerm->emitError(
                  "movedOps can not be empty before entering convertToSCF!");
              return;
            }
            convertToSCF(blockTerm, movedOps.back());
          }

          builder.create<scf::YieldOp>(loc);
        },
        /*elseBuilder=*/
        [&](OpBuilder &builder, Location loc) {
          SmallVector<Operation *> movedOps = llvm::map_to_vector(
              condBranchOp.getFalseDest()->without_terminator(),
              [](Operation &op) { return &op; });
          for (auto *innerOp : movedOps) {
            innerOp->moveBefore(builder.getInsertionBlock(),
                                builder.getInsertionPoint());
          }

          auto blockTerm = condBranchOp.getFalseDest()->getTerminator();
          if (isa<cf::CondBranchOp>(blockTerm)) {
            if (movedOps.empty()) {
              blockTerm->emitError(
                  "movedOps can not be empty before entering convertToSCF!");
              return;
            }
            convertToSCF(blockTerm, movedOps.back());
          }

          builder.create<scf::YieldOp>(loc);
        });
  };

  Block::iterator insertOp(candidate.front());
  --insertOp;
  convertToSCF(candidate.front(), &(*insertOp));

  if (!visitFlag.all())
    return failure();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(candidate.front());
  builder.create<func::ReturnOp>(candidate.front()->getLoc());

  for (Operation *eachTerm : candidate)
    eachTerm->erase();
  for (Block *block : llvm::reverse(eraseBlocks))
    block->erase();

  return success();
}

void convertFuncFunc(func::FuncOp func, const bool existDot) {
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

  // 消除多个return，否则会在one-shot buffer报错
  if (!funcFuncBody.hasOneBlock()) {
    if (failed(convertMultipleBlockControlFlow(funcFunc, builder))) {
      llvm_unreachable("Encounter unsupported control flow");
    }
  }

  for (Block &block : funcFuncBody.getBlocks()) {
    auto term = block.getTerminator();
    builder.setInsertionPoint(term);
    builder.create<func::ReturnOp>(func.getLoc(), term->getOperands());
    term->erase();
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

class TritonAnnotationConverter
    : public OpConversionPattern<triton::AnnotationOp> {
public:
  using OpConversionPattern<triton::AnnotationOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::AnnotationOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto markOp = rewriter.create<annotation::MarkOp>(op.getLoc(), op.getSrc());
    // Forward all annotations.
    markOp->setAttrs(op->getAttrs());
    rewriter.eraseOp(op);
    return success();
  }
};

class ExternElementwiseClOpConverter
    : public OpConversionPattern<triton::ExternElementwiseOp> {
public:
  using OpConversionPattern<triton::ExternElementwiseOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::ExternElementwiseOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    // 添加调试日志
    llvm::errs() << "zmz debug ExternElementwiseClOpConverter: Processing op: " << op.getSymbol() << "\n";
    llvm::errs() << "zmz debug ExternElementwiseClOpConverter: Op location: " << op.getLoc() << "\n";
    auto loc = op.getLoc();
    if (!op.getPure()) {
      op->emitWarning() << "impure elementwise op!";
      return failure();
    }
    if (op.getSymbol().contains("__hmf_")) {
      // 1. get or create the declaration of external elementwise function
      Type dstTy = op.getResult().getType();
      bool isDstScalar = !isa<RankedTensorType>(dstTy);
      Type dstElemTy =
          isDstScalar ? dstTy : cast<RankedTensorType>(dstTy).getElementType();
      SmallVector<Type, 4> srcElemTys;
      SmallVector<Value, 4> srcs;
      for (auto src : op.getSrcs()) {
        if (!isa<RankedTensorType>(src.getType())) {
          src = rewriter.create<tensor::FromElementsOp>(
              op.getLoc(), RankedTensorType::get({(int64_t)1}, src.getType()),
              src);
        }
        srcs.push_back(src);
        srcElemTys.push_back(
            cast<RankedTensorType>(src.getType()).getElementType());
      }

      FunctionType elemFuncType =
          FunctionType::get(rewriter.getContext(), srcElemTys, {dstElemTy});
      auto mod = SymbolTable::getNearestSymbolTable(op);
      auto extFunc = dyn_cast_or_null<SymbolOpInterface>(
          SymbolTable::lookupSymbolIn(mod, op.getSymbol()));
      if (!extFunc) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&mod->getRegion(0).front());
        extFunc = rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(),
                                                op.getSymbol(), elemFuncType);
        extFunc.setPrivate();
        extFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(),
                         UnitAttr::get(rewriter.getContext()));
      }
      assert(isa<FunctionOpInterface>(
          SymbolTable::lookupSymbolIn(mod, op.getSymbol())));
      // 2. prepare the output tensor
      Value output;
      if (isDstScalar) {
        dstTy = RankedTensorType::get({(int64_t)1}, dstElemTy);
      }
      bool found = false;
      for (Value v : srcs) {
        if (v.getType() == dstTy) {
          found = true;
          output = v;
          break;
        }
      }
      if (!found) {
        output = rewriter.create<tensor::EmptyOp>(
            op.getLoc(), cast<RankedTensorType>(dstTy).getShape(), dstElemTy);
      }
      // 3. create the linalg.map op
      auto mapOp = rewriter.create<linalg::MapOp>(
          loc,
          /*inputs=*/srcs,
          /*init=*/output,
          /*bodyBuilder=*/
          [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
            auto elemOp = builder.create<func::CallOp>(loc,
                                                       /*name=*/op.getSymbol(),
                                                       /*resultType=*/dstElemTy,
                                                       /*operands=*/regionArgs);
            builder.create<linalg::YieldOp>(loc, elemOp->getResults());
          });
      if (isDstScalar) {
        // need to convert tensor back to scalar
        auto indexType = rewriter.getIndexType();
        Value zeroConstant = rewriter.create<arith::ConstantOp>(
            loc, indexType, rewriter.getIntegerAttr(indexType, 0));
        auto extractOp = rewriter.create<tensor::ExtractOp>(
            loc, mapOp.getResults()[0], zeroConstant);
        rewriter.replaceOp(op, extractOp);
      } else {
        rewriter.replaceOp(op, mapOp);
      }
      return success();
    }
    return failure();
  }
};

class LinalgToLinkedPass : public LinalgToLinkedBase<LinalgToLinkedPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                cf::ControlFlowDialect, tensor::TensorDialect,
                LLVM::LLVMDialect, bufferization::BufferizationDialect,
                memref::MemRefDialect, annotation::AnnotationDialect>();
  }

  void populateLinalgToLinkedConversionPatterns(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns) {
    populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(
        patterns, typeConverter);
    patterns.add<TritonAnnotationConverter>(patterns.getContext());
    llvm::errs() << "zmz debug LinalgToLinkedPass: Adding ExternElementwiseClOpConverter pattern\n";
    patterns.add<ExternElementwiseClOpConverter>(patterns.getContext());
    if (!this->namedOps) {
      linalg::populateElementwiseToLinalgConversionPatterns(patterns);
    }
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TritonTypeConverter tritonTypeConverter{};

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect, LLVM::LLVMDialect,
        bufferization::BufferizationDialect, memref::MemRefDialect,
        annotation::AnnotationDialect>();
    target.addLegalOp<ModuleOp>();
    // 根据条件判断需要转换的OP
    target.addDynamicallyLegalOp<mlir::UnrealizedConversionCastOp>(
        [](mlir::Operation *op) {
          if (op->use_empty()) {
            return false;
          } else {
            return true;
          }
        });

    target.addDynamicallyLegalOp<triton::FuncOp>([&](triton::FuncOp op) {
      return tritonTypeConverter.isSignatureLegal(op.getFunctionType());
    });
    // Check if the kernel contains tl.dot. Without tl.dot,
    // the kernel would be pure AIV kernel.
    bool existDot = false;
    moduleOp.walk([&](linalg::MatmulOp dotOp) {
      existDot = true;
      return WalkResult::interrupt();
    });
    this->populateLinalgToLinkedConversionPatterns(tritonTypeConverter,
                                                   patterns);
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
    moduleOp.walk([&](func::FuncOp func) { convertFuncFunc(func, existDot); });

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
      NamedAttribute syncBlockLockArgAttr(
          StringAttr::get(context, "syncBlockLock"), UnitAttr::get(context));
      MemRefType syncBlockLockArgType =
          MemRefType::get(SmallVector<int64_t>(1, ShapedType::kDynamic),
                          IntegerType::get(context, 8));
      if (failed(func.insertArgument(syncBlockLockArgIdx, // argIndex
                                    syncBlockLockArgType, // argType
                                    nullptr, func->getLoc()))) {
        signalPassFailure();
        return;
      }
      func->setAttr("SyncBlockLockArgIdx",
                    IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                     0)); // 64: 64位整型

      constexpr int64_t workspaceArgIdx = 1;
      MemRefType workspaceArgType =
          MemRefType::get(SmallVector<int64_t>(1, ShapedType::kDynamic),
                          IntegerType::get(context, 8));
      NamedAttribute workspaceArgAttr(StringAttr::get(context, "workspace"),
                                      UnitAttr::get(context));

      if (failed(func.insertArgument(/*argIndex*/ workspaceArgIdx,
                                    /*argType*/ workspaceArgType,
                                    /*dicAttr*/ nullptr, func->getLoc()))) {
        signalPassFailure();
        return;
      }
      func->setAttr("WorkspaceArgIdx",
                    IntegerAttr::get(IntegerType::get(&getContext(), 64), 1));
    }

    target.addIllegalOp<triton::ExternElementwiseOp>();
    llvm::errs() << "zmz debug LinalgToLinkedPass: Applying conversion patterns\n";
    target.addIllegalOp<triton::AnnotationOp>();
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      moduleOp->emitError("failed to apply Convertion Patterns");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> linked::createLinalgToLinkedPass() {
  return std::make_unique<LinalgToLinkedPass>();
}
