#include "dicp/Conversion/LinalgToNPU/LinalgToNPU.h"
#include "dicp/Conversion/LinalgToNPU/AddWorkspaceAndAttrsPass.hpp"
#include "dicp/Dialect/NPU/IR/NPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

// 必要头文件
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Location.h"
// #include "mlir/Dialect/Linalg/IR/LinalgAttributes.h" // 必须包含这个头文件
#include "mlir/Dialect/Linalg/IR/Linalg.h"
// #include "mlir/Dialect/Utils/DialectUtilsEnums.h.inc"


#define DEBUG_TYPE "linalg-to-npu"
#include "llvm/Support/Debug.h"
#include <iostream>

using namespace mlir;
using namespace dicp;

#define GEN_PASS_CLASSES
#include "dicp/Conversion/LinalgToNPU/Passes.h.inc"

namespace {

const std::string globalKernelAttr = "global_kernel";
const std::string kernelMixModeName = "mix_mode";
inline constexpr unsigned getMaxEnumValForProgramIDDim() {
  return 2;
}
static auto constexpr LAUNCH_GRID_RANK = getMaxEnumValForProgramIDDim() + 1;
static unsigned int constexpr TRITON_PROGRAM_INFO_ARG_COUNT =
    LAUNCH_GRID_RANK * 2;

void convertTTFunc(func::FuncOp func, const bool existDot) {
  std::cout << "Converting Triton function: " << func.getName().str() << std::endl;
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
    builder.setInsertionPoint(term);
    builder.create<func::ReturnOp>(func.getLoc(), term->getOperands());
    term->erase();
  }
  std::cout << "Converted finish Triton function: " << func.getName().str() << std::endl;
  func.erase();
}


void addProgramInfo(func::FuncOp func, bool globalKernel) {
  std::cout << "Adding program info to function: " << func.getName().str() << std::endl;
  OpBuilder b(func);

  auto origFuncType = func.getFunctionType();
  auto origInputTypes = origFuncType.getInputs();
  SmallVector<Type> newInputTypes(origInputTypes);
  // newInputTypes.append(TRITON_PROGRAM_INFO_ARG_COUNT, b.getI32Type());

  // auto newFuncType =
  //     b.getFunctionType(newInputTypes, origFuncType.getResults());

  // func.setFunctionType(newFuncType);

  // // 如果需要，给参数新增属性
  // if (func.getAllArgAttrs()) {
  //   SmallVector<DictionaryAttr> newArgAttrs;
  //   func.getAllArgAttrs(newArgAttrs);
  //   newArgAttrs.append(TRITON_PROGRAM_INFO_ARG_COUNT, DictionaryAttr());
  //   func.setAllArgAttrs(newArgAttrs);
  // }

  // 添加对应参数到函数体中
  // for (unsigned i = 0; i < TRITON_PROGRAM_INFO_ARG_COUNT; i++) {
  //   func.getBody().front().addArgument(b.getI32Type(), func.getLoc());
  // }
  
  std::cout << "Adding program info attributes to function: " << func.getName().str() << std::endl;

  if (globalKernel) {
    std::cout << "Adding global kernel attribute to function: " << func.getName().str() << std::endl;
    func->setAttr(globalKernelAttr, b.getStringAttr(""));
  } else {
    std::cout << "Adding local kernel attribute to function: " << func.getName().str() << std::endl;
    func->setAttr(globalKernelAttr, b.getStringAttr("local"));
  }
}

class LinalgToNPUPass : public LinalgToNPUBase<LinalgToNPUPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        npu::NPUDialect, linalg::LinalgDialect, affine::AffineDialect,
        scf::SCFDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    std::cout << "Running Linalg to NPU conversion pass...\n";

    // Check if the kernel contains tl.dot. Without tl.dot,
    // the kernel would be pure AIV kernel.
    bool existDot = false;
    moduleOp.walk([&](triton::DotOp dotOp) {
      existDot = true;
      return WalkResult::interrupt();
    });

    std::cout << "Running Linalg to NPU conversion pass...\n";
    npu::populateLinalgToNPUConversionPatterns(patterns);
    std::cout << "Populating Linalg to NPU conversion patterns...\n";

    if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
      std::cout << "Failed to apply patterns and fold greedily!\n";
      moduleOp.emitError("Pattern application failed");
      signalPassFailure();
    }
    std::cout << "Linalg to NPU conversion pass completed successfully.\n";

    
    size_t tritonFuncCount = 0;
    for (auto func : getOperation().getOps<triton::FuncOp>()) {
      ++tritonFuncCount;
    }
    std::cout << "Number of triton::FuncOp in the module: " << tritonFuncCount << "\n";

    size_t funcOpCount = 0;
    // 遍历 func::FuncOp 操作
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      ++funcOpCount;
    }
    std::cout << "Number of func::FuncOp in the module: " << funcOpCount << "\n";

    // 6.遍历kernel中的function，修改program id、number of programs参数
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      addProgramInfo(func, globalKernel);
    }
    std::cout << "Program info added to functions successfully.\n";

    std::cout << "Running convertTTFunc passes...\n";
    // 8.函数头尾转换
    moduleOp.walk(
        [&](func::FuncOp func) { convertTTFunc(func, existDot); });
    std::cout << "Function header and footer conversion completed successfully.\n";

    std::cout << "Adding workspace argument to functions...\n";
    // 新增功能逻辑：强制在函数参数开头添加一个参数，代表工作空间的占位参数
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      if (!func->hasAttr("global_kernel"))
        continue;

      auto context = func.getContext();
      constexpr int64_t workspaceArgIdx = 0;
      MemRefType workspaceArgType =
          MemRefType::get(SmallVector<int64_t>(1, ShapedType::kDynamic),
                          IntegerType::get(context, 8));
      NamedAttribute workspaceArgAttr(StringAttr::get(context, "workspace"),
                                      UnitAttr::get(context));

      func.insertArgument(/*argIndex*/ workspaceArgIdx,
                          /*argType*/ workspaceArgType,
                          /*dicAttr*/ nullptr, func->getLoc());
      func->setAttr("WorkspaceArgIdx",
                    IntegerAttr::get(IntegerType::get(&getContext(), 64), 0));
    }

  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> npu::createLinalgToNPUPass() {
  return std::make_unique<LinalgToNPUPass>();
}
