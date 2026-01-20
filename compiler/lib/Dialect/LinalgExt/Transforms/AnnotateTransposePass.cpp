#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "dicp/Utils/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <cstdint>

#define DEBUG_TYPE "annotate-transpose-pass"

using namespace mlir;
using namespace mlir::dicp;

namespace mlir {
namespace dicp {
namespace LinalgExt {
#define GEN_PASS_DEF_ANNOTATETRANSPOSEPASS
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace LinalgExt
} // namespace dicp
} // namespace mlir

namespace {

struct AnnotateTransposePass 
    : public mlir::dicp::LinalgExt::impl::AnnotateTransposePassBase<AnnotateTransposePass> {
  
void runOnOperation() override {
  auto funcOp = getOperation();
  
  llvm::outs() << "[INFO] Starting AnnotateTransposePass on function: "
              << funcOp.getName() << "\n";


  llvm::outs() << "[INFO] Function body:\n";
  funcOp.print(llvm::outs());
  llvm::outs() << "\n";
  // 首先收集所有需要标记的bufferization.to_tensor操作
  SmallVector<bufferization::ToTensorOp> toTensorOpsToMark;
  
  // 检查 memref.copy 操作，看是否会传播非标准 stride
  funcOp.walk([&](memref::CopyOp copyOp) {
    auto source = copyOp.getSource();
    auto target = copyOp.getTarget();
    
    llvm::outs() << "[MEMREF_COPY] Copy operation: " << copyOp << "\n";
    
    // 检查源的 stride 信息
    if (auto sourceType = dyn_cast<MemRefType>(source.getType())) {
      auto [sourceStrides, sourceOffsets] = sourceType.getStridesAndOffset();
      llvm::outs() << "  Source strides: [";
      for (size_t i = 0; i < sourceStrides.size(); ++i) {
        llvm::outs() << sourceStrides[i] << (i < sourceStrides.size()-1 ? ", " : "");
      }
      llvm::outs() << "]\n";
      
      bool isSourcePermuted = mlir::dicp::isaPermutedMemRefType(sourceType);
      llvm::outs() << "  Source is permuted: " << isSourcePermuted << "\n";
      
      // 检查目标的 stride 信息
      if (auto targetType = dyn_cast<MemRefType>(target.getType())) {
        auto [targetStrides, targetOffsets] = targetType.getStridesAndOffset();
        llvm::outs() << "  Target strides: [";
        for (size_t i = 0; i < targetStrides.size(); ++i) {
          llvm::outs() << targetStrides[i] << (i < targetStrides.size()-1 ? ", " : "");
        }
        llvm::outs() << "]\n";
      
        bool isTargetPermuted = mlir::dicp::isaPermutedMemRefType(targetType);
        llvm::outs() << "  Target is permuted: " << isTargetPermuted << "\n";
        
        // 关键：如果源是置换的，即使目标不是，也要追踪目标的使用者
        if (isSourcePermuted) {
          // 检查目标的使用者
          for (auto user : target.getUsers()) {
            if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(user)) {
              toTensorOpsToMark.push_back(toTensorOp);
              llvm::outs() << "  [COPY_SRC_PERMUTED] Marked bufferization.to_tensor for annotation (source was permuted)\n";
            }
          }
          
          // 检查目标是否是某个分配的子视图，如果是，则追踪分配的使用者
          if (auto sourceDefOp = target.getDefiningOp()) {
            if (auto subviewOp = dyn_cast<memref::SubViewOp>(sourceDefOp)) {
              Value parentMemRef = subviewOp.getSource();
              
              // 检查父memref的使用者
              for (auto user : parentMemRef.getUsers()) {
                if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(user)) {
                  llvm::outs() << "  [COPY_SRC_PERMUTED_PARENT] Found bufferization.to_tensor user of parent of copy target: " << toTensorOp << "\n";
                  toTensorOpsToMark.push_back(toTensorOp);
                  llvm::outs() << "  [COPY_SRC_PERMUTED_PARENT] Marked bufferization.to_tensor for annotation from parent of copy target (source was permuted)\n";
                }
              }
            }
          }
        } else if (isTargetPermuted) {
          // 如果目标是置换的
          for (auto user : target.getUsers()) {
            if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(user)) {
              toTensorOpsToMark.push_back(toTensorOp);
              llvm::outs() << "  Marked bufferization.to_tensor for annotation from copy target\n";
            }
          }
        }
      }
    }
  });

  // 遍历所有 bufferization.to_tensor 操作
  funcOp.walk([&](bufferization::ToTensorOp toTensorOp) {
    // 检查源 memref 是否来自于具有非标准 stride 的操作
    Value sourceMemRef = toTensorOp.getOperand();
    
    // 检查源 memref 是否为置换类型
    if (auto memRefType = dyn_cast<MemRefType>(sourceMemRef.getType())) {
      bool isPermuted = mlir::dicp::isaPermutedMemRefType(memRefType);
      llvm::outs() << "[TO_TENSOR_CHECK] bufferization.to_tensor: " << toTensorOp << "\n";
      llvm::outs() << "  Source memref: " << sourceMemRef << "\n";
      llvm::outs() << "  MemRefType: ";
      memRefType.dump();
      llvm::outs() << "\n";
      llvm::outs() << "  Is permuted: " << isPermuted << "\n";
      
      if (isPermuted) {
        toTensorOpsToMark.push_back(toTensorOp);
        llvm::outs() << "  [MARK_ADDED] Marked bufferization.to_tensor for annotation\n";
      }
    }
  });

  // 现在对所有标记的to_tensor操作添加annotation
  // for (auto toTensorOp : toTensorOpsToMark) {
  //   llvm::outs() << "  [MARK_ADDED] About to add transpose annotation to bufferization.to_tensor result\n";
  //   OpBuilder builder(toTensorOp);  // 使用toTensorOp的builder，在其后插入
  //   auto markOp = builder.create<annotation::MarkOp>(
  //       toTensorOp->getLoc(), toTensorOp.getResult());
  //   markOp->setAttr("MayImplicitTransposeWithLastAxis", 
  //                  UnitAttr::get(&getContext()));
  //   llvm::outs() << "      toTensorOp: " << toTensorOp << ", markOp: " << markOp << "\n";
  //   llvm::outs() << "  [MARK_ADDED] Added transpose annotation to bufferization.to_tensor result\n";
  //   llvm::outs() << "  Created annotation::MarkOp: " << markOp << "\n";
  // }

  for (auto toTensorOp : toTensorOpsToMark) {
    llvm::outs() << "  [MARK_ADDED] About to add transpose annotation to bufferization.to_tensor result\n";
    
    // 修改点 1: 初始化 Builder (可以使用 context)
    OpBuilder builder(toTensorOp->getContext());
    
    // 修改点 2: 显式设置插入点在 toTensorOp 之后
    builder.setInsertionPointAfter(toTensorOp); 

    auto markOp = builder.create<annotation::MarkOp>(
        toTensorOp->getLoc(), toTensorOp.getResult());
    
    // 建议：使用 builder.getContext() 或者 toTensorOp->getContext() 获取上下文，
    // 以防外层的 getContext() 在某些闭包或静态函数中不可用
    markOp->setAttr("MayImplicitTransposeWithLastAxis", 
                   UnitAttr::get(builder.getContext()));

    llvm::outs() << "      toTensorOp: " << toTensorOp << ", markOp: " << markOp << "\n";
    llvm::outs() << "  [MARK_ADDED] Added transpose annotation to bufferization.to_tensor result\n";
    llvm::outs() << "  Created annotation::MarkOp: " << markOp << "\n";
  }

  llvm::outs() << "[INFO] After Function body:\n";
  funcOp.print(llvm::outs());
  llvm::outs() << "\n";

  llvm::outs() << "[INFO] Finished AnnotateTransposePass on function: "
              << funcOp.getName() << "\n";
}

private:
  bool needsImplicitTranspose(Value value) {
    if (auto memRefType = dyn_cast<MemRefType>(value.getType())) {
      // 检查是否是置换类型的内存布局
      bool isPermuted = mlir::dicp::isaPermutedMemRefType(memRefType);
      llvm::outs() << "zmz [DEBUG] isPermuted: " << isPermuted << "\n";
      llvm::outs() << "Detected permuted memref type: ";
      memRefType.dump();
      llvm::outs() << "\n";
      LLVM_DEBUG({
        if(isPermuted) {
          llvm::dbgs() << "Detected permuted memref type: ";
          memRefType.dump();
          llvm::dbgs() << "\n";
        }
      });
      return isPermuted;
    }
    llvm::outs() << "zmz [DEBUG] Not a MemRefType, skipping.\n";
    return false;
  }
};
} // namespace

namespace mlir::dicp::LinalgExt {
std::unique_ptr<OperationPass<func::FuncOp>> createAnnotateTransposePass() {
  return std::make_unique<AnnotateTransposePass>();
}
} // namespace mlir::dicp::LinalgExt
