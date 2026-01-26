#include "dicp/Dialect/LinalgExt/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

// ==============================================================================
// 辅助函数定义
// ==============================================================================

/// 检查一个MemRefType是否是置换类型的辅助函数
/// 判定标准：使用 dicp::isaPermutedMemRefType 或 检查 stride 是否非标准
bool isPermutedOrHasNonUnitLastStride(MemRefType memRefType) {
  if (!memRefType) return false;
  
  // 1. 使用现有的判定函数
  if (mlir::dicp::isaPermutedMemRefType(memRefType)) {
    return true;
  }
  
  // 2. 额外检查：最后维度的stride是否为1
  // 对于 Ascend 来说，如果最后维度 stride != 1，通常意味着不是连续内存，可能需要隐式转置
  auto [strides, offset] = memRefType.getStridesAndOffset();
  if (!strides.empty() && strides.back() != 1) {
    return true;
  }
  
  return false;
}

/// 递归检查值的来源是否具有非标准stride
bool checkValueOriginHasNonStandardStride(Value value) {
  if (auto memRefType = dyn_cast<MemRefType>(value.getType())) {
    if (isPermutedOrHasNonUnitLastStride(memRefType)) {
      return true;
    }
  }
  
  // 检查定义操作
  if (Operation *defOp = value.getDefiningOp()) {
    // 检查Subview操作
    if (auto subViewOp = dyn_cast<memref::SubViewOp>(defOp)) {
      return checkValueOriginHasNonStandardStride(subViewOp.getSource());
    }
    // 检查ReinterpretCast操作
    if (auto castOp = dyn_cast<memref::ReinterpretCastOp>(defOp)) {
      return checkValueOriginHasNonStandardStride(castOp.getSource());
    }
  }
  
  return false;
}

struct AnnotateTransposePass 
    : public mlir::dicp::LinalgExt::impl::AnnotateTransposePassBase<AnnotateTransposePass> {
  
void runOnOperation() override {
  auto funcOp = getOperation();
  
  llvm::outs() << "[INFO] Starting AnnotateTransposePass on function: "
              << funcOp.getName() << "\n";

  // 待处理列表
  SmallVector<bufferization::ToTensorOp> toTensorOpsToMark;
  SmallVector<Operation*> opsToErase; // 用于存储被重写后需要删除的旧Op

  // ==============================================================================
  // 1. 遍历 memref.copy 操作
  //    核心逻辑：检测 Dynamic Subview Copy -> 重写为 Static Full Copy + Annotation
  // ==============================================================================
  funcOp.walk([&](memref::CopyOp copyOp) {
    auto source = copyOp.getSource();
    auto target = copyOp.getTarget();
    
    llvm::outs() << "[MEMREF_COPY_VISIT] " << copyOp << "\n";

    // --- 尝试进行 IR 重写 (Rewrite) ---
    // 目标：将 memref.copy(subview(A), subview(B)) 转换为 memref.copy(A, B)
    // 条件：A 是静态 Permuted，B 是静态 Contiguous，且形状匹配
    
    auto srcSubView = source.getDefiningOp<memref::SubViewOp>();
    auto dstSubView = target.getDefiningOp<memref::SubViewOp>();

    if (srcSubView && dstSubView) {
      Value baseSource = srcSubView.getSource();
      Value baseTarget = dstSubView.getSource();
      
      auto baseSourceType = dyn_cast<MemRefType>(baseSource.getType());
      auto baseTargetType = dyn_cast<MemRefType>(baseTarget.getType());

      if (baseSourceType && baseTargetType && 
          baseSourceType.hasStaticShape() && baseTargetType.hasStaticShape()) {
        
        bool isBaseSourcePermuted = isPermutedOrHasNonUnitLastStride(baseSourceType);
        // 简化判定：如果不是 Permuted 且 stride 正常，视为 Contiguous
        bool isBaseTargetContiguous = !isPermutedOrHasNonUnitLastStride(baseTargetType);

        // 检查 Static Shape 是否一致 (例如都是 2x8xf32)
        if (isBaseSourcePermuted && isBaseTargetContiguous && 
            baseSourceType.getShape() == baseTargetType.getShape()) {
            
            llvm::outs() << "  [REWRITE_MATCH] Found Dynamic Subview Copy candidate for Static Rewrite.\n";
            llvm::outs() << "    Base Source (Permuted): " << baseSourceType << "\n";
            llvm::outs() << "    Base Target (Contiguous): " << baseTargetType << "\n";
            
            // 执行重写
            OpBuilder builder(copyOp->getContext());
            builder.setInsertionPoint(copyOp);
            
            // 1. 创建新的静态 Copy (Base -> Base)
            auto newCopyOp = builder.create<memref::CopyOp>(copyOp.getLoc(), baseSource, baseTarget);
            llvm::outs() << "    -> Replaced with Static Copy: " << newCopyOp << "\n";

            // 2. 关键：在 Base Target (MemRef) 上添加 Annotation
            // 这指导 Ascend 编译器生成隐式转置指令
            builder.setInsertionPointAfter(newCopyOp);
            auto markOp = builder.create<annotation::MarkOp>(copyOp.getLoc(), baseTarget);
            markOp->setAttr("MayImplicitTransposeWithLastAxis", UnitAttr::get(builder.getContext()));
            llvm::outs() << "    -> Added Annotation to Base Target MemRef: " << markOp << "\n";

            // 3. 追踪 Base Target 的 Tensor 使用者
            // 我们需要标记 bufferization.to_tensor(BaseTarget)，这样后续的 MatMul 才能识别到 Layout 变化
            for (auto user : baseTarget.getUsers()) {
                if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(user)) {
                    // 去重检查
                    bool exists = false;
                    for(auto op : toTensorOpsToMark) if(op == toTensorOp) exists = true;
                    
                    if(!exists) {
                        toTensorOpsToMark.push_back(toTensorOp);
                        llvm::outs() << "    -> Scheduled Base Target's ToTensorOp for annotation: " << toTensorOp << "\n";
                    }
                }
            }

            // 4. 标记旧的 Copy Op 待删除
            opsToErase.push_back(copyOp);
            
            // 重写完成，跳过后续分析
            return; 
        }
      }
    }

    // --- 如果没有触发重写，执行常规的传播分析 ---
    // (针对代码中已经是静态 Copy 的情况，或者仅仅进行标记传播)
    
    if (auto sourceType = dyn_cast<MemRefType>(source.getType())) {
      bool isSourcePermuted = isPermutedOrHasNonUnitLastStride(sourceType);
      
      if (auto targetType = dyn_cast<MemRefType>(target.getType())) {
        bool isTargetPermuted = isPermutedOrHasNonUnitLastStride(targetType);
        
        // 如果源是置换的，追踪目标的使用者
        if (isSourcePermuted) {
          // 检查目标的使用者
          for (auto user : target.getUsers()) {
            if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(user)) {
               bool exists = false;
               for(auto op : toTensorOpsToMark) if(op == toTensorOp) exists = true;
               if(!exists) {
                  toTensorOpsToMark.push_back(toTensorOp);
                  llvm::outs() << "  [PROPAGATE] Marked bufferization.to_tensor (Source was permuted)\n";
               }
            }
          }
          
          // 如果目标是 Subview，追踪其父 MemRef
          if (auto sourceDefOp = target.getDefiningOp()) {
            if (auto subviewOp = dyn_cast<memref::SubViewOp>(sourceDefOp)) {
              Value parentMemRef = subviewOp.getSource();
              for (auto user : parentMemRef.getUsers()) {
                if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(user)) {
                    bool exists = false;
                    for(auto op : toTensorOpsToMark) if(op == toTensorOp) exists = true;
                    if(!exists) {
                        toTensorOpsToMark.push_back(toTensorOp);
                        llvm::outs() << "  [PROPAGATE_PARENT] Marked bufferization.to_tensor of Parent MemRef\n";
                    }
                }
              }
            }
          }
        } else if (isTargetPermuted) {
          // 如果目标本身就是置换的
          for (auto user : target.getUsers()) {
            if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(user)) {
                bool exists = false;
                for(auto op : toTensorOpsToMark) if(op == toTensorOp) exists = true;
                if(!exists) {
                    toTensorOpsToMark.push_back(toTensorOp);
                    llvm::outs() << "  [PROPAGATE_TARGET] Marked bufferization.to_tensor (Target is permuted)\n";
                }
            }
          }
        }
      }
    }
  });

  // 删除被重写的旧 Op
  for (auto op : opsToErase) {
    op->erase();
  }

  // ==============================================================================
  // 2. 扫描所有 bufferization.to_tensor 操作 (查漏补缺)
  // ==============================================================================
  funcOp.walk([&](bufferization::ToTensorOp toTensorOp) {
    // 如果已经在列表中，跳过
    for(auto existing : toTensorOpsToMark) { if (existing == toTensorOp) return; }

    Value sourceMemRef = toTensorOp.getOperand();
    bool hasNonStandardStride = checkValueOriginHasNonStandardStride(sourceMemRef);
    
    bool shouldMark = false;
    if (auto memRefType = dyn_cast<MemRefType>(sourceMemRef.getType())) {
      if (isPermutedOrHasNonUnitLastStride(memRefType)) {
        shouldMark = true;
      }
    }
    
    if (shouldMark || hasNonStandardStride) {
      toTensorOpsToMark.push_back(toTensorOp);
      llvm::outs() << "[TO_TENSOR_CHECK] Found permuted/strided origin: " << toTensorOp << "\n";
    }
  });

  // ==============================================================================
  // 3. 执行最终标记：为收集到的 Tensor 添加 Annotation
  // ==============================================================================
  for (auto toTensorOp : toTensorOpsToMark) {
    // 双重检查：防止重复添加 MarkOp (虽然 OpBuilder 会创建新的 Op，但逻辑上我们不希望冗余)
    // 简单检查该 Value 是否已经被 MarkOp 使用
    bool alreadyMarked = false;
    // 注意：annotation::MarkOp 通常不直接作为 User 挂在 Value 上，而是作为一个独立的 Op 存在。
    // 为了稳妥，这里我们假设 list 中可能有重复（如果 func.walk 逻辑有交集），去重已经在 push_back 时做了。
    
    llvm::outs() << "  [ANNOTATE_ACTION] Adding annotation to: " << toTensorOp << "\n";
    
    OpBuilder builder(toTensorOp->getContext());
    builder.setInsertionPointAfter(toTensorOp); 

    auto markOp = builder.create<annotation::MarkOp>(
        toTensorOp->getLoc(), toTensorOp.getResult());
    
    markOp->setAttr("MayImplicitTransposeWithLastAxis", 
                   UnitAttr::get(builder.getContext()));

    llvm::outs() << "      -> Created annotation::MarkOp: " << markOp << "\n";
  }

  llvm::outs() << "[INFO] Finished AnnotateTransposePass on function: "
              << funcOp.getName() << "\n";
}

};
} // namespace

namespace mlir::dicp::LinalgExt {
std::unique_ptr<OperationPass<func::FuncOp>> createAnnotateTransposePass() {
  return std::make_unique<AnnotateTransposePass>();
}
} // namespace mlir::dicp::LinalgExt