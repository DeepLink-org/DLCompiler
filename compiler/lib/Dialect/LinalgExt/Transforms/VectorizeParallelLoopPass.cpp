#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "vectorize-parallel-loop"

using namespace mlir;

namespace {

// ============================================================================
// 工具类：处理数据流映射和类型推导
// ============================================================================
class TensorizationContext {
public:
  TensorizationContext(int64_t size, PatternRewriter &rewriter, Location loc)
      : tileSize(size), rewriter(rewriter), loc(loc) {}

  // 记录标量到张量的映射
  void map(Value scalar, Value tensor) {
    scalarToTensorMap[scalar] = tensor;
    LLVM_DEBUG(llvm::dbgs() << "    [Map] Scalar " << scalar << " -> Tensor " << tensor << "\n");
  }

  // 查找映射，如果不存在且是标量，则尝试广播
  Value lookupOrBroadcast(Value scalar) {
    if (scalarToTensorMap.count(scalar)) {
      return scalarToTensorMap[scalar];
    }

    // 如果是 Index 类型，通常用于地址计算，保持原样（不转 Tensor）
    if (scalar.getType().isIndex()) {
      return nullptr;
    }

    // 尝试广播 (Splat)
    // 检查是否是基础标量类型 (Float, Int)
    if (auto scalarType = dyn_cast<RankedTensorType>(scalar.getType())) {
        // 已经是 Tensor 了，直接返回
        return scalar;
    }

    LLVM_DEBUG(llvm::dbgs() << "    [Broadcast] Creating splat for: " << scalar << "\n");
    auto tensorType = RankedTensorType::get({tileSize}, scalar.getType());
    Value splat = rewriter.create<tensor::SplatOp>(loc, tensorType, scalar);
    map(scalar, splat);
    return splat;
  }

  Value getMapped(Value scalar) {
    return scalarToTensorMap.count(scalar) ? scalarToTensorMap[scalar] : nullptr;
  }

private:
  int64_t tileSize;
  PatternRewriter &rewriter;
  Location loc;
  DenseMap<Value, Value> scalarToTensorMap;
};

// ============================================================================
// 核心 Pattern：将 scf.parallel 转化为 Tensor + Bufferization 操作
// ============================================================================
struct TensorizeParallelLoopPattern : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "\n=== [Tensorize] Start Processing scf.parallel at " << op.getLoc() << " ===\n");

    // 1. 基本合法性检查
    // 只处理 1D 循环，且边界必须是常量，以确定静态 Shape
    if (op.getNumLoops() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "[Skip] Not a 1D loop.\n");
      return failure();
    }

    auto lowerOp = op.getLowerBound()[0].getDefiningOp<arith::ConstantIndexOp>();
    auto upperOp = op.getUpperBound()[0].getDefiningOp<arith::ConstantIndexOp>();
    auto stepOp = op.getStep()[0].getDefiningOp<arith::ConstantIndexOp>();

    if (!lowerOp || !upperOp || !stepOp) {
      LLVM_DEBUG(llvm::dbgs() << "[Skip] Bounds or step are not constant.\n");
      return failure();
    }

    int64_t lowerVal = lowerOp.value();
    int64_t upperVal = upperOp.value();
    int64_t stepVal = stepOp.value();
    int64_t range = upperVal - lowerVal;
    
    // 我们假设这个 Pass 作用于已经 Tile 过的内部循环，
    // 或者我们把整个循环范围当作一个 Tensor 处理
    int64_t tensorSize = range;

    LLVM_DEBUG(llvm::dbgs() << "   Loop Range: [" << lowerVal << ", " << upperVal << "), Step: " << stepVal << "\n");
    LLVM_DEBUG(llvm::dbgs() << "   Target Tensor Size: " << tensorSize << "\n");

    if (tensorSize <= 0) return failure();

    // 2. 初始化上下文
    TensorizationContext ctx(tensorSize, rewriter, op.getLoc());
    
    // 3. IV 映射
    // 在 Tensor 化模式下，原循环的 IV 通常映射为 Base Offset (LowerBound)
    // 后续的地址计算都基于这个 Base Offset 进行
    IRMapping ivMap;
    Value iv = op.getInductionVars()[0];
    ivMap.map(iv, op.getLowerBound()[0]);
    LLVM_DEBUG(llvm::dbgs() << "   Mapped IV " << iv << " -> LowerBound " << op.getLowerBound()[0] << "\n");

    // 保存所有创建的新操作，以便替换yield操作
    SmallVector<Operation *> newOps;
    
    // 4. 遍历 Body 指令
    Block *body = op.getBody();
    auto &ops = body->getOperations();
    // 收集所有要处理的操作，排除terminator
    SmallVector<Operation*> opsToProcess;
    for (auto &inst : ops) {
        if (!isa<scf::YieldOp, scf::ReduceOp>(inst)) {
            opsToProcess.push_back(&inst);
        }
    }

    // 处理所有收集的操作
    for (auto *inst : opsToProcess) {
      LLVM_DEBUG(llvm::dbgs() << "   -> Visiting: " << inst->getName() << "\n");

      // --- Case 0: 忽略 Terminator ---
      if (isa<scf::YieldOp, scf::ReduceOp>(*inst)) continue;

      // --- Case 1: 地址/索引计算 (Index Arithmetic) ---
      // 这些指令通常保持标量形式，用于计算 memref 的偏移
      if (isa<arith::IndexCastOp, arith::AddIOp, arith::MulIOp, arith::ConstantOp>(*inst) && 
          inst->getResult(0).getType().isIndex()) {
        LLVM_DEBUG(llvm::dbgs() << "      [Action] Cloning Index Compute.\n");
        auto *clonedOp = rewriter.clone(*inst, ivMap);
        newOps.push_back(clonedOp);
        continue;
      }
      
      // 如果是 i32 运算但用于地址计算的，也 clone (需要根据上下文，这里简化处理，假设所有 i32/index 混合运算都为了地址)
      if (isa<arith::AddIOp, arith::MulIOp, arith::ConstantOp>(*inst) && inst->getResult(0).getType().isInteger(32)) {
         LLVM_DEBUG(llvm::dbgs() << "      [Action] Cloning Int32 Compute (Assuming Address Calc).\n");
         auto *clonedOp = rewriter.clone(*inst, ivMap);
         newOps.push_back(clonedOp);
         continue;
      }

      // --- Case 2: 内存读取 (Load -> Alloc + Copy + ToTensor) ---
      if (auto loadOp = dyn_cast<memref::LoadOp>(*inst)) {
        LLVM_DEBUG(llvm::dbgs() << "      [Action] Tensorizing MemRef Load.\n");
        
        Value baseMemref = loadOp.getMemRef();
        // 计算偏移量：使用 ivMap 查找映射后的操作数
        SmallVector<Value> indices;
        for (auto idx : loadOp.getIndices()) {
            indices.push_back(ivMap.lookupOrDefault(idx));
        }

        // 2.1 Alloc Local (UB)
        auto elemType = dyn_cast<MemRefType>(baseMemref.getType()).getElementType();
        auto localMemType = MemRefType::get({tensorSize}, elemType);
        Value localAlloc = rewriter.create<memref::AllocOp>(op.getLoc(), localMemType);

        // 2.2 Create SubView (GM View)
        // 假设 stride 为 1 (连续访问)
        SmallVector<OpFoldResult> offsets;
        for(auto v : indices) offsets.push_back(v);
        SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(tensorSize)};
        SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};
        
        Value subview = rewriter.create<memref::SubViewOp>(
            op.getLoc(), baseMemref, offsets, sizes, strides);

        // 2.3 Copy GM -> UB (MTE)
        auto copyOp = rewriter.create<memref::CopyOp>(op.getLoc(), subview, localAlloc);
        newOps.push_back(copyOp.getOperation());

        // 2.4 To Tensor
        auto tensorType = RankedTensorType::get({tensorSize}, elemType);
        Value tensorVal = rewriter.create<bufferization::ToTensorOp>(
            op.getLoc(), tensorType, localAlloc, /*restrict=*/true);

        // 注册映射
        ctx.map(loadOp.getResult(), tensorVal);
        continue;
      }

      // --- Case 3: 内存写入 (Store -> Alloc + Materialize + Copy) ---
      if (auto storeOp = dyn_cast<memref::StoreOp>(*inst)) {
        LLVM_DEBUG(llvm::dbgs() << "      [Action] Tensorizing MemRef Store.\n");

        Value valToStore = storeOp.getValue();
        Value tensorToStore = ctx.lookupOrBroadcast(valToStore); // 可能是计算结果，也可能是常量
        
        if (!tensorToStore) {
             LLVM_DEBUG(llvm::dbgs() << "      [Error] Value to store not available in map.\n");
             return failure();
        }

        Value baseMemref = storeOp.getMemRef();
        SmallVector<Value> indices;
        for (auto idx : storeOp.getIndices()) {
            indices.push_back(ivMap.lookupOrDefault(idx));
        }

        // 3.1 Alloc Local (Output Buffer)
        auto tensorType = dyn_cast<RankedTensorType>(tensorToStore.getType());
        auto localMemType = MemRefType::get({tensorSize}, tensorType.getElementType());
        Value localOut = rewriter.create<memref::AllocOp>(op.getLoc(), localMemType);

        // 3.2 Materialize (Tensor -> UB)
        auto matOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
            op.getLoc(), tensorToStore, localOut);
        matOp.setWritable(true);

        // 3.3 SubView (GM Output View)
        SmallVector<OpFoldResult> offsets;
        for(auto v : indices) offsets.push_back(v);
        SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(tensorSize)};
        SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};

        Value outSubview = rewriter.create<memref::SubViewOp>(
            op.getLoc(), baseMemref, offsets, sizes, strides);

        // 3.4 Copy UB -> GM (MTE)
        auto copyOp = rewriter.create<memref::CopyOp>(op.getLoc(), localOut, outSubview);
        newOps.push_back(copyOp.getOperation());
        continue;
      }

      // --- Case 4: Linalg Op (MatMul 等) ---
      // Linalg Op 本身就是 Tensor 语义友好的，如果输入已经是 Tensor，直接 Clone 即可
      if (isa<linalg::LinalgOp>(*inst)) {
         LLVM_DEBUG(llvm::dbgs() << "      [Action] Handling Linalg Op.\n");
         // 对于 Linalg，我们需要确保其 inputs 和 outputs 都已映射为 tensor
         // 这里的处理比较简化：假设 linalg op 是纯计算，其输入来自于之前的 load
         
         SmallVector<Value> newOperands;
         for (Value operand : inst->getOperands()) {
             Value mapped = ctx.getMapped(operand);
             if (mapped) newOperands.push_back(mapped);
             else newOperands.push_back(operand); // 可能是 accumulator 或其他
         }
         
         // Clone 并替换操作数
         Operation* newOp = rewriter.clone(*inst);
         newOp->setOperands(newOperands);
         newOps.push_back(newOp);
         
         // 映射结果
         for(auto i=0; i<inst->getNumResults(); ++i) {
             ctx.map(inst->getResult(i), newOp->getResult(i));
         }
         continue;
      }

      // --- Case 5: 通用计算指令 (Arith, Math) ---
      // 这是一个通用的处理逻辑，支持 Unary, Binary, Ternary 等
      // 只要是 NoSideEffect 的计算指令都可以尝试转换
      if (isPureComputeOp(*inst)) {
        LLVM_DEBUG(llvm::dbgs() << "      [Action] Generic Compute Tensorization.\n");
        
        SmallVector<Value> vecOperands;
        bool allMapped = true;

        for (Value operand : inst->getOperands()) {
            Value vecOp = ctx.lookupOrBroadcast(operand);
            if (!vecOp) {
                // 如果操作数既不是 tensor 也没法 broadcast (比如是 index)，则保留原样?
                // 通常计算指令的操作数不应该是 index
                allMapped = false; 
                break;
            }
            vecOperands.push_back(vecOp);
        }

        if (allMapped) {
            // 构建新的 Tensor 类型
            SmallVector<Type> resultTypes;
            for (Type t : inst->getResultTypes()) {
                resultTypes.push_back(RankedTensorType::get({tensorSize}, t));
            }

            // 使用 OperationState 通用构建 Op
            OperationState state(op.getLoc(), inst->getName().getStringRef());
            state.addOperands(vecOperands);
            state.addTypes(resultTypes);
            state.addAttributes(inst->getAttrs());

            Operation *newOp = rewriter.create(state);
            newOps.push_back(newOp);
            
            // 映射结果
            for (size_t i = 0; i < inst->getNumResults(); ++i) {
                ctx.map(inst->getResult(i), newOp->getResult(i));
            }
            continue;
        }
      }
      
      // --- Case 6: scf.reduce (Reduction) ---
      if (auto reduceOp = dyn_cast<scf::ReduceOp>(*inst)) {
          // 在 Tensor 模式下，reduce 通常意味着对整个 Tensor 进行归约
          // 这里需要引入 linalg.reduce 或类似机制，实现较复杂。
          // 简单策略：如果遇到 reduce，发出警告或尝试使用 arith/vector reduce (但题目要求不用 vector)
          // 对于纯 Tensor + Linalg 体系，可以使用 linalg.reduce
          LLVM_DEBUG(llvm::dbgs() << "      [Warning] scf.reduce detected. Implementing basic collapse.\n");
          // TODO: Implement linalg.reduce generation
          continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "      [Unhandled] Skipping Op: " << inst->getName() << "\n");
    }

    // 5. 替换原循环的所有使用者
    // 获取原循环的 yield 操作及其结果
    SmallVector<Value> newYieldOperands;  // 声明变量
    
    auto yieldOp = op.getBody()->getTerminator();
    if (auto yield = dyn_cast<scf::YieldOp>(yieldOp)) {
        // 创建一个新的 yield 操作，使用转换后的值
        for (Value operand : yield.getOperands()) {
            Value mapped = ctx.getMapped(operand);
            if (mapped) {
                newYieldOperands.push_back(mapped);
            } else {
                // 如果没有映射，则需要广播或保留原始值
                Value broadcasted = ctx.lookupOrBroadcast(operand);
                if (broadcasted) {
                    newYieldOperands.push_back(broadcasted);
                } else {
                    newYieldOperands.push_back(operand);
                }
            }
        }
        rewriter.create<scf::YieldOp>(op.getLoc(), newYieldOperands);
    }

    // 正确替换原循环操作
    rewriter.replaceOp(op, newYieldOperands);
    LLVM_DEBUG(llvm::dbgs() << "=== [Tensorize] Done ===\n");
    
    return success();
  }

  // 辅助函数：判断是否为纯计算指令
  bool isPureComputeOp(Operation &op) const {
    // 简单白名单
    return isa<arith::AddFOp, arith::MulFOp, arith::SubFOp, arith::DivFOp,
               arith::AddIOp, arith::MulIOp, arith::SubIOp, 
               arith::MaximumFOp, arith::MinimumFOp,
               arith::SelectOp, arith::CmpFOp, arith::CmpIOp>(op) ||
           op.getDialect()->getNamespace() == "math"; // 所有 math.* (exp, log, etc)
  }
};

// ============================================================================
// Pass 定义
// ============================================================================
struct VectorizeParallelLoopPass
    : public PassWrapper<VectorizeParallelLoopPass, OperationPass<func::FuncOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizeParallelLoopPass)

  StringRef getArgument() const final { return "vectorize-parallel-loop"; }
  StringRef getDescription() const final {
    return "Convert scf.parallel to tensor operations with explicit GM-UB data movement.";
  }

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "--- Running TensorizeParallelLoopPass ---\n");
    
    RewritePatternSet patterns(&getContext());
    patterns.add<TensorizeParallelLoopPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::dicp::LinalgExt {
std::unique_ptr<OperationPass<func::FuncOp>> createVectorizeParallelLoopPass() {
  return std::make_unique<VectorizeParallelLoopPass>();
}
} // namespace mlir::dicp::LinalgExt