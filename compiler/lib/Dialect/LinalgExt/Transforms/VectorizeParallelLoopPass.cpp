#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir {
namespace dicp {
namespace LinalgExt {
#define GEN_PASS_DEF_VECTORIZEPARALLELLOOPPASS
#include "dicp/Dialect/LinalgExt/Transforms/Passes.h.inc"
} // namespace LinalgExt
} // namespace dicp
} // namespace mlir

#define DEBUG_TYPE "vectorize-parallel-loop-pass"

namespace {

// 核心 Pattern：将标量并行循环展开为向量化的顺序操作
struct VectorizeParallelLoopPattern : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(
        llvm::dbgs()
        << "\n[VectorizeParallelLoop] >>> Start matching scf.parallel at "
        << op.getLoc() << "\n");

    // 1. 检查循环结构
    if (op.getNumLoops() != 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[VectorizeParallelLoop] Skip: Multi-dimensional loop.\n");
      return failure();
    }

    Value lowerBound = op.getLowerBound()[0];
    Value upperBound = op.getUpperBound()[0];

    auto lowerOp = lowerBound.getDefiningOp<arith::ConstantIndexOp>();
    auto upperOp = upperBound.getDefiningOp<arith::ConstantIndexOp>();

    if (!lowerOp || !upperOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[VectorizeParallelLoop] Skip: Bounds are not constant.\n");
      return failure();
    }

    int64_t lowerVal = lowerOp.value();
    int64_t upperVal = upperOp.value();
    int64_t size = upperVal - lowerVal;

    LLVM_DEBUG(llvm::dbgs() << "[VectorizeParallelLoop] Loop Bounds: ["
                            << lowerVal << ", " << upperVal << ")\n");
    LLVM_DEBUG(llvm::dbgs()
               << "[VectorizeParallelLoop] Calculated Vector Size: " << size
               << "\n");

    // 只有当有实际计算量时才处理
    if (size <= 0) {
      LLVM_DEBUG(llvm::dbgs() << "[VectorizeParallelLoop] Skip: Size <= 0.\n");
      return failure();
    }

    // 2. 准备映射表
    // mapper: 用于处理索引计算 (将 Loop IV 映射为常数 LowerBound)
    IRMapping mapper;
    Block *body = op.getBody();
    Value iv = body->getArgument(0);

    LLVM_DEBUG(llvm::dbgs()
               << "[VectorizeParallelLoop] Mapping Induction Variable " << iv
               << " -> Constant " << lowerBound << "\n");
    mapper.map(iv, lowerBound); // 关键修复：将 IV 替换为 Loop 起始值

    // scalarToTensorMap: 用于数据流向量化 (标量 Value -> 向量 Tensor Value)
    DenseMap<Value, Value> scalarToTensorMap;

    LLVM_DEBUG(
        llvm::dbgs()
        << "[VectorizeParallelLoop] Starting to process body operations...\n");

    // 3. 遍历原循环体，按顺序生成向量化代码
    for (Operation &inst : body->getOperations()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  -> Visiting Op: " << inst.getName() << "\n");

      // 跳过 terminator
      if (isa<scf::ReduceOp>(inst) || isa<scf::YieldOp>(inst)) {
        LLVM_DEBUG(llvm::dbgs() << "     Skipping terminator.\n");
        continue;
      }

      // --- Case A: 索引计算 (Index Cast, Add, Mul 等) ---
      // 直接克隆，但使用 mapper 将 IV 替换为常数
      if (isa<arith::IndexCastOp, arith::AddIOp, arith::MulIOp,
              arith::ConstantOp>(inst)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "     [Action] Cloning index calculation.\n");
        Operation *newOp = rewriter.clone(inst, mapper);
        LLVM_DEBUG(llvm::dbgs()
                   << "     New Op result: " << newOp->getResult(0) << "\n");
        continue;
      }

      // --- Case B: 读取内存 (Load -> Vectorize) ---
      if (auto loadOp = dyn_cast<memref::LoadOp>(inst)) {
        LLVM_DEBUG(llvm::dbgs() << "     [Action] Vectorizing LoadOp.\n");
        Value memref = loadOp.getMemRef();
        // 获取计算好的索引 (通过 mapper 查找)
        Value index = mapper.lookup(loadOp.getIndices()[0]);
        LLVM_DEBUG(llvm::dbgs() << "     Base MemRef: " << memref << "\n");
        LLVM_DEBUG(llvm::dbgs() << "     Mapped Index: " << index << "\n");

        // 1. Alloc Local Buffer
        auto memrefType = dyn_cast<MemRefType>(memref.getType());
        if (!memrefType) {
          LLVM_DEBUG(llvm::dbgs()
                     << "[VectorizeParallelLoop] ERROR: MemRef type expected "
                        "but not found.\n");
          return failure();
        }
        auto localType = MemRefType::get({size}, memrefType.getElementType());
        Value localAlloc =
            rewriter.create<memref::AllocOp>(op.getLoc(), localType);
        LLVM_DEBUG(llvm::dbgs() << "     Created Local Alloc: "
                                << localAlloc.getType() << "\n");

        // 2. Subview Global Memory
        SmallVector<OpFoldResult> offsets = {index};
        SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(size)};
        SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};
        Value subview = rewriter.create<memref::SubViewOp>(
            op.getLoc(), memref, offsets, sizes, strides);
        LLVM_DEBUG(llvm::dbgs() << "     Created Subview.\n");

        // 3. Copy Global -> Local
        rewriter.create<memref::CopyOp>(op.getLoc(), subview, localAlloc);
        LLVM_DEBUG(llvm::dbgs() << "     Created Copy (Global -> Local).\n");

        // 4. Local Buffer -> Tensor
        auto tensorType =
            RankedTensorType::get({size}, memrefType.getElementType());
        auto toTensor = rewriter.create<bufferization::ToTensorOp>(
            op.getLoc(), tensorType, localAlloc, /*restrict=*/true);
        LLVM_DEBUG(llvm::dbgs() << "     Created ToTensorOp (Result: "
                                << toTensor.getResult() << ").\n");

        // 5. 注册映射：原 Load 的标量结果 -> 新的 Tensor 结果
        scalarToTensorMap[loadOp.getResult()] = toTensor.getResult();
        continue;
      }

      // --- Case C: 计算逻辑 (Generic Binary Operations -> Vector Binary
      // Operations) --- 检查是否为二元运算操作
      bool isBinaryOp =
          inst.getNumOperands() == 2 &&
          (isa<arith::AddFOp, arith::MulFOp, arith::AddIOp, arith::MulIOp,
               arith::SubFOp, arith::SubIOp, arith::DivFOp, arith::DivSIOp,
               arith::DivUIOp>(inst));

      if (isBinaryOp) {
        LLVM_DEBUG(llvm::dbgs() << "     [Action] Processing Binary ArithOp: "
                                << inst.getName() << "\n");

        Value lhs = inst.getOperand(0);
        Value rhs = inst.getOperand(1);

        // 检查操作数是否已向量化
        Value vecLhs =
            scalarToTensorMap.count(lhs) ? scalarToTensorMap[lhs] : nullptr;
        Value vecRhs =
            scalarToTensorMap.count(rhs) ? scalarToTensorMap[rhs] : nullptr;

        if (vecLhs)
          LLVM_DEBUG(llvm::dbgs() << "     LHS is vectorized.\n");
        if (vecRhs)
          LLVM_DEBUG(llvm::dbgs() << "     RHS is vectorized.\n");

        // 如果两个输入都是向量，生成向量运算
        if (vecLhs && vecRhs) {
          // 创建一个新的OperationState，使用与原操作相同的操作码
          OperationState state(op.getLoc(), inst.getName().getStringRef());

          // 添加向量化的操作数
          state.addOperands({vecLhs, vecRhs});

          // 从原操作复制结果类型，但转换为向量类型
          llvm::SmallVector<Type> resultTypes;
          for (auto result : inst.getResults()) {
            Type scalarType = result.getType();
            ShapedType vectorType;

            if (auto shapedType = dyn_cast<ShapedType>(scalarType)) {
              // 如果已经是shaped type，则保持形状但可能更新为tensor类型
              vectorType = RankedTensorType::get(shapedType.getShape(),
                                                 shapedType.getElementType());
            } else {
              // 如果是标量类型，转换为对应元素类型的向量
              vectorType = RankedTensorType::get({size}, scalarType);
            }

            resultTypes.push_back(vectorType);
          }
          state.addTypes(resultTypes);

          // 创建新的向量化操作
          auto newOp = rewriter.create(state);

          // 将新操作的结果映射到scalarToTensorMap
          for (size_t i = 0; i < inst.getNumResults(); ++i) {
            scalarToTensorMap[inst.getResult(i)] = newOp->getResult(i);
          }

          LLVM_DEBUG({
            llvm::dbgs() << "     Created Vector Operation: " << inst.getName()
                         << "\n";
            llvm::dbgs() << "     Result Type: "
                         << newOp->getResult(0).getType() << "\n";
          });
        } else {
          // 如果不是向量操作（可能是索引计算的一部分），则回退到普通 clone
          LLVM_DEBUG(
              llvm::dbgs()
              << "     WARNING: Operands not vectorized, cloning scalar op.\n");
          rewriter.clone(inst, mapper);
        }
        continue;
      }

      // --- Case D: 写回逻辑 (Materialize) ---
      if (auto matOp =
              dyn_cast<bufferization::MaterializeInDestinationOp>(inst)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "     [Action] Processing MaterializeInDestinationOp.\n");
        Value source = matOp.getSource();
        Value destMemref = matOp.getDest();

        Value vectorResult = nullptr;

        // 追踪数据来源
        if (auto insertOp = source.getDefiningOp<tensor::InsertOp>()) {
          LLVM_DEBUG(
              llvm::dbgs()
              << "     Source is tensor.insert, tracing scalar input...\n");
          Value scalarInput = insertOp.getScalar();
          if (scalarToTensorMap.count(scalarInput)) {
            vectorResult = scalarToTensorMap[scalarInput];
            LLVM_DEBUG(llvm::dbgs() << "     Found vectorized source.\n");
          }
        } else if (scalarToTensorMap.count(source)) {
          vectorResult = scalarToTensorMap[source];
          LLVM_DEBUG(llvm::dbgs()
                     << "     Found vectorized source directly.\n");
        }

        if (vectorResult) {
          // 1. Alloc Output Buffer
          auto tensorType = dyn_cast<RankedTensorType>(vectorResult.getType());
          if (!tensorType) {
            LLVM_DEBUG(llvm::dbgs()
                       << "[VectorizeParallelLoop] ERROR: Expected "
                          "RankedTensorType for vector result.\n");
            continue;
          }
          auto elemType = tensorType.getElementType();
          auto localOutType = MemRefType::get({size}, elemType);
          Value localOut =
              rewriter.create<memref::AllocOp>(op.getLoc(), localOutType);
          LLVM_DEBUG(llvm::dbgs() << "     Created Local Output Alloc: "
                                  << localOutType << "\n");

          // 2. Materialize Tensor -> Local Buffer
          // Fix: capture operation and set writable to true
          auto newMatOp =
              rewriter.create<bufferization::MaterializeInDestinationOp>(
                  op.getLoc(), vectorResult, localOut);
          newMatOp.setWritable(true);
          LLVM_DEBUG(
              llvm::dbgs()
              << "     Created Vectorized Materialize (writable=true).\n");

          // 3. 处理输出地址 (ReinterpretCast -> Subview)
          Value baseMemref = destMemref;
          Value writeOffset = nullptr;

          if (auto castOp =
                  destMemref.getDefiningOp<memref::ReinterpretCastOp>()) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "     Dest is ReinterpretCast, resolving offset...\n");
            baseMemref = castOp.getSource();
            if (!castOp.getOffsets().empty()) {
              // Fix: Directly use the Value, do not use dyn_cast<Value>
              Value loopOffset = castOp.getOffsets()[0];
              writeOffset = mapper.lookup(loopOffset);
              LLVM_DEBUG(llvm::dbgs() << "     Resolved write offset: "
                                      << writeOffset << "\n");
            }
          } else {
            LLVM_DEBUG(llvm::dbgs()
                       << "     Dest is not ReinterpretCast. Handling logic "
                          "might be incomplete for simple memrefs.\n");
          }

          // 如果找到了写入位置，执行 Copy Local -> Global
          if (baseMemref && writeOffset) {
            SmallVector<OpFoldResult> offsets = {writeOffset};
            SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(size)};
            SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};

            Value outSubview = rewriter.create<memref::SubViewOp>(
                op.getLoc(), baseMemref, offsets, sizes, strides);

            rewriter.create<memref::CopyOp>(op.getLoc(), localOut, outSubview);
            LLVM_DEBUG(llvm::dbgs()
                       << "     Created Copy (Local -> Global).\n");
          }
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "     WARNING: Could not find vectorized source for "
                        "materialize.\n");
        }
        continue;
      }

      // 忽略不需要的操作
      if (isa<tensor::InsertOp>(inst)) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "     Skipping tensor.insert (handled in materialize).\n");
        continue;
      }
      if (isa<tensor::EmptyOp>(inst)) {
        LLVM_DEBUG(llvm::dbgs() << "     Skipping tensor.empty.\n");
        continue;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "     [Unhandled] Operation not handled specifically: "
                 << inst.getName() << "\n");
    }

    // 打印当前op
    LLVM_DEBUG({
      llvm::dbgs() << "[VectorizeParallelLoop] Current Op: ";
      op.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
    // 打印映射表
    LLVM_DEBUG({
      llvm::dbgs() << "[VectorizeParallelLoop] Scalar to Tensor Map:\n";
      for (auto &[scalar, tensor] : scalarToTensorMap) {
        llvm::dbgs() << "  " << scalar << " -> " << tensor << "\n";
      }
    });

    // 4. 删除原循环
    LLVM_DEBUG(
        llvm::dbgs()
        << "[VectorizeParallelLoop] Erasing original scf.parallel op.\n");
    rewriter.eraseOp(op);

    LLVM_DEBUG(llvm::dbgs()
               << "[VectorizeParallelLoop] <<< MatchAndRewrite Done.\n\n");
    return success();
  }
};

struct VectorizeParallelLoopPass
    : public PassWrapper<VectorizeParallelLoopPass,
                         OperationPass<func::FuncOp>> {
  StringRef getArgument() const final { return "vectorize-parallel-loop"; }
  StringRef getDescription() const final {
    return "Vectorize scf.parallel loops by unrolling and using bulk memory "
           "ops.";
  }

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs()
               << "[Pass] Starting VectorizeParallelLoopPass on function...\n");
    RewritePatternSet patterns(&getContext());
    patterns.add<VectorizeParallelLoopPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "[Pass] Pattern application failed.\n");
      signalPassFailure();
    } else {
      LLVM_DEBUG(llvm::dbgs() << "[Pass] Pattern application succeeded.\n");
    }
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizeParallelLoopPass)
};

} // namespace

namespace mlir::dicp::LinalgExt {
std::unique_ptr<OperationPass<func::FuncOp>> createVectorizeParallelLoopPass() {
  return std::make_unique<VectorizeParallelLoopPass>();
}
} // namespace mlir::dicp::LinalgExt