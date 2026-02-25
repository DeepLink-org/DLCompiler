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

namespace {

// 核心 Pattern：将标量并行循环展开为向量化的顺序操作
struct VectorizeParallelLoopPattern : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    llvm::outs()
        << "\n[VectorizeParallelLoop] >>> Start matching scf.parallel at "
        << op.getLoc() << "\n";

    // 1. 检查循环结构
    if (op.getNumLoops() != 1) {
      llvm::outs() << "[VectorizeParallelLoop] Skip: Multi-dimensional loop.\n";
      return failure();
    }

    Value lowerBound = op.getLowerBound()[0];
    Value upperBound = op.getUpperBound()[0];

    auto lowerOp = lowerBound.getDefiningOp<arith::ConstantIndexOp>();
    auto upperOp = upperBound.getDefiningOp<arith::ConstantIndexOp>();

    if (!lowerOp || !upperOp) {
      llvm::outs()
          << "[VectorizeParallelLoop] Skip: Bounds are not constant.\n";
      return failure();
    }

    int64_t lowerVal = lowerOp.value();
    int64_t upperVal = upperOp.value();
    int64_t size = upperVal - lowerVal;

    llvm::outs() << "[VectorizeParallelLoop] Loop Bounds: [" << lowerVal << ", "
                 << upperVal << ")\n";
    llvm::outs() << "[VectorizeParallelLoop] Calculated Vector Size: " << size
                 << "\n";

    // 只有当有实际计算量时才处理
    if (size <= 0) {
      llvm::outs() << "[VectorizeParallelLoop] Skip: Size <= 0.\n";
      return failure();
    }

    // 2. 准备映射表
    // mapper: 用于处理索引计算 (将 Loop IV 映射为常数 LowerBound)
    IRMapping mapper;
    Block *body = op.getBody();
    Value iv = body->getArgument(0);

    llvm::outs() << "[VectorizeParallelLoop] Mapping Induction Variable " << iv
                 << " -> Constant " << lowerBound << "\n";
    mapper.map(iv, lowerBound); // 关键修复：将 IV 替换为 Loop 起始值

    // scalarToTensorMap: 用于数据流向量化 (标量 Value -> 向量 Tensor Value)
    DenseMap<Value, Value> scalarToTensorMap;

    llvm::outs()
        << "[VectorizeParallelLoop] Starting to process body operations...\n";

    // 3. 遍历原循环体，按顺序生成向量化代码
    for (Operation &inst : body->getOperations()) {
      llvm::outs() << "  -> Visiting Op: " << inst.getName() << "\n";

      // 跳过 terminator
      if (isa<scf::ReduceOp>(inst) || isa<scf::YieldOp>(inst)) {
        llvm::outs() << "     Skipping terminator.\n";
        continue;
      }

      // --- Case A: 索引计算 (Index Cast, Add, Mul 等) ---
      // 直接克隆，但使用 mapper 将 IV 替换为常数
      if (isa<arith::IndexCastOp, arith::AddIOp, arith::MulIOp,
              arith::ConstantOp>(inst)) {
        llvm::outs() << "     [Action] Cloning index calculation.\n";
        Operation *newOp = rewriter.clone(inst, mapper);
        llvm::outs() << "     New Op result: " << newOp->getResult(0) << "\n";
        continue;
      }

      // --- Case B: 读取内存 (Load -> Vectorize) ---
      if (auto loadOp = dyn_cast<memref::LoadOp>(inst)) {
        llvm::outs() << "     [Action] Vectorizing LoadOp.\n";
        Value memref = loadOp.getMemRef();
        // 获取计算好的索引 (通过 mapper 查找)
        Value index = mapper.lookup(loadOp.getIndices()[0]);
        llvm::outs() << "     Base MemRef: " << memref << "\n";
        llvm::outs() << "     Mapped Index: " << index << "\n";

        // 1. Alloc Local Buffer
        auto memrefType = dyn_cast<MemRefType>(memref.getType());
        if (!memrefType) {
          llvm::outs() << "[VectorizeParallelLoop] ERROR: MemRef type expected "
                          "but not found.\n";
          return failure();
        }
        auto localType = MemRefType::get({size}, memrefType.getElementType());
        Value localAlloc =
            rewriter.create<memref::AllocOp>(op.getLoc(), localType);
        llvm::outs() << "     Created Local Alloc: " << localAlloc.getType()
                     << "\n";

        // 2. Subview Global Memory
        SmallVector<OpFoldResult> offsets = {index};
        SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(size)};
        SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};
        Value subview = rewriter.create<memref::SubViewOp>(
            op.getLoc(), memref, offsets, sizes, strides);
        llvm::outs() << "     Created Subview.\n";

        // 3. Copy Global -> Local
        rewriter.create<memref::CopyOp>(op.getLoc(), subview, localAlloc);
        llvm::outs() << "     Created Copy (Global -> Local).\n";

        // 4. Local Buffer -> Tensor
        auto tensorType =
            RankedTensorType::get({size}, memrefType.getElementType());
        auto toTensor = rewriter.create<bufferization::ToTensorOp>(
            op.getLoc(), tensorType, localAlloc, /*restrict=*/true);
        llvm::outs() << "     Created ToTensorOp (Result: "
                     << toTensor.getResult() << ").\n";

        // 5. 注册映射：原 Load 的标量结果 -> 新的 Tensor 结果
        scalarToTensorMap[loadOp.getResult()] = toTensor.getResult();
        continue;
      }

      // --- Case C: 计算逻辑 (AddF -> Vector AddF) ---
      if (auto addFOp = dyn_cast<arith::AddFOp>(inst)) {
        llvm::outs() << "     [Action] Processing ArithOp (AddF).\n";
        Value lhs = addFOp.getLhs();
        Value rhs = addFOp.getRhs();

        // 检查操作数是否已向量化
        Value vecLhs =
            scalarToTensorMap.count(lhs) ? scalarToTensorMap[lhs] : nullptr;
        Value vecRhs =
            scalarToTensorMap.count(rhs) ? scalarToTensorMap[rhs] : nullptr;

        if (vecLhs)
          llvm::outs() << "     LHS is vectorized.\n";
        if (vecRhs)
          llvm::outs() << "     RHS is vectorized.\n";

        // 如果两个输入都是向量，生成向量加法
        if (vecLhs && vecRhs) {
          Value vecRes =
              rewriter.create<arith::AddFOp>(op.getLoc(), vecLhs, vecRhs);
          scalarToTensorMap[addFOp.getResult()] = vecRes;
          llvm::outs() << "     Created Vector AddF: " << vecRes.getType()
                       << "\n";
        } else {
          // 如果不是向量操作（可能是索引计算的一部分），则回退到普通 clone
          llvm::outs()
              << "     WARNING: Operands not vectorized, cloning scalar op.\n";
          rewriter.clone(inst, mapper);
        }
        continue;
      }

      // --- Case D: 写回逻辑 (Materialize) ---
      if (auto matOp =
              dyn_cast<bufferization::MaterializeInDestinationOp>(inst)) {
        llvm::outs()
            << "     [Action] Processing MaterializeInDestinationOp.\n";
        Value source = matOp.getSource();
        Value destMemref = matOp.getDest();

        Value vectorResult = nullptr;

        // 追踪数据来源
        if (auto insertOp = source.getDefiningOp<tensor::InsertOp>()) {
          llvm::outs()
              << "     Source is tensor.insert, tracing scalar input...\n";
          Value scalarInput = insertOp.getScalar();
          if (scalarToTensorMap.count(scalarInput)) {
            vectorResult = scalarToTensorMap[scalarInput];
            llvm::outs() << "     Found vectorized source.\n";
          }
        } else if (scalarToTensorMap.count(source)) {
          vectorResult = scalarToTensorMap[source];
          llvm::outs() << "     Found vectorized source directly.\n";
        }

        if (vectorResult) {
          // 1. Alloc Output Buffer
          auto tensorType = dyn_cast<RankedTensorType>(vectorResult.getType());
          if (!tensorType) {
            llvm::outs() << "[VectorizeParallelLoop] ERROR: Expected "
                            "RankedTensorType for vector result.\n";
            continue;
          }
          auto elemType = tensorType.getElementType();
          auto localOutType = MemRefType::get({size}, elemType);
          Value localOut =
              rewriter.create<memref::AllocOp>(op.getLoc(), localOutType);
          llvm::outs() << "     Created Local Output Alloc: " << localOutType
                       << "\n";

          // 2. Materialize Tensor -> Local Buffer
          // Fix: capture operation and set writable to true
          auto newMatOp =
              rewriter.create<bufferization::MaterializeInDestinationOp>(
                  op.getLoc(), vectorResult, localOut);
          newMatOp.setWritable(true);
          llvm::outs()
              << "     Created Vectorized Materialize (writable=true).\n";

          // 3. 处理输出地址 (ReinterpretCast -> Subview)
          Value baseMemref = destMemref;
          Value writeOffset = nullptr;

          if (auto castOp =
                  destMemref.getDefiningOp<memref::ReinterpretCastOp>()) {
            llvm::outs()
                << "     Dest is ReinterpretCast, resolving offset...\n";
            baseMemref = castOp.getSource();
            if (!castOp.getOffsets().empty()) {
              // Fix: Directly use the Value, do not use dyn_cast<Value>
              Value loopOffset = castOp.getOffsets()[0];
              writeOffset = mapper.lookup(loopOffset);
              llvm::outs() << "     Resolved write offset: " << writeOffset
                           << "\n";
            }
          } else {
            llvm::outs() << "     Dest is not ReinterpretCast. Handling logic "
                            "might be incomplete for simple memrefs.\n";
          }

          // 如果找到了写入位置，执行 Copy Local -> Global
          if (baseMemref && writeOffset) {
            SmallVector<OpFoldResult> offsets = {writeOffset};
            SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(size)};
            SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};

            Value outSubview = rewriter.create<memref::SubViewOp>(
                op.getLoc(), baseMemref, offsets, sizes, strides);

            rewriter.create<memref::CopyOp>(op.getLoc(), localOut, outSubview);
            llvm::outs() << "     Created Copy (Local -> Global).\n";
          }
        } else {
          llvm::outs() << "     WARNING: Could not find vectorized source for "
                          "materialize.\n";
        }
        continue;
      }

      // 忽略不需要的操作
      if (isa<tensor::InsertOp>(inst)) {
        llvm::outs()
            << "     Skipping tensor.insert (handled in materialize).\n";
        continue;
      }
      if (isa<tensor::EmptyOp>(inst)) {
        llvm::outs() << "     Skipping tensor.empty.\n";
        continue;
      }

      llvm::outs() << "     [Unhandled] Operation not handled specifically: "
                   << inst.getName() << "\n";
    }

    // 打印当前op
    llvm::outs() << "[VectorizeParallelLoop] Current Op: ";
    op.print(llvm::outs());
    llvm::outs() << "\n";
    // 打印映射表
    llvm::outs() << "[VectorizeParallelLoop] Scalar to Tensor Map:\n";
    for (auto &[scalar, tensor] : scalarToTensorMap) {
      llvm::outs() << "  " << scalar << " -> " << tensor << "\n";
    }

    // 4. 删除原循环
    llvm::outs()
        << "[VectorizeParallelLoop] Erasing original scf.parallel op.\n";
    rewriter.eraseOp(op);

    llvm::outs() << "[VectorizeParallelLoop] <<< MatchAndRewrite Done.\n\n";
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
    llvm::outs()
        << "[Pass] Starting VectorizeParallelLoopPass on function...\n";
    RewritePatternSet patterns(&getContext());
    patterns.add<VectorizeParallelLoopPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      llvm::outs() << "[Pass] Pattern application failed.\n";
      signalPassFailure();
    } else {
      llvm::outs() << "[Pass] Pattern application succeeded.\n";
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