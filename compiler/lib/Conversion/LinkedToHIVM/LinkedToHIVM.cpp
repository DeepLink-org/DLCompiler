// #include "bishengir/HIVM/HIVMSynchronization.h"
// #include "bishengir/HIVM/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "dicp/Conversion/LinkedToHIVM/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace dicp;
using namespace linked;
using namespace hivm;

#define GEN_PASS_CLASSES
#include "dicp/Conversion/LinkedToHIVM/Passes.h.inc"

namespace {

struct CoreAndPipes {
  TCoreTypeAttr core;
  PipeAttr producer;
  PipeAttr consumer;
};

static LogicalResult EmitUnknownOpError(Operation *op, llvm::StringRef opName) {
  op->emitError("Unknown custom operation: ") << opName;
  return failure();
}

static void CreateSyncBlock(PatternRewriter &rewriter, Location loc,
                            MLIRContext *ctx, Operation *op, int64_t id,
                            hivm::SyncBlockMode mode, PipeAttr pipe1,
                            PipeAttr pipe2) {
  auto syncMode = hivm::SyncBlockModeAttr::get(ctx, mode);
  auto newOp = rewriter.create<hivm::SyncBlockOp>(
      loc, syncMode, rewriter.getI16IntegerAttr(id), Value{}, pipe1, pipe2);
  rewriter.replaceOp(op, newOp);
}

static CoreAndPipes GetCoreAndPipes(MLIRContext *ctx, llvm::StringRef opName,
                                    llvm::StringRef sender) {
  // Step 1: Decide pipes
  PipeAttr producer;
  PipeAttr consumer = PipeAttr::get(ctx, PIPE::PIPE_MTE2);

  if (sender == "cube") {
    producer = PipeAttr::get(ctx, PIPE::PIPE_FIX);
  } else {
    producer = PipeAttr::get(ctx, PIPE::PIPE_MTE3);
  }

  // Step 2: Decide core type
  TCoreTypeAttr core;
  if (sender == "cube") {
    if (opName == "sync_block_set")
      core = TCoreTypeAttr::get(ctx, TCoreType::CUBE);
    else
      core = TCoreTypeAttr::get(ctx, TCoreType::VECTOR);
  } else {
    if (opName == "sync_block_set")
      core = TCoreTypeAttr::get(ctx, TCoreType::VECTOR);
    else
      core = TCoreTypeAttr::get(ctx, TCoreType::CUBE);
  }

  return {core, producer, consumer};
}

struct LinkedToHIVMPass : public LinkedToHIVMBase<LinkedToHIVMPass> {
  void runOnOperation() override;
};

struct TritonCustomSyncOpToHIVMSyncOpConversion
    : OpRewritePattern<triton::CustomSyncOp> {
  using OpRewritePattern<triton::CustomSyncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::CustomSyncOp op,
                                PatternRewriter &rewriter) const final {
    auto *ctx = op->getContext();
    auto loc = op->getLoc();
    llvm::StringRef opName = op.getOpName();
    llvm::StringRef arg = op.getModeOrSender();
    auto id = op.getId();

    if (opName == "sync_block_all") {
      if (arg == "all_cube") {
        CreateSyncBlock(rewriter, loc, ctx, op, id,
                        hivm::SyncBlockMode::ALL_CUBE,
                        PipeAttr::get(ctx, PIPE::PIPE_FIX), hivm::PipeAttr{});
      } else if (arg == "all_vector") {
        CreateSyncBlock(rewriter, loc, ctx, op, id,
                        hivm::SyncBlockMode::ALL_VECTOR, hivm::PipeAttr{},
                        PipeAttr::get(ctx, PIPE::PIPE_MTE3));
      } else if (arg == "all") {
        CreateSyncBlock(rewriter, loc, ctx, op, id, hivm::SyncBlockMode::ALL,
                        PipeAttr::get(ctx, PIPE::PIPE_FIX),
                        PipeAttr::get(ctx, PIPE::PIPE_MTE3));
      } else {
        return EmitUnknownOpError(op, opName);
      }
      return success();
    }

    if (opName == "sync_block_set") {
      auto [coreAttr, prodPipe, consPipe] = GetCoreAndPipes(ctx, opName, arg);
      rewriter.replaceOp(op, rewriter.create<hivm::SyncBlockSetOp>(
                                 loc, coreAttr, prodPipe, consPipe,
                                 rewriter.getIndexAttr(id)));
      return success();
    }

    if (opName == "sync_block_wait") {
      auto [coreAttr, prodPipe, consPipe] = GetCoreAndPipes(ctx, opName, arg);
      rewriter.replaceOp(op, rewriter.create<hivm::SyncBlockWaitOp>(
                                 loc, coreAttr, prodPipe, consPipe,
                                 rewriter.getIndexAttr(id)));
      return success();
    }

    return EmitUnknownOpError(op, opName);
  }
};

void LinkedToHIVMPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<hivm::HIVMDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<TritonCustomSyncOpToHIVMSyncOpConversion>(patterns.getContext());
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> linked::createLinkedToHIVMPass() {
  return std::make_unique<LinkedToHIVMPass>();
}