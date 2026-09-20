#include "triton/Dialect/Triton/IR/Dialect.h"

#include "triton-shared/Conversion/MemrefCopyToDMA_FlagTree/MemrefCopyToDMAFlagTree.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <optional>

#define DEBUG_TYPE "structured-to-memref-flagtree"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_MEMREFCOPYTODMAFLAGTREE
#include "triton-shared/Conversion/MemrefCopyToDMA_FlagTree/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class PtrToUnrankedMemrefConverter : public TypeConverter {
public:
  PtrToUnrankedMemrefConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrType) {
      return UnrankedMemRefType::get(ptrType.getPointeeType(), 0);
    });
    addTargetMaterialization([&](OpBuilder &builder,
                                 UnrankedMemRefType resultType,
                                 ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
  }
};

class MemrefCopyToDMAFlagTreePass
    : public triton::impl::MemrefCopyToDMAFlagTreeBase<
          MemrefCopyToDMAFlagTreePass> {
  using MemrefCopyToDMAFlagTreeBase<
      MemrefCopyToDMAFlagTreePass>::MemrefCopyToDMAFlagTreeBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    linalg::LinalgDialect, affine::AffineDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    bufferization::BufferizationDialect, triton::TritonDialect,
                    ttx::TritonTilingExtDialect, memref::MemRefDialect>();
  }

  // Rewrite on the copied module. Ignore if there is an failure; otherwise,
  // replace the original
  void runOnOperation() override {
    auto module = getOperation();
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, ttx::TritonTilingExtDialect,
        memref::MemRefDialect>();

    target.addIllegalOp<tts::LoadOp, tts::StoreOp, tts::MakeTensorPtrOp>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    target.addIllegalOp<memref::CopyOp>();

    PtrToUnrankedMemrefConverter typeConverter;

    SmallVector<std::pair<mlir::func::FuncOp, mlir::func::FuncOp>> replacements;
    module.walk([&](mlir::func::FuncOp funcOp) {
      mlir::func::FuncOp cloned = funcOp.clone();
      cloned->setAttrs(funcOp->getAttrs());
      cloned.setPublic();

      RewritePatternSet localPatterns(&getContext());
      {
        PtrToUnrankedMemrefConverter typeConverter;
        triton::populateMemrefCopyToDMAFlagTreeConversionPatterns(
            localPatterns, typeConverter);
      }

      // Try to apply partial conversion on the cloned operation.
      // If it fails, erase the cloned op and return.
      if (failed(applyPartialConversion(cloned.getOperation(), target,
                                        std::move(localPatterns)))) {
        cloned.erase();
        return;
      }

      replacements.emplace_back(funcOp, cloned);
    });

    // Replace original functions with their cloned versions if symbol
    // replacement succeeds.
    IRRewriter rewriter(&getContext());
    for (auto &p : replacements) {
      mlir::func::FuncOp original = p.first;
      mlir::func::FuncOp cloned = p.second;

      rewriter.setInsertionPointAfter(original);
      rewriter.insert(cloned.getOperation());

      if (failed(SymbolTable::replaceAllSymbolUses(original.getNameAttr(),
                                                   cloned.getNameAttr(),
                                                   module.getOperation()))) {
        cloned.erase();
        original.emitError("failed to replace symbol uses, keeping original");
        continue;
      }

      rewriter.eraseOp(original);
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createMemrefCopyToDMAFlagTreePass() {
  return std::make_unique<MemrefCopyToDMAFlagTreePass>();
}
