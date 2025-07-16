#pragma once

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h" 
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::func;

namespace mlir {
namespace dicp {
namespace linked {

struct VerifyNoLinalgGenericPass : public PassWrapper<VerifyNoLinalgGenericPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "verify-no-linalg-generic"; }
  StringRef getDescription() const final {
    return "Verify that no 'linalg.generic' operations exist";
  }

  void runOnOperation() override {
    bool foundGeneric = false;
    getOperation()->walk([&](linalg::GenericOp op) {
      op.emitError() << "linalg.generic is not allowed in this pass pipeline.";
      foundGeneric = true;
    });

    if (foundGeneric) {
      signalPassFailure();
    }
  }
};

inline std::unique_ptr<mlir::Pass> createVerifyNoLinalgGenericPass() {
  return std::make_unique<VerifyNoLinalgGenericPass>();
}

} // namespace linked
} // namespace dicp
} // namespace mlir
