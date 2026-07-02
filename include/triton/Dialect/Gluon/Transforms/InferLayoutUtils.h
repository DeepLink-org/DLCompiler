#ifndef TRITON_DIALECT_GLUON_TRANSFORMS_INFERLAYOUTUTILS_H_
#define TRITON_DIALECT_GLUON_TRANSFORMS_INFERLAYOUTUTILS_H_

#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PriorityWorklist.h"
#include <functional>

namespace mlir::triton::gluon {

struct LayoutInferenceHooks {
  std::function<Attribute(ExtractSliceOp, Attribute)> inferExtractResult;
  std::function<Attribute(ExtractSliceOp, Attribute)> inferExtractSource;
  std::function<Attribute(InsertSliceOp, Attribute)> inferInsertSub;
};

LogicalResult
inferLayout(FuncOp func, llvm::function_ref<bool(Type)> typeCheck,
            const SmallVector<std::pair<Value, Attribute>> &seedEncodings);

LogicalResult
inferLayout(FuncOp func, llvm::function_ref<bool(Type)> typeCheck,
            const SmallVector<std::pair<Value, Attribute>> &seedEncodings,
            const LayoutInferenceHooks &hooks);

LogicalResult doubleCheckEncodings(ModuleOp &mod,
                                   llvm::function_ref<bool(Type)> typeCheck);

} // namespace mlir::triton::gluon

#endif // TRITON_DIALECT_GLUON_TRANSFORMS_INFERLAYOUTUTILS_H_
