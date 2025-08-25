#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::dicp::linked {

const unsigned INT_BIT_WIDTH = 32;
const unsigned SET_INIT_SIZE = 16;
enum TensorKind { NONE = -1, INPUT = 0, OUTPUT = 1, INPUT_OUTPUT = 2 };


static void setBlockArgumentAttr(BlockArgument blockArg, triton::FuncOp func, TensorKind tensorKind)
{
    unsigned argIdx = blockArg.getArgNumber();
    auto existingAttr = func.getArgAttrOfType<IntegerAttr>(argIdx, "tt.tensor_kind");
    TensorKind oldVal = existingAttr ? static_cast<TensorKind>(existingAttr.getInt()) : TensorKind::NONE;

    TensorKind finalVal = tensorKind;
    if ((oldVal == TensorKind::INPUT && tensorKind == TensorKind::OUTPUT) ||
        (oldVal == TensorKind::OUTPUT && tensorKind == TensorKind::INPUT)) {
        finalVal = TensorKind::INPUT_OUTPUT;
    } else if (oldVal == TensorKind::INPUT_OUTPUT) {
        finalVal = oldVal;
    }

    func.setArgAttr(argIdx, "tt.tensor_kind",
                    IntegerAttr::get(IntegerType::get(func.getContext(), INT_BIT_WIDTH), static_cast<int>(finalVal)));
}

template <typename OpTy>
void addTensorKindToArguments(OpTy op, triton::FuncOp func, TensorKind tensorKind)
{
    Value ptr = op.getPtr();
    if (!ptr)
        return;

    Value cur = ptr;
    llvm::SmallPtrSet<Value, SET_INIT_SIZE> visited;
    // 回溯 def-use 链，找到起源 BlockArgument
    while (visited.insert(cur).second) {
        // 如果是 BlockArgument，则尝试设置属性
        if (auto blockArg = dyn_cast<BlockArgument>(cur)) {
            if (blockArg.getOwner() == &func.getBody().front()) {
                auto type = blockArg.getType();
                // 检查是否是 triton::PointerType
                if (!isa<triton::PointerType>(type))
                    break;
                setBlockArgumentAttr(blockArg, func, tensorKind);
                break;
            }
        }

        Operation *defOp = cur.getDefiningOp();
        if (!defOp)
            break;
        cur = defOp->getOperand(0);
    }
}

std::unique_ptr<OperationPass<ModuleOp>> createLinalgToLinkedPass();

} // namespace mlir::dicp::linked
