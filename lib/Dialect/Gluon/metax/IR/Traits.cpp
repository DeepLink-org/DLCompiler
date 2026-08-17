#include "triton/Dialect/Gluon/metax/IR/Traits.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;

LogicalResult mlir::OpTrait::impl::verifySameOperandAndResultMemorySpace(
    Operation *op) {
  if (failed(verifyAtLeastNOperands(op, 1)) ||
      failed(verifyAtLeastNResults(op, 1)))
    return failure();

  auto operandType = dyn_cast<ttg::MemDescType>(op->getOperand(0).getType());
  auto resultType = dyn_cast<ttg::MemDescType>(op->getResult(0).getType());
  if (!operandType || !resultType)
    return op->emitOpError("requires MemDescType operands and results");
  if (operandType.getMemorySpace() == resultType.getMemorySpace())
    return success();
  return op->emitOpError(
      "requires the same memory space for all operands and results");
}
