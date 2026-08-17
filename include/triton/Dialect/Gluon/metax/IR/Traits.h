#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::OpTrait::impl {

LogicalResult verifySameOperandAndResultMemorySpace(Operation *op);

} // namespace mlir::OpTrait::impl

namespace mlir::OpTrait {

template <typename ConcreteType>
class SameOperandAndResultMemorySpace
    : public TraitBase<ConcreteType, SameOperandAndResultMemorySpace> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandAndResultMemorySpace(op);
  }
};

} // namespace mlir::OpTrait
