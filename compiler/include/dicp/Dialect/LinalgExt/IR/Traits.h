#ifndef DIALECT_LINALGEXT_IR_TAITS_H_
#define DIALECT_LINALGEXT_IR_TAITS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace OpTrait {

template <class ConcreteType>
class LinalgExt_ElementwiseTrait
    : public TraitBase<ConcreteType, LinalgExt_ElementwiseTrait> {};

template <class ConcreteType>
class LinalgExt_ReduceTrait
    : public TraitBase<ConcreteType, LinalgExt_ReduceTrait> {};

template <class ConcreteType>
class LinalgExt_TensorCoreTrait
    : public TraitBase<ConcreteType, LinalgExt_TensorCoreTrait> {};

} // namespace OpTrait
} // namespace mlir

#endif // DIALECT_LINALGEXT_IR_TAITS_H_