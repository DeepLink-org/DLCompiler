#include "dicp/DynamicCVPipeline/Common/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
namespace mlir {
namespace CVPipeline {

CoreType getOpCoreType(Operation *op) {
  if (!op) {
    return UNDETERMINED;
  }
  if (auto a = op->getAttrOfType<StringAttr>(kCoreType)) {
    return fromStrCoreType(a.getValue());
  }
  return UNDETERMINED;
}

} // namespace CVPipeline
} // namespace mlir
