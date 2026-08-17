#include "triton/Dialect/Gluon/metax/IR/Types.h"

#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton::gluon;

#include "triton/Dialect/Gluon/metax/IR/GluonTypesEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/Gluon/metax/IR/GluonTypes.cpp.inc"

LogicalResult StorageAliasSpecType::verify(
    function_ref<InFlightDiagnostic()> emitError, StorageKind storage,
    std::optional<int64_t> bufferSizeBytes) {
  if (storage != StorageKind::smem)
    return emitError() << "only smem storage aliases are supported";
  if (bufferSizeBytes && *bufferSizeBytes <= 0)
    return emitError() << "buffer_size_bytes must be positive, got "
                       << *bufferSizeBytes;
  return success();
}

LogicalResult ReuseGroupType::verify(
    function_ref<InFlightDiagnostic()>, ReuseGroupKind) {
  return success();
}

void GluonDialect::registerMetaXTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton/Dialect/Gluon/metax/IR/GluonTypes.cpp.inc"
      >();
}
