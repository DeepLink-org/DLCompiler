

#ifndef ADD_AUTO_SCHEDULING_COMMON_UTILS_H
#define ADD_AUTO_SCHEDULING_COMMON_UTILS_H
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include <string_view>

namespace mlir {
namespace CVPipeline {

inline constexpr llvm::StringLiteral kCoreType = "ssbuffer.core_type";

enum CoreType {
  UNDETERMINED = 0,
  VECTOR_ONLY = 1 << 0,
  CUBE_ONLY = 1 << 1,
  CUBE_AND_VECTOR = VECTOR_ONLY | CUBE_ONLY,
};

inline constexpr CoreType fromStrCoreType(std::string_view s) {
  if (s == "VECTOR") {
    return CoreType::VECTOR_ONLY;
  }
  if (s == "CUBE") {
    return CoreType::CUBE_ONLY;
  }

  return CoreType::UNDETERMINED;
}

// Functions for managing core types
CoreType getOpCoreType(Operation *op);
} // namespace CVPipeline
} // namespace mlir

#endif
