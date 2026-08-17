#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "triton/Dialect/Gluon/metax/IR/GluonTypesEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/Gluon/metax/IR/GluonTypes.h.inc"
