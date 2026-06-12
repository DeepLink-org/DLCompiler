//===- TritonDicpDialect.h - MLIR TritonDicp dialect --------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TritonDicp dialect in MLIR, containing DICP operations.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_DICP_DIALECT_H
#define TRITON_DIALECT_DICP_DIALECT_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "dicp/Dialect/TritonDicp/IR/TritonDicpDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "dicp/Dialect/TritonDicp/IR/TritonDicpOpsAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "dicp/Dialect/TritonDicp/IR/TritonDicpOps.h.inc"

namespace mlir::triton::dicp {} // namespace mlir::triton::dicp

#endif // TRITON_DIALECT_DICP_DIALECT_H
