//===-- TritonDicpDialect.td - dialect op definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DICP_DIALECT
#define TRITON_DICP_DIALECT

include "mlir/IR/OpBase.td"

def TritonDicp_Dialect : Dialect {
  let name = "dicp";
  let cppNamespace = "::mlir::triton::dicp";
  let summary = "The TritonDicp dialect in Triton.";

  let description = [{
    TritonDicp is a dialect for representing operations on DICP NPUs.
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "triton::TritonDialect",
  ];

  let extraClassDeclaration = [{}];
}

#endif // TRITON_DICP_DIALECT
