//===-- TritonDicpAttrDefs.td - dialect attributes def.  ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DICP_ATTRDEFS
#define TRITON_DICP_ATTRDEFS

include "dicp/Dialect/TritonDicp/IR/TritonDicpDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class TritonDicp_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TritonDicp_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
  let cppNamespace = "::mlir::triton::dicp";
}



#endif // TRITON_DICP_ATTRDEFS
