//===- AnnotationBase.td - Base definitions for annotation dialect ---------==//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATIONBASE_TD
#define BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATIONBASE_TD

include "mlir/IR/OpBase.td"

def Annotation_Dialect : Dialect {
  let name = "annotation";
  let description = [{
    Annotation dialects for mark operations to
    define some extra attrs for a certain op:

    ```mlir
    // Mark extra attrs
    annotation.mark %a { attr-dict } : f64
    ```
  }];
  let cppNamespace = "::mlir::annotation";
}

#endif // BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATIONBASE_TD
