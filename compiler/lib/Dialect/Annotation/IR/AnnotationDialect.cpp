//===- AnnotationDialect.cpp - MLIR dialect for Annotation implementation -===//
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

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::annotation;

#include "bishengir/Dialect/Annotation/IR/AnnotationOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/Annotation/IR/AnnotationOps.cpp.inc"

void mlir::annotation::AnnotationDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/Annotation/IR/AnnotationOps.cpp.inc"
      >();
}
