#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir::triton::gluon::layout_autotune {

enum class CandidateStage : int32_t { Mma = 0, Shared = 1, Blocked = 2 };

inline constexpr StringLiteral kMmaCandidatesAttr =
    "ttg.gluon.mma-layout-candidates";
inline constexpr StringLiteral kSharedCandidatesAttr =
    "ttg.gluon.shared-layout-candidates";
inline constexpr StringLiteral kBlockedCandidatesAttr =
    "ttg.gluon.blocked-layout-candidates";
inline constexpr StringLiteral kLineageAttr =
    "ttg.gluon.layout-candidate-lineage";

StringRef getCandidateAttrName(CandidateStage stage);
ArrayAttr getCandidates(triton::FuncOp function, CandidateStage stage);
void setCandidates(triton::FuncOp function, CandidateStage stage,
                   ArrayRef<ArrayAttr> lanes);
void clearCandidates(triton::FuncOp function, CandidateStage stage);

/// Visit every SSA value type in stable IR order, including block arguments.
void walkValueTypes(triton::FuncOp function,
                    llvm::function_ref<void(Type)> visitor);

DictionaryAttr getEncodingReplacement(MLIRContext *context, Attribute source,
                                      Attribute target);
FailureOr<std::pair<Attribute, Attribute>>
parseEncodingReplacement(DictionaryAttr replacement, Operation *diagnosticOp);

} // namespace mlir::triton::gluon::layout_autotune
