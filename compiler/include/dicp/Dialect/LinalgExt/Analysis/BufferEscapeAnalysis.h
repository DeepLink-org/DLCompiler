#ifndef DICP_DIALECT_LINALGEXT_ANALYSIS_BUFFERESCAPEANALYSIS_H
#define DICP_DIALECT_LINALGEXT_ANALYSIS_BUFFERESCAPEANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::dicp {

/// Summary of how a local buffer's forward slice interacts with operations
/// outside the current analysis scope.
struct BufferEscapeSummary {
  SetVector<Operation *> forwardSlice;
  SmallVector<Operation *, 8> externalUsers;
  SmallVector<Operation *, 4> callEscapes;
  SmallVector<Operation *, 4> modifyingEscapes;

  bool isReadOnlyOutsideScope() const { return modifyingEscapes.empty(); }
  bool hasCallEscape() const { return !callEscapes.empty(); }
};

/// Builds the single-block forward slice of `allocOp` and classifies all
/// out-of-scope users. A user is considered modifying if alias analysis proves
/// that it may write the allocated buffer or one of its aliases.
BufferEscapeSummary
analyzeBufferEscape(memref::AllocOp allocOp,
                    function_ref<bool(Operation *)> isInScope,
                    AliasAnalysis *aliasAnalysis = nullptr);

} // namespace mlir::dicp

#endif // DICP_DIALECT_LINALGEXT_ANALYSIS_BUFFERESCAPEANALYSIS_H
