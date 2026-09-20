#ifndef NO_BUFFERIZE_FLAGTREE_CONVERSION_PASSES
#define NO_BUFFERIZE_FLAGTREE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def NoBufferizeFlagTree : Pass<"no-bufferize-flagtree", "mlir::ModuleOp"> {
  let summary = "Add no_bufferize to memref ops in shared memory space";
}

#endif
