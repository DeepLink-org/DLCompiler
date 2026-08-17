#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include <memory>
#include <string>
#include <utility>

namespace mlir::triton::gluon {

/// Materialize the Gluon C500 MACA MMA contract, preserving the ordinary C500
/// baseline except for direct aggregate accumulator leaf topology.
std::unique_ptr<Pass>
createGluonAccelerateMatmulPass(int numStages = 2,
                               bool disablePrefetch = false,
                               bool storeCoalesce = false,
                               int computeCapability = 80);

/// Align bounded register-only dot and atomic-rmw consumers with their
/// unique upstream C500 MMA ownership.
std::unique_ptr<Pass> createGluonAlignMmaConsumersPass();

/// Make the existing C500 Accelerate dot-operand and Shared layout
/// requirements explicit without rewriting producer types.
std::unique_ptr<Pass> createGluonInsertRequireLayoutPass();

/// Lower late Gluon-only register views and logical C500 BSM to the standard
/// TritonGPU operations consumed by the existing C500 LLVM lowering.
std::unique_ptr<Pass> createGluonToTritonGPUConversionPass();

std::unique_ptr<Pass>
createGluonMmaLayoutCandidatePass(int computeCapability = 80);
std::unique_ptr<Pass> createGluonSharedLayoutCandidatePass();
std::unique_ptr<Pass> createGluonBlockedLayoutCandidatePass();
std::unique_ptr<Pass> createGluonExpandLayoutCandidatesPass(int stage = 0);

namespace layout_autotune {
FailureOr<SmallVector<std::pair<std::string, std::string>>>
exportLayoutCandidateModules(ModuleOp module);
} // namespace layout_autotune

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h.inc"

} // namespace mlir::triton::gluon
