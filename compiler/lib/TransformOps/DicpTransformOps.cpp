#include "dicp/TransformOps/DicpTransformOps.h"
#include "dicp/TransformOps/Transforms.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/Syntax.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dicp-transform-op"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::transform;
using namespace mlir::dicp;

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure ReverseOp::apply(TransformRewriter &rewriter,
                                             TransformResults &transformResults,
                                             TransformState &state) {
  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));
  SmallVector<Operation *> reversedOperations = {targets.rbegin(),
                                                 targets.rend()};
  transformResults.set(cast<OpResult>(getResult()), reversedOperations);
  return DiagnosedSilenceableFailure::success();
}

void ReverseOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

// ============================================================================
// ForwardInitToIterArgOp
// ============================================================================

/// Checks if two operations operating on subsets (extraction vs insertion)
/// are geometrically equivalent (same offsets, sizes, and strides).
///
/// This uses the `OffsetSizeAndStrideOpInterface` to be dialect-agnostic.
static bool areSlicesEquivalent(Operation *readOp, Operation *writeOp) {
  auto readInterface = dyn_cast<OffsetSizeAndStrideOpInterface>(readOp);
  auto writeInterface = dyn_cast<OffsetSizeAndStrideOpInterface>(writeOp);

  if (!readInterface || !writeInterface) {
    LDBG("  One of the ops does not implement OffsetSizeAndStrideOpInterface.");
    return false;
  }

  // Compare mixed offsets, sizes, and strides.
  // Note: This relies on SSA value equality for dynamic dims.
  bool offsetsMatch = llvm::equal(readInterface.getMixedOffsets(),
                                  writeInterface.getMixedOffsets());
  bool sizesMatch = llvm::equal(readInterface.getMixedSizes(),
                                writeInterface.getMixedSizes());
  bool stridesMatch = llvm::equal(readInterface.getMixedStrides(),
                                  writeInterface.getMixedStrides());

  if (!offsetsMatch || !sizesMatch || !stridesMatch) {
    LLVM_DEBUG({
      if (!offsetsMatch)
        DBGS() << "  Offsets mismatch.\n";
      if (!sizesMatch)
        DBGS() << "  Sizes mismatch.\n";
      if (!stridesMatch)
        DBGS() << "  Strides mismatch.\n";
    });
    return false;
  }

  return true;
}

/// Strategy for scf.forall:
/// The write-back occurs in the `scf.in_parallel` terminator via a
/// SubsetInsertionOp (usually tensor.parallel_insert_slice).
static Operation *findWriteBackOp(scf::ForallOp loopOp,
                                  BlockArgument regionArg) {
  scf::InParallelOp terminator = loopOp.getTerminator();
  Block &terminatorBlock = terminator.getRegion().front();

  for (Operation &op : terminatorBlock) {
    // Check if it's a subset insertion (like parallel_insert_slice)
    if (auto insertOp = dyn_cast<SubsetInsertionOpInterface>(op)) {
      // For parallel insert, the destination is the BlockArgument of the loop
      // that corresponds to the output.
      if (insertOp.getDestinationOperand().get() == regionArg) {
        return &op;
      }
    }
  }
  return nullptr;
}

/// Strategy for scf.for:
/// The write-back is the value yielded by `scf.yield`. We need to check if
/// that yielded value is defined by a SubsetInsertionOp that inserts *into*
/// the corresponding region argument.
static Operation *findWriteBackOp(scf::ForOp loopOp, BlockArgument regionArg) {
  auto yieldOp = cast<scf::YieldOp>(loopOp.getBody()->getTerminator());

  // 1. Identify which result index this regionArg corresponds to.
  // scf.for region args: [iv, iter_arg_0, iter_arg_1, ...]
  // The iter_args start at index 1 (since index 0 is IV).
  unsigned iterArgIndex = regionArg.getArgNumber() - 1; // Subtract IV

  if (iterArgIndex >= yieldOp.getResults().size()) {
    return nullptr; // Should not happen if IR is valid
  }

  Value yieldedVal = yieldOp.getOperand(iterArgIndex);
  Operation *defOp = yieldedVal.getDefiningOp();

  if (!defOp)
    return nullptr;

  // 2. Check if the yielded value comes from an insertion op
  if (auto insertOp = dyn_cast<SubsetInsertionOpInterface>(defOp)) {
    // 3. Check if the insertion destination IS the region argument.
    // i.e., %new = insert_slice %update into %iter_arg
    if (insertOp.getDestinationOperand().get() == regionArg) {
      return defOp;
    }
  }

  return nullptr;
}

/// Generic processor that works for both scf.for and scf.forall.
/// It relies on the `findWriteBackOp` overload to handle structural
/// differences.
template <typename LoopTy>
static void processLoop(LoopTy loopOp, RewriterBase &rewriter) {
  LDBG("Processing loop: " << loopOp.getOperation()->getName());

  auto regionIterArgs = loopOp.getRegionIterArgs();

  // scf.forall uses getOutputs(), scf.for uses getInitArgs().
  // We use a lambda to abstract this access.
  auto getInitOperands = [&](auto op) -> OperandRange {
    if constexpr (std::is_same_v<decltype(op), scf::ForallOp>)
      return op.getOutputs();
    else
      return op.getInitArgs();
  };

  auto initOperands = getInitOperands(loopOp);

  // Iterate over each (InitOperand, RegionIterArg) pair
  for (auto it : llvm::zip(initOperands, regionIterArgs)) {
    Value initVal = std::get<0>(it);
    BlockArgument regionArg = std::get<1>(it);

    LDBG("  Analyzing pair: InitVal=" << initVal
                                      << ", RegionArg=" << regionArg);

    // 1. Find the write-back operation (Insertion)
    Operation *writeOp = findWriteBackOp(loopOp, regionArg);
    if (!writeOp) {
      LDBG("    No valid write-back (insertion) found for this argument. "
           "Skipping.");
      continue;
    }
    LDBG("    Found write-back op: " << *writeOp);

    // 2. Find read operations (Extraction) inside the loop body
    // We look for extractions that read from the *external* 'initVal'.
    SmallVector<Operation *> candidates;
    for (Operation *user : initVal.getUsers()) {
      // Ensure the user is strictly inside the loop body
      if (loopOp.getBody()->findAncestorOpInBlock(*user)) {
        if (isa<SubsetExtractionOpInterface>(user)) {
          candidates.push_back(user);
        }
      }
    }

    if (candidates.empty()) {
      LDBG("    No extraction users of InitVal found inside loop.");
      continue;
    }

    // 3. Compare and Replace
    for (Operation *readOp : candidates) {
      LDBG("    Checking candidate read op: " << *readOp);

      if (areSlicesEquivalent(readOp, writeOp)) {
        LDBG("      MATCH! Slices are equivalent. Forwarding init arg to iter "
             "arg.");

        // Transform: Replace the source of the extraction (which is currently
        // the external init_arg) with the internal region_arg (iter_arg).
        // This enables in-place bufferization.

        rewriter.setListener(
            nullptr); // Disable listener for simple operand updates
        rewriter.modifyOpInPlace(readOp, [&]() {
          // Use the interface to set the source operand generically
          // Note: getSourceOperand() returns an OpOperand&.
          auto subsetOp = cast<SubsetExtractionOpInterface>(readOp);
          subsetOp.getSourceOperand().set(regionArg);
        });
      } else {
        LDBG("      Mismatch: Geometry differs.");
      }
    }
  }
}

DiagnosedSilenceableFailure
ForwardInitToIterArgOp::apply(TransformRewriter &rewriter,
                              TransformResults &transformResults,
                              TransformState &state) {
  SmallVector<Operation *> processedOps;
  auto payloadOps = state.getPayloadOps(getTarget());

  for (Operation *op : payloadOps) {
    bool isProcessed = false;
    if (!op) {
      LDBG("Skipping op: " << op->getName() << " is null");
      continue;
    }

    // Dispatch to the appropriate template instantiation
    if (auto forallOp = dyn_cast<scf::ForallOp>(op)) {
      processLoop(forallOp, rewriter);
      isProcessed = true;
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      processLoop(forOp, rewriter);
      isProcessed = true;
    }

    if (!isProcessed) {
      LDBG("Skipping op: " << op->getName() << " (not scf.for or scf.forall)");
    }

    // We preserve the operations in the result handle
    processedOps.push_back(op);
  }

  transformResults.set(cast<OpResult>(getResult()), processedOps);
  return DiagnosedSilenceableFailure::success();
}

void ForwardInitToIterArgOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// ExtendedFuseIntoContainingOp
//===----------------------------------------------------------------------===//

void transform::ExtendedFuseIntoContainingOp::build(OpBuilder &builder,
                                                    OperationState &result,
                                                    Value producerOp,
                                                    Value containingOp) {
  result.addOperands({producerOp, containingOp});
  auto resultType = transform::AnyOpType::get(builder.getContext());
  result.addTypes({resultType, resultType});
}

bool transform::ExtendedFuseIntoContainingOp::allowsRepeatedHandleOperands() {
  // Allow repeated handles since we are fusing everything anyway.
  return true;
}

DiagnosedSilenceableFailure
transform::ExtendedFuseIntoContainingOp::fuseIntoOneContaining(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state,
    size_t index, Operation *containingOp) {
  assert(index < getFusedOp().size());
  assert(index < getNewContainingOp().size());

  SmallVector<Operation *> fusedOps;
  auto producerOps = state.getPayloadOps(getProducerOp());

  LLVM_DEBUG({
    DBGS() << "=== ExtendedFuseIntoContainingOp: producerOps ===\n";
    for (Operation *op : producerOps) {
      DBGS() << "producerOp @" << op << ":\n";
      op->print(DBGS());
      DBGS() << "\n----------------------------------------\n";
    }
    DBGS() << "containingOp @" << containingOp << " :\n ";
    containingOp->print(DBGS());
    DBGS() << "=== end producerOps ===\n";
  });

  // If nothing to fuse, propagate success.
  if (std::empty(producerOps)) {
    results.set(cast<OpResult>(getFusedOp()[index]),
                SmallVector<mlir::Operation *>{});
    results.set(cast<OpResult>(getNewContainingOp()[index]), {containingOp});
    return DiagnosedSilenceableFailure::success();
  }

  SetVector<Operation *> remainingProducers(producerOps.begin(),
                                            producerOps.end());
  auto getNextProducer = [&]() -> FailureOr<std::pair<Operation *, size_t>> {
    for (const auto &it : enumerate(remainingProducers)) {
      Operation *producerOp = it.value();
      // The containing op may be a user of producerOp: use isAncestor.
      int64_t numUsesInContainingOp =
          llvm::count_if(producerOp->getUsers(), [&](Operation *op) {
            return containingOp->isAncestor(op);
          });
      LLVM_DEBUG(DBGS() << "producerOp: " << *producerOp << "\n");
      LLVM_DEBUG(DBGS() << "numUsesInContainingOp: " << numUsesInContainingOp
                        << "\n");
      if (numUsesInContainingOp > 0) {
        return std::make_pair(producerOp, it.index());
      }
    }
    return failure();
  };

  // Helper function to erase producerOp from eraseRemainingProducer if no
  // users.
  auto eraseRemainingProducer = [&](Operation *producerOp, size_t pos) {
    int64_t numUsesInContainingOp =
        llvm::count_if(producerOp->getUsers(), [&](Operation *op) {
          return containingOp->isAncestor(op);
        });
    if (numUsesInContainingOp == 0) {
      remainingProducers.erase(remainingProducers.begin() + pos);
    }
  };

  while (!remainingProducers.empty()) {
    auto nextProducer = getNextProducer();
    if (failed(nextProducer)) {
      auto diag = mlir::emitSilenceableFailure(getLoc())
                  << "could not find next producer to fuse into container";
      diag.attachNote(containingOp->getLoc()) << "containing op";
      return diag;
    }

    Operation *producerOp;
    size_t producerIndex;
    std::tie(producerOp, producerIndex) = *nextProducer;

    // Default diagnostic, to be complemented with more failure information.
    Diagnostic diag(producerOp->getLoc(), DiagnosticSeverity::Remark);
    diag << "could not fuse " << *producerOp << " into " << *containingOp;

    // 1. Try to tile and fuse all subset users (extract slice, etc.)
    // Note: unionProducerUsers is removed because tileAndFuseAllSubsetOps
    // handles multiple users individually, removing the need to pre-union them.
    auto [tiledOps, newContainingOp] = mlir::dicp::tileAndFuseAllSubsetOps(
        rewriter, diag, producerOp, containingOp, getDuplicateProducer());

    if (!tiledOps.empty()) {
      LLVM_DEBUG(DBGS() << "\nFused direct subset ops\n"
                        << *containingOp << "\n");
      fusedOps.append(tiledOps);
      if (newContainingOp) {
        // Update handles associated with the containing op so we don't need
        // to invalidate them. This supports better composability between
        // tiling and fusion.
        LLVM_DEBUG({
          llvm::dbgs() << "[extended_fuse] replacing containing op\n";
          llvm::dbgs() << "  old: ";
          containingOp->print(llvm::dbgs());
          llvm::dbgs() << "\n  new: ";
          newContainingOp->print(llvm::dbgs());
          llvm::dbgs() << "\n";
        });

        LogicalResult replacementStatus =
            rewriter.notifyPayloadOperationReplaced(containingOp,
                                                    newContainingOp);
        (void)replacementStatus;
        assert(succeeded(replacementStatus) &&
               "unable to update transform state mapping");
        containingOp = newContainingOp;
      }
      eraseRemainingProducer(producerOp, producerIndex);
      continue;
    }

    // 2. Try to tile and fuse subset users of the block argument
    // (e.g., when the producer is passed as an init operand to scf.forall)
    SmallVector<Operation *> tiledContainingOpOperand;
    if (auto loopLike = dyn_cast<LoopLikeOpInterface>(containingOp)) {
      tiledContainingOpOperand =
          mlir::dicp::tileAndFuseAllSubsetOpsThroughContainingOpBlockArgument(
              rewriter, diag, producerOp, loopLike);
    }
    if (!tiledContainingOpOperand.empty()) {
      LLVM_DEBUG(DBGS() << "\nFused subset ops through block argument\n"
                        << *containingOp);
      fusedOps.append(tiledContainingOpOperand);
      eraseRemainingProducer(producerOp, producerIndex);
      continue;
    }

    // 3. Try to clone and fuse users (element-wise fusion by cloning)
    Operation *cloned = mlir::dicp::cloneAndFuseAllSubsetOps(
        rewriter, diag, producerOp, containingOp);
    if (cloned) {
      LLVM_DEBUG(DBGS() << "\nFused uses by cloning\n" << *containingOp);
      // We append the single representative fused op returned by cloneAndFuse.
      // Ideally, we might want to track all cloned ops, but the interface
      // returns the last one currently.
      fusedOps.push_back(cloned);
      eraseRemainingProducer(producerOp, producerIndex);
      continue;
    }

    return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
  }

  results.set(cast<OpResult>(getFusedOp()[index]), fusedOps);
  results.set(cast<OpResult>(getNewContainingOp()[index]), {containingOp});
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform::ExtendedFuseIntoContainingOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  auto containingOps = getContainingOp();
  LLVM_DEBUG({
    for (auto containingOpHandle : containingOps) {
      auto payloads = state.getPayloadOps(containingOpHandle);
      DBGS() << "Containing op handle has "
             << std::distance(payloads.begin(), payloads.end())
             << " payload operations\n";
    }
  });

  for (auto it : llvm::enumerate(containingOps)) {
    auto containingOpPayloads = state.getPayloadOps(it.value());
    if (!llvm::hasSingleElement(containingOpPayloads)) {
      return emitDefiniteFailure()
             << "requires exactly one containing_op handle (got "
             << llvm::range_size(containingOpPayloads) << ")";
    }
    Operation *currentOp = *containingOpPayloads.begin();
    auto status =
        fuseIntoOneContaining(rewriter, results, state, it.index(), currentOp);
    if (!status.succeeded())
      return status;
  }
  return DiagnosedSilenceableFailure::success();
}

ParseResult ExtendedFuseIntoContainingOp::parse(OpAsmParser &parser,
                                                OperationState &result) {
  OpAsmParser::UnresolvedOperand producer;
  SmallVector<OpAsmParser::UnresolvedOperand> containingOps;
  FunctionType functionalType;
  llvm::SMLoc producerLoc;
  llvm::SMLoc containingOpsLoc;

  if (parser.getCurrentLocation(&producerLoc) || parser.parseOperand(producer))
    return ParseResult::failure();

  if (parser.parseKeyword("into"))
    return ParseResult::failure();

  if (parser.getCurrentLocation(&containingOpsLoc) ||
      parser.parseOperandList(containingOps))
    return ParseResult::failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return ParseResult::failure();

  if (result.propertiesAttr) {
    NamedAttrList attrs = llvm::cast<DictionaryAttr>(result.propertiesAttr);
    attrs.append("resultSegmentSizes",
                 parser.getBuilder().getDenseI32ArrayAttr(
                     {static_cast<int32_t>(containingOps.size()),
                      static_cast<int32_t>(containingOps.size())}));
    result.propertiesAttr = attrs.getDictionary(parser.getContext());
  } else {
    result.addAttribute("resultSegmentSizes",
                        parser.getBuilder().getDenseI32ArrayAttr(
                            {static_cast<int32_t>(containingOps.size()),
                             static_cast<int32_t>(containingOps.size())}));
  }

  if (parser.parseColonType(functionalType))
    return ParseResult::failure();

  if (parser.resolveOperand(producer, functionalType.getInputs().front(),
                            result.operands) ||
      parser.resolveOperands(containingOps,
                             functionalType.getInputs().drop_front(),
                             containingOpsLoc, result.operands)) {
    return ParseResult::failure();
  }

  result.addTypes(functionalType.getResults());
  return ParseResult::success();
}

void ExtendedFuseIntoContainingOp::print(OpAsmPrinter &p) {
  p << ' ' << getProducerOp();
  p << ' ' << "into";
  p << ' ';
  p.printOperands(getContainingOp());
  p.printOptionalAttrDict((*this)->getAttrs(), {"resultSegmentSizes"});
  p << " : ";
  p.printFunctionalType(getOperands().getTypes(), getResults().getTypes());
}

void transform::ExtendedFuseIntoContainingOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getProducerOpMutable(), effects);
  onlyReadsHandle(getContainingOpMutable(), effects);
  producesHandle(getResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// LoopFuseSiblingOp
//===----------------------------------------------------------------------===//

/// Check if `target` and `source` are siblings, in the context that `target`
/// is being fused into `source`.
///
/// This is a simple check that just checks if both operations are in the same
/// block and some checks to ensure that the fused IR does not violate
/// dominance.
static DiagnosedSilenceableFailure isOpSibling(Operation *target,
                                               Operation *source) {
  // Check if both operations are same.
  if (target == source)
    return emitSilenceableFailure(source)
           << "target and source need to be different loops";

  // Check if both operations are in the same block.
  if (target->getBlock() != source->getBlock())
    return emitSilenceableFailure(source)
           << "target and source are not in the same block";

  // Check if fusion will violate dominance.
  DominanceInfo domInfo(source);
  if (target->isBeforeInBlock(source)) {
    // Since `target` is before `source`, all users of results of `target`
    // need to be dominated by `source`.
    for (Operation *user : target->getUsers()) {
      if (!domInfo.properlyDominates(source, user, /*enclosingOpOk=*/false)) {
        return emitSilenceableFailure(target)
               << "user of results of target should be properly dominated by "
                  "source";
      }
    }
  } else {
    // Since `target` is after `source`, all values used by `target` need
    // to dominate `source`.

    // Check if operands of `target` are dominated by `source`.
    for (Value operand : target->getOperands()) {
      Operation *operandOp = operand.getDefiningOp();
      // Operands without defining operations are block arguments. When `target`
      // and `source` occur in the same block, these operands dominate `source`.
      if (!operandOp)
        continue;

      // Operand's defining operation should properly dominate `source`.
      if (!domInfo.properlyDominates(operandOp, source,
                                     /*enclosingOpOk=*/false))
        return emitSilenceableFailure(target)
               << "operands of target should be properly dominated by source";
    }

    // Check if values used by `target` are dominated by `source`.
    bool failed = false;
    OpOperand *failedValue = nullptr;
    visitUsedValuesDefinedAbove(target->getRegions(), [&](OpOperand *operand) {
      Operation *operandOp = operand->get().getDefiningOp();
      if (operandOp && !domInfo.properlyDominates(operandOp, source,
                                                  /*enclosingOpOk=*/false)) {
        // `operand` is not an argument of an enclosing block and the defining
        // op of `operand` is outside `target` but does not dominate `source`.
        failed = true;
        failedValue = operand;
      }
    });

    if (failed)
      return emitSilenceableFailure(failedValue->getOwner())
             << "values used inside regions of target should be properly "
                "dominated by source";
  }

  return DiagnosedSilenceableFailure::success();
}

/// Check if `target` scf.forall can be fused into `source` scf.forall.
///
/// This simply checks if both loops have the same bounds, steps and mapping.
/// No attempt is made at checking that the side effects of `target` and
/// `source` are independent of each other.
static bool isForallWithIdenticalConfiguration(Operation *target,
                                               Operation *source) {
  auto targetOp = dyn_cast<scf::ForallOp>(target);
  auto sourceOp = dyn_cast<scf::ForallOp>(source);
  if (!targetOp || !sourceOp)
    return false;

  return targetOp.getMixedLowerBound() == sourceOp.getMixedLowerBound() &&
         targetOp.getMixedUpperBound() == sourceOp.getMixedUpperBound() &&
         targetOp.getMixedStep() == sourceOp.getMixedStep() &&
         targetOp.getMapping() == sourceOp.getMapping();
}

/// Check if `target` scf.for can be fused into `source` scf.for.
///
/// This simply checks if both loops have the same bounds and steps. No attempt
/// is made at checking that the side effects of `target` and `source` are
/// independent of each other.
static bool isForWithIdenticalConfiguration(Operation *target,
                                            Operation *source) {
  auto targetOp = dyn_cast<scf::ForOp>(target);
  auto sourceOp = dyn_cast<scf::ForOp>(source);
  if (!targetOp || !sourceOp)
    return false;

  return targetOp.getLowerBound() == sourceOp.getLowerBound() &&
         targetOp.getUpperBound() == sourceOp.getUpperBound() &&
         targetOp.getStep() == sourceOp.getStep();
}

DiagnosedSilenceableFailure
ExtendedLoopFuseSiblingOp::apply(transform::TransformRewriter &rewriter,
                                 transform::TransformResults &results,
                                 transform::TransformState &state) {
  auto targetOps = state.getPayloadOps(getTarget());
  auto sourceOps = state.getPayloadOps(getSource());

  if (!llvm::hasSingleElement(targetOps) ||
      !llvm::hasSingleElement(sourceOps)) {
    return emitDefiniteFailure()
           << "requires exactly one target handle (got "
           << llvm::range_size(targetOps) << ") and exactly one "
           << "source handle (got " << llvm::range_size(sourceOps) << ")";
  }

  Operation *target = *targetOps.begin();
  Operation *source = *sourceOps.begin();

  // Check if the target and source are siblings.
  DiagnosedSilenceableFailure diag = isOpSibling(target, source);
  if (!diag.succeeded())
    return diag;

  Operation *fusedLoop;
  /// TODO: Support fusion for loop-like ops besides scf.for and scf.forall.
  if (isForWithIdenticalConfiguration(target, source)) {
    fusedLoop = fuseIndependentSiblingForLoops(
        cast<scf::ForOp>(target), cast<scf::ForOp>(source), rewriter);
  } else if (isForallWithIdenticalConfiguration(target, source)) {
    fusedLoop = fuseIndependentSiblingForallLoops(
        cast<scf::ForallOp>(target), cast<scf::ForallOp>(source), rewriter);
  } else {
    return emitSilenceableFailure(target->getLoc())
           << "operations cannot be fused";
  }

  assert(fusedLoop && "failed to fuse operations");

  results.set(cast<OpResult>(getFusedLoop()), {fusedLoop});
  return DiagnosedSilenceableFailure::success();
}

#define GET_OP_CLASSES
#include "dicp/TransformOps/DicpTransformOps.cpp.inc"
