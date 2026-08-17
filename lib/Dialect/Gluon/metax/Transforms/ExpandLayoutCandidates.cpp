#include "triton/Dialect/Gluon/metax/Transforms/LayoutCandidates.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gluon-expand-layout-candidates"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[gluon-layout-autotune][expand] " << X << "\n")

namespace ttg = mlir::triton::gpu;
namespace candidate = mlir::triton::gluon::layout_autotune;

namespace mlir::triton::gluon {

#define GEN_PASS_DEF_GLUONEXPANDLAYOUTCANDIDATESPASS
#include "triton/Dialect/Gluon/metax/Transforms/Passes.h.inc"

namespace {

FailureOr<candidate::CandidateStage> parseStage(int32_t value,
                                                Operation *diagnosticOp) {
  switch (value) {
  case 0:
    return candidate::CandidateStage::Mma;
  case 1:
    return candidate::CandidateStage::Shared;
  case 2:
    return candidate::CandidateStage::Blocked;
  default:
    diagnosticOp->emitError() << "invalid layout candidate stage " << value;
    return failure();
  }
}

void appendLineage(FuncOp function, unsigned lane) {
  SmallVector<Attribute> lineage;
  if (auto existing = dyn_cast_or_null<ArrayAttr>(
          function->getDiscardableAttr(candidate::kLineageAttr)))
    llvm::append_range(lineage, existing);
  lineage.push_back(IntegerAttr::get(IntegerType::get(function.getContext(), 32),
                                    lane));
  function->setDiscardableAttr(candidate::kLineageAttr,
                               ArrayAttr::get(function.getContext(), lineage));
}

FailureOr<DenseMap<Attribute, Attribute>>
parseLane(ArrayAttr lane, Operation *diagnosticOp) {
  DenseMap<Attribute, Attribute> replacements;
  for (Attribute encoded : lane) {
    auto dictionary = dyn_cast<DictionaryAttr>(encoded);
    if (!dictionary) {
      diagnosticOp->emitError()
          << "layout candidate lane must contain replacement dictionaries";
      return failure();
    }
    FailureOr<std::pair<Attribute, Attribute>> replacement =
        candidate::parseEncodingReplacement(dictionary, diagnosticOp);
    if (failed(replacement))
      return failure();
    auto [it, inserted] = replacements.try_emplace(replacement->first,
                                                   replacement->second);
    if (!inserted && it->second != replacement->second) {
      diagnosticOp->emitError()
          << "layout candidate assigns two targets to one source encoding";
      return failure();
    }
  }
  return replacements;
}

Attribute remapEncoding(Attribute encoding,
                        const DenseMap<Attribute, Attribute> &replacements) {
  if (!encoding)
    return encoding;
  if (auto it = replacements.find(encoding); it != replacements.end())
    return it->second;

  return TypeSwitch<Attribute, Attribute>(encoding)
      .Case<ttg::DotOperandEncodingAttr>([&](auto dot) -> Attribute {
        Attribute parent = remapEncoding(dot.getParent(), replacements);
        if (parent == dot.getParent())
          return dot;
        return ttg::DotOperandEncodingAttr::get(
            dot.getContext(), dot.getOpIdx(), parent, dot.getKWidth());
      })
      .Case<ttg::SliceEncodingAttr>([&](auto slice) -> Attribute {
        Attribute parent = remapEncoding(slice.getParent(), replacements);
        if (parent == slice.getParent())
          return slice;
        auto distributed = dyn_cast<ttg::DistributedEncodingTrait>(parent);
        return distributed ? ttg::SliceEncodingAttr::get(
                                 slice.getContext(), slice.getDim(), distributed)
                           : Attribute();
      })
      .Default([&](Attribute unchanged) { return unchanged; });
}

FailureOr<Type> remapType(Type type,
                          const DenseMap<Attribute, Attribute> &replacements) {
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    Attribute encoding = remapEncoding(tensor.getEncoding(), replacements);
    if (tensor.getEncoding() && !encoding)
      return failure();
    return tensor.cloneWithEncoding(encoding);
  }
  if (auto memdesc = dyn_cast<ttg::MemDescType>(type)) {
    Attribute encoding = remapEncoding(memdesc.getEncoding(), replacements);
    if (!encoding)
      return failure();
    return ttg::MemDescType::get(
        memdesc.getShape(), memdesc.getElementType(), encoding,
        memdesc.getMemorySpace(), memdesc.getMutableMemory(),
        memdesc.getAllocShape());
  }
  return type;
}

DenseMap<Value, Attribute> collectPhysicalSliceEncodings(FuncOp function) {
  DenseMap<Value, Attribute> encodings;
  function.walk([&](Operation *operation) {
    if (!isa<ttg::ExtractTensorOp, ttg::InsertTensorOp>(operation))
      return;
    for (Value value : llvm::concat<Value>(operation->getOperands(),
                                           operation->getResults())) {
      if (auto tensor = dyn_cast<RankedTensorType>(value.getType()))
        encodings.try_emplace(value, tensor.getEncoding());
    }
  });
  return encodings;
}

bool changesPhysicalSlice(const DenseMap<Value, Attribute> &original) {
  return llvm::any_of(original, [](const auto &entry) {
    auto tensor = dyn_cast<RankedTensorType>(entry.first.getType());
    return !tensor || tensor.getEncoding() != entry.second;
  });
}

LogicalResult remapConstant(arith::ConstantOp constant, Type oldType,
                            Type newType) {
  if (oldType == newType)
    return success();
  auto splat = dyn_cast<SplatElementsAttr>(constant.getValue());
  auto tensor = dyn_cast<RankedTensorType>(newType);
  if (!splat || !tensor)
    return constant.emitError()
           << "layout candidate cannot retag a non-splat tensor constant";
  constant.setValueAttr(
      SplatElementsAttr::get(tensor, splat.getSplatValue<Attribute>()));
  return success();
}

/// Rebuild dialect-owned Shared view results after assigning a root layout.
/// A memdesc view is part of its root owner, not an independently selectable
/// layout. Reusing InferTypeOpInterface keeps transpose semantics in the
/// dialect and avoids duplicating its order/alloc-shape rules here.
LogicalResult normalizeSharedViewTypes(FuncOp function) {
  WalkResult normalization = function.walk<WalkOrder::PreOrder>(
      [&](Operation *operation) -> WalkResult {
        if (operation->getNumRegions() != 0 ||
            llvm::none_of(operation->getResultTypes(),
                          [](Type type) { return isa<ttg::MemDescType>(type); }))
          return WalkResult::advance();

        SmallVector<Type> inferredTypes;
        if (auto typeInference = dyn_cast<InferTypeOpInterface>(operation)) {
          if (failed(typeInference.inferReturnTypes(
                  operation->getContext(), operation->getLoc(),
                  operation->getOperands(), operation->getAttrDictionary(),
                  operation->getPropertiesStorage(), operation->getRegions(),
                  inferredTypes)))
            return WalkResult::interrupt();
        } else if (auto reshape = dyn_cast<ttg::MemDescReshapeOp>(operation)) {
          auto sourceType = cast<ttg::MemDescType>(reshape.getSrc().getType());
          auto resultType = cast<ttg::MemDescType>(reshape.getType());
          ttg::MemDescType inferred;
          if (failed(ttg::MemDescReshapeOp::inferReturnTypes(
                  operation->getContext(), operation->getLoc(), sourceType,
                  resultType.getShape(), inferred)))
            return WalkResult::interrupt();
          inferredTypes.push_back(inferred);
        } else {
          return WalkResult::advance();
        }

        if (inferredTypes.size() != operation->getNumResults())
          return WalkResult::interrupt();
        for (auto [result, inferred] :
             llvm::zip_equal(operation->getResults(), inferredTypes))
          result.setType(inferred);
        return WalkResult::advance();
      });
  return normalization.wasInterrupted() ? failure() : success();
}

/// Close register type relations with existing dialect inference. Candidate
/// formulas choose owner layouts; transpose and same-encoding results are
/// derived uses of those owners. RegionBranch signatures were remapped as one
/// exact assignment and are intentionally excluded from local inference.
LogicalResult normalizeDerivedRegisterEncodings(FuncOp function) {
  WalkResult normalization = function.walk<WalkOrder::PreOrder>(
      [&](Operation *operation) -> WalkResult {
        // Region signatures require predecessor consensus. Exact assignment
        // remapping above updates them together; local forward inference must
        // not attempt to solve control-flow types from a single predecessor.
        if (operation->getNumRegions() != 0 ||
            isa<RegionBranchOpInterface,
                RegionBranchTerminatorOpInterface>(operation))
          return WalkResult::advance();

        if (auto typeInference = dyn_cast<InferTypeOpInterface>(operation);
            typeInference &&
            llvm::any_of(operation->getResultTypes(), [](Type type) {
              return isa<RankedTensorType>(type);
            })) {
          SmallVector<Type> inferredTypes;
          if (failed(typeInference.inferReturnTypes(
                  operation->getContext(), operation->getLoc(),
                  operation->getOperands(), operation->getAttrDictionary(),
                  operation->getPropertiesStorage(), operation->getRegions(),
                  inferredTypes)) ||
              inferredTypes.size() != operation->getNumResults()) {
            LDBG("op=" << operation->getName()
                         << " action=reject reason=type-inference-failure");
            return WalkResult::interrupt();
          }
          for (auto [result, inferred] :
               llvm::zip_equal(operation->getResults(), inferredTypes))
            result.setType(inferred);
          return WalkResult::advance();
        }

        Attribute inferred;
        for (Value operand : operation->getOperands()) {
          auto tensor = dyn_cast<RankedTensorType>(operand.getType());
          if (!tensor || !tensor.getEncoding())
            continue;
          Attribute current = inferDstEncoding(operation, tensor.getEncoding());
          if (!current)
            continue;
          if (inferred && inferred != current) {
            LDBG("op=" << operation->getName()
                         << " action=reject reason=inference-conflict");
            return WalkResult::interrupt();
          }
          inferred = current;
        }
        if (!inferred)
          return WalkResult::advance();

        for (Value result : operation->getResults()) {
          auto tensor = dyn_cast<RankedTensorType>(result.getType());
          if (!tensor)
            continue;
          result.setType(tensor.cloneWithEncoding(inferred));
        }
        return WalkResult::advance();
      });
  return normalization.wasInterrupted() ? failure() : success();
}

LogicalResult applyLane(FuncOp function, ArrayAttr lane) {
  FailureOr<DenseMap<Attribute, Attribute>> replacements =
      parseLane(lane, function);
  if (failed(replacements))
    return failure();
  DenseMap<Value, Attribute> physicalSlices =
      collectPhysicalSliceEncodings(function);

  WalkResult blockRewrite = function.walk([&](Block *block) -> WalkResult {
    for (BlockArgument argument : block->getArguments()) {
      FailureOr<Type> type = remapType(argument.getType(), *replacements);
      if (failed(type))
        return WalkResult::interrupt();
      argument.setType(*type);
    }
    return WalkResult::advance();
  });
  if (blockRewrite.wasInterrupted())
    return failure();
  SmallVector<Type> argumentTypes(
      function.getBlocks().front().getArgumentTypes());
  function.setFunctionType(FunctionType::get(
      function.getContext(), argumentTypes,
      function.getFunctionType().getResults()));

  WalkResult operationRewrite =
      function.walk([&](Operation *operation) -> WalkResult {
        for (Value result : operation->getResults()) {
          Type oldType = result.getType();
          FailureOr<Type> newType = remapType(oldType, *replacements);
          if (failed(newType))
            return WalkResult::interrupt();
          result.setType(*newType);
          if (auto constant = dyn_cast<arith::ConstantOp>(operation))
            if (failed(remapConstant(constant, oldType, *newType)))
              return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  if (operationRewrite.wasInterrupted() ||
      failed(normalizeSharedViewTypes(function)) ||
      failed(normalizeDerivedRegisterEncodings(function)))
    return failure();
  if (changesPhysicalSlice(physicalSlices)) {
    LDBG("function=" << function.getSymName()
                      << " action=reject reason=physical-slice-fence");
    return failure();
  }
  return success();
}

std::string getSiblingName(FuncOp function, candidate::CandidateStage stage,
                           unsigned lane) {
  return (function.getSymName() + ".gluon.layout." +
          Twine(static_cast<int32_t>(stage)) + "." + Twine(lane))
      .str();
}

LogicalResult expandFunction(FuncOp function, SymbolTable &symbols,
                             candidate::CandidateStage stage) {
  ArrayAttr lanes = candidate::getCandidates(function, stage);
  if (!lanes) {
    appendLineage(function, 0);
    return success();
  }
  if (lanes.empty())
    return function.emitError() << "layout candidate set has no baseline lane";

  Block::iterator insertionPoint = std::next(function->getIterator());
  for (unsigned lane = 1; lane < lanes.size(); ++lane) {
    IRMapping mapping;
    FuncOp sibling = cast<FuncOp>(function->clone(mapping));
    sibling.setSymName(getSiblingName(function, stage, lane));
    sibling.setPrivate();
    symbols.insert(sibling, insertionPoint);
    insertionPoint = std::next(sibling->getIterator());

    candidate::clearCandidates(sibling, stage);
    appendLineage(sibling, lane);
    auto assignment = dyn_cast<ArrayAttr>(lanes[lane]);
    LogicalResult status = failure();
    if (assignment) {
      ScopedDiagnosticHandler diagnostics(
          function.getContext(), [&](Diagnostic &diagnostic) {
            LLVM_DEBUG({
              llvm::dbgs() << "[gluon-layout-autotune][expand] function="
                           << sibling.getSymName() << " diagnostic=";
              diagnostic.print(llvm::dbgs());
              llvm::dbgs() << '\n';
            });
            return success();
          });
      status = applyLane(sibling, assignment);
      if (succeeded(status))
        status = verify(sibling);
    }
    if (failed(status)) {
      LDBG("function=" << sibling.getSymName() << " stage="
                        << static_cast<int32_t>(stage) << " lane=" << lane
                        << " action=discard");
      symbols.erase(sibling);
      continue;
    }
  }

  candidate::clearCandidates(function, stage);
  appendLineage(function, 0);
  if (failed(verify(function)))
    return function.emitError() << "baseline layout candidate is invalid";
  return success();
}

class GluonExpandLayoutCandidatesPass final
    : public impl::GluonExpandLayoutCandidatesPassBase<
          GluonExpandLayoutCandidatesPass> {
public:
  GluonExpandLayoutCandidatesPass() = default;
  explicit GluonExpandLayoutCandidatesPass(int candidateStage) {
    stage = candidateStage;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    FailureOr<candidate::CandidateStage> candidateStage =
        parseStage(stage, module);
    if (failed(candidateStage)) {
      signalPassFailure();
      return;
    }

    SymbolTable symbols(module);
    SmallVector<FuncOp> functions(module.getOps<FuncOp>());
    for (FuncOp function : functions) {
      if (function.isExternal())
        continue;
      if (failed(expandFunction(function, symbols, *candidateStage))) {
        signalPassFailure();
        return;
      }
    }
  }
};

std::string printModule(ModuleOp module) {
  std::string source;
  llvm::raw_string_ostream os(source);
  module.print(os);
  os << '\n';
  return os.str();
}

std::string printLineage(FuncOp function) {
  std::string lineage;
  llvm::raw_string_ostream os(lineage);
  auto values = dyn_cast_or_null<ArrayAttr>(
      function->getDiscardableAttr(candidate::kLineageAttr));
  if (values) {
    for (auto [index, value] : llvm::enumerate(values)) {
      if (index)
        os << '.';
      os << cast<IntegerAttr>(value).getInt();
    }
  }
  return os.str();
}

} // namespace

std::unique_ptr<Pass> createGluonExpandLayoutCandidatesPass() {
  return std::make_unique<GluonExpandLayoutCandidatesPass>();
}

std::unique_ptr<Pass> createGluonExpandLayoutCandidatesPass(int stage) {
  return std::make_unique<GluonExpandLayoutCandidatesPass>(stage);
}

namespace layout_autotune {

FailureOr<SmallVector<std::pair<std::string, std::string>>>
exportLayoutCandidateModules(ModuleOp module) {
  SmallVector<FuncOp> functions(module.getOps<FuncOp>());
  auto publicEntry = llvm::find_if(functions, [](FuncOp function) {
    return function.isPublic() && !function.isExternal();
  });
  if (publicEntry == functions.end())
    return module.emitError() << "layout candidate export needs one public entry";
  const std::string entryName = publicEntry->getSymName().str();

  SmallVector<std::pair<std::string, std::string>> variants;
  for (FuncOp function : functions) {
    if (function.isExternal())
      continue;
    const std::string selectedName = function.getSymName().str();
    const std::string lineage = printLineage(function);
    OwningOpRef<ModuleOp> variant = ModuleOp::create(module.getLoc());
    variant->getOperation()->setAttrs(module->getAttrs());
    IRMapping mapping;
    FuncOp selected = cast<FuncOp>(function->clone(mapping));
    variant->getBody()->push_back(selected);
    selected.setSymName(entryName);
    selected.setPublic();
    selected->removeDiscardableAttr(kLineageAttr);
    if (failed(verify(*variant))) {
      LDBG("function=" << selectedName
                        << " action=discard-export-verifier-failure");
      continue;
    }
    variants.emplace_back(printModule(*variant), lineage);
  }

  for (FuncOp function : llvm::make_early_inc_range(module.getOps<FuncOp>())) {
    if (function.isPrivate() && !function.isExternal())
      function.erase();
    else
      function->removeDiscardableAttr(kLineageAttr);
  }
  return variants;
}

} // namespace layout_autotune
} // namespace mlir::triton::gluon

#undef LDBG
#undef DEBUG_TYPE
