#include "triton/Dialect/Gluon/metax/Transforms/LayoutCandidates.h"

#include "mlir/IR/Builders.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::triton::gluon::layout_autotune {

StringRef getCandidateAttrName(CandidateStage stage) {
  switch (stage) {
  case CandidateStage::Mma:
    return kMmaCandidatesAttr;
  case CandidateStage::Shared:
    return kSharedCandidatesAttr;
  case CandidateStage::Blocked:
    return kBlockedCandidatesAttr;
  }
  llvm_unreachable("unknown Gluon layout candidate stage");
}

ArrayAttr getCandidates(triton::FuncOp function, CandidateStage stage) {
  return dyn_cast_or_null<ArrayAttr>(
      function->getDiscardableAttr(getCandidateAttrName(stage)));
}

void setCandidates(triton::FuncOp function, CandidateStage stage,
                   ArrayRef<ArrayAttr> lanes) {
  SmallVector<Attribute> encoded(lanes.begin(), lanes.end());
  function->setDiscardableAttr(getCandidateAttrName(stage),
                               ArrayAttr::get(function.getContext(), encoded));
}

void clearCandidates(triton::FuncOp function, CandidateStage stage) {
  function->removeDiscardableAttr(getCandidateAttrName(stage));
}

void walkValueTypes(triton::FuncOp function,
                    llvm::function_ref<void(Type)> visitor) {
  function.walk([&](Block *block) {
    for (BlockArgument argument : block->getArguments())
      visitor(argument.getType());
  });
  function.walk([&](Operation *operation) {
    for (Type type : operation->getResultTypes())
      visitor(type);
  });
}

DictionaryAttr getEncodingReplacement(MLIRContext *context, Attribute source,
                                      Attribute target) {
  NamedAttrList attributes;
  attributes.set("source", source);
  attributes.set("target", target);
  return DictionaryAttr::get(context, attributes);
}

FailureOr<std::pair<Attribute, Attribute>>
parseEncodingReplacement(DictionaryAttr replacement, Operation *diagnosticOp) {
  Attribute source = replacement.get("source");
  Attribute target = replacement.get("target");
  if (!source || !target) {
    diagnosticOp->emitError()
        << "layout candidate replacement requires source and target attrs";
    return failure();
  }
  return std::make_pair(source, target);
}

} // namespace mlir::triton::gluon::layout_autotune
