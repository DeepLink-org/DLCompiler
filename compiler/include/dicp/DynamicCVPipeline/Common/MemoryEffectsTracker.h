

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_MEMORY_EFFECTS_TRACKER_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_MEMORY_EFFECTS_TRACKER_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace CVPipeline {
constexpr int INIT_SIZE = 4;
class MemoryDependenceGraph {
public:
  MemoryDependenceGraph(Operation *root, AliasAnalysis &aa);

  ArrayRef<Operation *> getMemDefs(Operation *op) const;
  ArrayRef<Operation *> getMemUsers(Operation *op) const;

  ArrayRef<Operation *> getExecBefore(Operation *op) const;
  ArrayRef<Operation *> getExecAfter(Operation *op) const;

private:
  struct MemSlot {
    Value memref;
    Operation *lastWriter = nullptr;
    SmallPtrSet<Operation *, INIT_SIZE> pendingReads;
    bool rejectMayAlias = false;
    explicit MemSlot(Value v) : memref(v) {}
  };

  struct Snapshot {
    SmallVector<MemSlot> states;
  };

  void analyzeOp(Operation *op);
  void analyzeRegionsOf(Operation *op);

  SmallVector<MemoryEffects::EffectInstance> collectOuterEffects(Operation *op,
                                                                 bool &unknown);

  SmallVector<MemSlot *> findAliasSlots(Value v);
  ArrayRef<MemSlot *>
  resolveAliasSlots(Value v, DenseMap<Value, SmallVector<MemSlot *>> &cache);
  MemSlot *getOrCreateSlot(Value v);

  void collectPreds(ArrayRef<MemoryEffects::EffectInstance> effects,
                    bool unknown, SmallVectorImpl<Operation *> &defsOut,
                    SmallVectorImpl<Operation *> &predsOut);

  void applyEffects(Operation *op,
                    ArrayRef<MemoryEffects::EffectInstance> effects,
                    bool unknown);

  Snapshot takeSnapshot() const;
  void restoreSnapshot(Snapshot &&snap);

  void recordEdges(Operation *op, ArrayRef<Operation *> defs,
                   ArrayRef<Operation *> preds);

  Operation *root;
  AliasAnalysis &aa;

  // Data dependency
  DenseMap<Operation *, SmallVector<Operation *>> memDefs;
  DenseMap<Operation *, SmallVector<Operation *>> memUsers;

  // Execute order
  DenseMap<Operation *, SmallVector<Operation *>> execBefore;
  DenseMap<Operation *, SmallVector<Operation *>> execAfter;

  SmallVector<std::unique_ptr<MemSlot>> slots;
  DenseMap<Value, MemSlot *> valueToSlot;
};

} // namespace CVPipeline
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_MEMORY_EFFECTS_TRACKER_H
