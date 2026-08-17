#ifndef TRITON_DIALECT_GLUON_ANALYSIS_LAYOUTPROPAGATION_H_
#define TRITON_DIALECT_GLUON_ANALYSIS_LAYOUTPROPAGATION_H_

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <optional>

namespace mlir::triton::gluon {

inline bool isTransparentLayoutCarrierOp(Operation *op) {
  return isa<RegionBranchOpInterface, RegionBranchTerminatorOpInterface>(op);
}

inline bool isSupportedDotConstraintEncoding(Attribute encoding) {
  auto dot = dyn_cast<gpu::DotOperandEncodingAttr>(encoding);
  return dot && isa<gpu::MACAMmaEncodingAttr>(dot.getParent());
}

class LayoutEncoding {
public:
  LayoutEncoding() = default;
  explicit LayoutEncoding(Attribute encoding) : encoding(encoding) {}

  bool operator==(const LayoutEncoding &rhs) const {
    return encoding == rhs.encoding;
  }
  bool isUninitialized() const { return !encoding.has_value(); }
  bool isUnknown() const { return encoding == nullptr; }
  Attribute getLayoutEncoding() const {
    assert(!isUninitialized() && !isUnknown());
    return *encoding;
  }

  static LayoutEncoding meet(const LayoutEncoding &lhs,
                             const LayoutEncoding &rhs);
  static LayoutEncoding join(const LayoutEncoding &lhs,
                             const LayoutEncoding &rhs);
  static LayoutEncoding getUnknownLayout() {
    return LayoutEncoding(Attribute());
  }

  void print(raw_ostream &os) const;
  friend raw_ostream &operator<<(raw_ostream &os,
                                 const LayoutEncoding &layout) {
    layout.print(os);
    return os;
  }

private:
  std::optional<Attribute> encoding;
};

class LayoutEncodingLattice
    : public dataflow::Lattice<LayoutEncoding> {
public:
  using Lattice::Lattice;
};

class LayoutBackwardPropagation
    : public dataflow::SparseBackwardDataFlowAnalysis<
          LayoutEncodingLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;
  using SparseBackwardDataFlowAnalysis::propagateIfChanged;

  LogicalResult
  visitOperation(Operation *op, ArrayRef<LayoutEncodingLattice *> operands,
                 ArrayRef<const LayoutEncodingLattice *> results) override;
  void visitBranchOperand(OpOperand &operand) override;
  void visitCallOperand(OpOperand &operand) override;
  void setToExitState(LayoutEncodingLattice *) override;
  void visitNonControlFlowArguments(RegionSuccessor &,
                                    ArrayRef<BlockArgument>) {}

private:
  LogicalResult visitRegionInReverse(Operation *op);
  void visitWarpSpecRegionArgs(Operation *op, Value operand,
                               const LayoutEncoding &encoding);
};

class LayoutForwardPropagation
    : public dataflow::SparseForwardDataFlowAnalysis<LayoutEncodingLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  using SparseForwardDataFlowAnalysis::propagateIfChanged;

  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const LayoutEncodingLattice *> operands,
                 ArrayRef<LayoutEncodingLattice *> results) override;
  void setToEntryState(LayoutEncodingLattice *) override;

private:
  LogicalResult visitRegion(Operation *op);
  LogicalResult visitWarpSpecRegionArgs(Value result,
                                        const LayoutEncoding &encoding);
};

class TensorLayout {
public:
  TensorLayout() = default;
  explicit TensorLayout(Attribute encoding) : encoding(encoding) {}

  bool operator==(const TensorLayout &rhs) const {
    return encoding == rhs.encoding;
  }
  bool isUninitialized() const { return !encoding.has_value(); }
  bool isUnknown() const { return encoding == nullptr; }
  Attribute getLayoutEncoding() const {
    assert(!isUninitialized() && !isUnknown());
    return *encoding;
  }

  static TensorLayout meet(const TensorLayout &lhs, const TensorLayout &rhs);
  static TensorLayout join(const TensorLayout &lhs, const TensorLayout &rhs);
  static TensorLayout getUnknownLayout() { return TensorLayout(Attribute()); }

  void print(raw_ostream &os) const;
  friend raw_ostream &operator<<(raw_ostream &os, const TensorLayout &layout) {
    layout.print(os);
    return os;
  }

private:
  std::optional<Attribute> encoding;
};

class TensorLayoutLattice : public dataflow::Lattice<TensorLayout> {
public:
  using Lattice::Lattice;
};

class TensorBackwardPropagation
    : public dataflow::SparseBackwardDataFlowAnalysis<TensorLayoutLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;
  using SparseBackwardDataFlowAnalysis::propagateIfChanged;

  LogicalResult
  visitOperation(Operation *op, ArrayRef<TensorLayoutLattice *> operands,
                 ArrayRef<const TensorLayoutLattice *> results) override;
  void visitBranchOperand(OpOperand &operand) override;
  void visitCallOperand(OpOperand &operand) override;
  void setToExitState(TensorLayoutLattice *) override;
  void visitNonControlFlowArguments(RegionSuccessor &,
                                    ArrayRef<BlockArgument>) {}
};

} // namespace mlir::triton::gluon

#endif // TRITON_DIALECT_GLUON_ANALYSIS_LAYOUTPROPAGATION_H_
