

#ifndef TRITON_ADAPTER_ASCEND_NPU_IR_LEGALIZE_PASS_H
#define TRITON_ADAPTER_ASCEND_NPU_IR_LEGALIZE_PASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_DECL_ASCENDNPUIRLEGALIZE
#define GEN_PASS_DEF_ASCENDNPUIRLEGALIZE
#include "dicp/TritonToLinalg/Passes.h.inc"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAscendNPUIRLegalizePass();

std::unique_ptr<OperationPass<ModuleOp>>
createAscendNPUIRLegalizePass(const AscendNPUIRLegalizeOptions &options);

} // namespace triton
} // namespace mlir

class AscendNPUIRLegalizePass
    : public ::impl::AscendNPUIRLegalizeBase<AscendNPUIRLegalizePass> {
public:
  AscendNPUIRLegalizePass() = default;

  explicit AscendNPUIRLegalizePass(const AscendNPUIRLegalizeOptions &options)
      : AscendNPUIRLegalizeBase(options) {}

  void runOnOperation() override;
};

#endif // TRITON_ADAPTER_ASCEND_NPU_IR_LEGALIZE_PASS_H
