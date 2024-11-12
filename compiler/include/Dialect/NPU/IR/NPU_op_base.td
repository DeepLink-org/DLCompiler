#ifndef NPU_IR_OP_BASE
#define NPU_IR_OP_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def NPU_Dialect : Dialect {
  let name = "npu";
  let description = [{NPU Dialect}];
  let cppNamespace = "::triton::deeplink::npu";
    let extraClassDeclaration = [{
      static std::string getTargetAttrStr() {return "target_name";
}
static std::string getTarget(mlir::ModuleOp modOp) {
  if (!modOp->hasAttr("target_name")) {
    llvm::report_fatal_error("Miss to define target");
  }
  return cast<mlir::StringAttr>(modOp->getAttr("target_name")).str();
}
}];
}

def AnyRankedTensorOrMemRefType : RankedTensorOrMemRefOf<[AnyType]>;
def NPU_ElementwiseTrait : NativeOpTrait<"ElementwiseTrait">;
def NPU_ReduceTrait : NativeOpTrait<"ReduceTrait">;
def NPU_ReduceWindowTrait : NativeOpTrait<"ReduceWindowTrait">;

#endif