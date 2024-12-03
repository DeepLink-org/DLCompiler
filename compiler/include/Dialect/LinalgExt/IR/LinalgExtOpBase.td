#ifndef LINALG_EXT_OPS_BASE
#define LINALG_EXT_OPS_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def LinalgExt_Dialect : Dialect {
  let name = "linalgext";
  let cppNamespace = "::mlir::dicp::LinalgExt";
  let summary = [{Linalg Extensions}];
  let description = [{
    A dialect designed for experimenting with non-structured operations that
    cannot be represented efficiently/directly by the Linalg dialect.
  }];
  // let extraClassDeclaration = [{ void registerTypes(); }];
  // let useDefaultAttributePrinterParser = 0;
}

class LinalgExt_Op<string mnemonic, list<Trait> traits = []>
    : Op<LinalgExt_Dialect, mnemonic, traits> {}

class RankedTensorOrMemRefOf<list<Type> allowedTypes>
    : ShapedContainerType<
          allowedTypes,
          Or<[IsMemRefTypePred, And<[IsTensorTypePred, HasRankPred]>]>,
          "ranked tensor or memref", "::mlir::ShapedType">;

def AnyRankedTensorOrMemRefType : RankedTensorOrMemRefOf<[AnyType]>;


def LinalgExt_ElementwiseTrait : NativeOpTrait<"LinalgExt_ElementwiseTrait">;
def LinalgExt_ReduceTrait : NativeOpTrait<"LinalgExt_ReduceTrait">;
def LinalgExt_TensorCoreTrait : NativeOpTrait<"LinalgExt_TensorCoreTrait">;

#endif // LINALG_EXT_OPS_BASE