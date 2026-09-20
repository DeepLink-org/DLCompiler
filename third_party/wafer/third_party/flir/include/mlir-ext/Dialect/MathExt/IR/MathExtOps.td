#ifndef MATHEXT_OPS
#define MATHEXT_OPS

include "mlir-ext/Dialect/MathExt/IR/MathExtBase.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for math dialect ops.
class MathExt_Op<string mnemonic, list<Trait> traits = []> :
    Op<MathExt_Dialect, mnemonic, traits # [Pure,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface>] #
    ElementwiseMappable.traits>;

// Base class for unary math operations on integer types. Require an operand
// and result of the same type. This type can be an integer type, vector or
// tensor thereof.
class MathExt_IntegerUnaryOp<string mnemonic, list<Trait> traits = []> :
    MathExt_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins SignlessIntegerOrIndexLike:$operand);
  let results = (outs SignlessIntegerOrIndexLike:$result);

  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

// Base class for floating point classification ops. Require an operand and
// result of the same shape, which can be a floating point scalar, a vector or a
// tensor thereof.
class MathExt_FloatClassificationOp<string mnemonic, list<Trait> traits = []> :
    MathExt_Op<mnemonic,
      traits # [DeclareOpInterfaceMethods<ArithFastMathInterface>,
                TypesMatchWith<
          "result type has i1 element type and same shape as operands",
          "operand", "result", "::getI1SameShape($_self)">]> {
  let arguments = (ins FloatLike:$operand,
      DefaultValuedAttr<Arith_FastMathAttr,
                        "::mlir::arith::FastMathFlags::none">:$fastmath);
  let results = (outs BoolLike:$result);

  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

// Base class for unary math operations on floating point types. Require an
// operand and result of the same type. This type can be a floating point type,
// vector or tensor thereof.
class MathExt_FloatUnaryOp<string mnemonic, list<Trait> traits = []> :
    MathExt_Op<mnemonic,
        traits # [SameOperandsAndResultType,
                  DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let arguments = (ins FloatLike:$operand,
      DefaultValuedAttr<Arith_FastMathAttr,
                        "::mlir::arith::FastMathFlags::none">:$fastmath);
  let results = (outs FloatLike:$result);

  let assemblyFormat = [{ $operand (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($result) }];
}

// Base class for binary math operations on integer types. Require two
// operands and one result of the same type. This type can be an integer
// type, vector or tensor thereof.
class MathExt_IntegerBinaryOp<string mnemonic, list<Trait> traits = []> :
    MathExt_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins SignlessIntegerOrIndexLike:$lhs, SignlessIntegerOrIndexLike:$rhs);
  let results = (outs SignlessIntegerOrIndexLike:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// Base class for binary math operations on floating point types. Require two
// operands and one result of the same type. This type can be a floating point
// type, vector or tensor thereof.
class MathExt_FloatBinaryOp<string mnemonic, list<Trait> traits = []> :
    MathExt_Op<mnemonic,
        traits # [SameOperandsAndResultType,
                  DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let arguments = (ins FloatLike:$lhs, FloatLike:$rhs,
      DefaultValuedAttr<Arith_FastMathAttr,
                        "::mlir::arith::FastMathFlags::none">:$fastmath);
  let results = (outs FloatLike:$result);

  let assemblyFormat = [{ $lhs `,` $rhs (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($result) }];
}

// Base class for floating point ternary operations. Require three operands and
// one result of the same type. This type can be a floating point type, vector
// or tensor thereof.
class MathExt_FloatTernaryOp<string mnemonic, list<Trait> traits = []> :
    MathExt_Op<mnemonic,
        traits # [SameOperandsAndResultType,
                  DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let arguments = (ins FloatLike:$a, FloatLike:$b, FloatLike:$c,
      DefaultValuedAttr<Arith_FastMathAttr,
                        "::mlir::arith::FastMathFlags::none">:$fastmath);
  let results = (outs FloatLike:$result);

  let assemblyFormat = [{ $a `,` $b `,` $c (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($result) }];
}

//===----------------------------------------------------------------------===//
// FModOp
//===----------------------------------------------------------------------===//

def MathExt_FModOp : MathExt_FloatBinaryOp<"fmod"> {
  let summary = "floating point modulo (remainder) operation";
  let description = [{
    `%r = mathext.fmod %a, %b : f32`
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// DivRzOp
//===----------------------------------------------------------------------===//

def MathExt_DivRzOp : MathExt_FloatBinaryOp<"div_rz"> {
  let summary = "floating point division operation with round-zero";
  let description = [{
    `%r = mathext.div_rz %a, %b : f32`
  }];
  let hasFolder = 1;
}

#endif // MATHEXT_OPS
