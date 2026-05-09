import inspect
import types
import typing
import itertools
import enum
from triton.language import core, semantic
import triton.language as tl

__all__ = ["custom", "custom_semantic", "register_custom_op", "CORE", "PIPE", "MODE"]

_custom_op_registry = {}


class CORE(enum.Enum):
    CUBE = "CUBE"
    VECTOR = "VECTOR"
    CUBE_OR_VECTOR = "CUBE_OR_VECTOR"
    CUBE_AND_VECTOR = "CUBE_AND_VECTOR"


class PIPE(enum.Enum):
    PIPE_S = "PIPE_S"
    PIPE_V = "PIPE_V"
    PIPE_M = "PIPE_M"
    PIPE_MTE1 = "PIPE_MTE1"
    PIPE_MTE2 = "PIPE_MTE2"
    PIPE_MTE3 = "PIPE_MTE3"
    PIPE_ALL = "PIPE_ALL"
    PIPE_FIX = "PIPE_FIX"


class MODE(enum.Enum):
    SIMD = "SIMD"
    SIMT = "SIMT"
    MIX = "MIX"


def _get_op_class(name):
    # Try to get op class in _custom_op_registry.
    op_class = _custom_op_registry.get(name)
    if op_class is None:
        # Allow bulitin custom ops used without registry.
        assert name.startswith("__builtin_"), f"Custom Op '{name}' not registered."
        # Return a dummy op class for builtin custom op.
        op_class = type(
            "_builtin_custom_op",
            (object,),
            {
                "name": name,
                "core": core.CORE.VECTOR,
                "pipe": core.PIPE.PIPE_V,
                "mode": core.MODE.SIMT,
                "signature": inspect.signature(object),
            },
        )
    return op_class


def _unwrap_constexpr(arg):
    if isinstance(arg, tl.constexpr):
        return arg.value
    if isinstance(arg, tuple):
        return tuple(_unwrap_constexpr(x) for x in arg)
    if isinstance(arg, list):
        return [_unwrap_constexpr(x) for x in arg]
    if isinstance(arg, dict):
        return {k: _unwrap_constexpr(v) for k, v in arg.items()}
    return arg


def _to_value(value, builder, ty=None):
    # Try to use 'type' attribute if ty not set.
    ty = getattr(value, "type", ty) if ty is None else ty
    if isinstance(value, tl.tensor):
        if not value.type.is_block() and isinstance(ty, tl.dtype) and value.type != ty:
            # For a scalar variable, if its type is not the expected one
            # that specified by type hint 'ty', insert a cast for it.
            return tl.semantic.cast(value, ty, builder).handle
        return value.handle
    if isinstance(value, bool):
        return builder.get_int1(value)
    if isinstance(value, int):
        if isinstance(ty, tl.dtype):
            if ty.is_int64():
                return builder.get_int64(value)
            if ty.is_uint64():
                return builder.get_uint64(value)
            if ty.is_int32():
                return builder.get_int32(value)
            if ty.is_uint32():
                return builder.get_uint32(value)
            if ty.is_int16():
                return builder.get_int16(value)
            if ty.is_uint16():
                return builder.get_uint16(value)
            if ty.is_int8():
                return builder.get_int8(value)
            if ty.is_uint8():
                return builder.get_uint8(value)
        # default int32
        return builder.get_int32(value)
    if isinstance(value, float):
        if isinstance(ty, tl.dtype):
            if ty.is_fp64():
                return builder.get_fp64(value)
            if ty.is_fp32():
                return builder.get_fp32(value)
            if ty.is_fp16():
                return builder.get_fp16(value)
            if ty.is_bf16():
                return builder.get_bf16(value)
        # default float32
        return builder.get_fp32(value)
    if isinstance(value, tl.constexpr):
        return _to_value(value.value, builder)
    raise TypeError(f"Unsupported argument type {value} : {type(value)}")


def _to_operands(args, builder):
    operands = []
    for value in args:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                operands.append(_to_value(item, builder))
        else:
            operands.append(_to_value(value, builder))
    return operands


def _get_element_type(ty):
    if isinstance(ty, types.GenericAlias):
        return typing.get_args(ty)[0]
    return ty


def _args_to_operands(op, builder, args, kwargs):
    if not op.signature.parameters:
        # Without parameters in signature, use the actual parameter order.
        return _to_operands(itertools.chain(args, kwargs.values()), builder)

    # Convert arguments to operands according the signature.
    operands = []
    bind = op.signature.bind(*args, **kwargs)
    for param in op.signature.parameters.values():
        value = bind.arguments.get(param.name, None)
        if value is None:
            continue
        ty = op.arg_type.get(param.name, param.annotation)
        if isinstance(value, (list, tuple)):
            ty = _get_element_type(ty)
            for item in value:
                operands.append(_to_value(item, builder, ty))
        else:
            operands.append(_to_value(value, builder, ty))
    return operands


def _add_optional_attr(op, name, builder, attrs):
    if hasattr(op, name):
        attrs[name] = getattr(op, name)


def _add_bitcode_attr(op, builder, attrs):
    if not hasattr(op, "bitcode"):
        return
    from pathlib import Path

    bitcode = Path(getattr(op, "bitcode"))
    assert bitcode.exists(), f"Provided bitcode ({bitcode}) not exist"
    attrs["bitcode"] = str(bitcode.absolute())


def _make_attrs(op, builder):
    attrs = {
        "hivm.tcore_type": f"#hivm.tcore_type<{op.core.value}>",
        "hivm.pipe": f"#hivm.pipe<{op.pipe.value}>",
        "hivm.vf_mode": f"#hivm.vf_mode<{op.mode.value}>",
    }

    if not op.name.startswith("__builtin_"):
        assert hasattr(op, "symbol"), f"Non builtin custom op, symbol is required."
        assert hasattr(
            op, "bitcode"
        ), f"Non builtin custom op, bitcode path is required."

    # Add bit code path attribute, formalize to abosulte path.
    _add_bitcode_attr(op, builder, attrs)

    _add_optional_attr(op, "symbol", builder, attrs)
    _add_optional_attr(op, "source", builder, attrs)
    _add_optional_attr(op, "compile", builder, attrs)
    # Extra attributes can be added here, such as op.extra_attr="attr_a=xx"
    _add_optional_attr(op, "extra_attr", builder, attrs)
    _add_optional_attr(op, "iterator_types", builder, attrs)
    return attrs


def _to_result(res, res_types):
    assert len(res) == len(res_types)
    n_res = len(res)
    if n_res == 0:
        return None
    if n_res == 1:
        return tl.tensor(res[0], res_types[0])
    return tuple(tl.tensor(res[i], res_types[i]) for i in range(n_res))


def _init_op(op_class, *args, **kwargs):
    op = op_class.__new__(op_class)
    # Add arg_type dict to support dynamic argument type specifying.
    setattr(op, "arg_type", {})
    if op_class.signature.parameters:
        # Init with arguments validate.
        op_class.__init__(op, *args, **kwargs)
    return op


def custom_semantic(name: str, *args, _semantic=None, **kwargs):
    _builder = _semantic.builder
    name = _unwrap_constexpr(name)
    assert name in _custom_op_registry, f"Custom op '{name}' not found."
    # Get op class according the name.
    op_class = _get_op_class(name)
    # Convert constexpr to value in arguments.
    args = _unwrap_constexpr(args)
    kwargs = _unwrap_constexpr(kwargs)
    # Create op instance from op class with the arguments.
    op = _init_op(op_class, *args, **kwargs)
    # Prepare inputs and outputs operands.
    out = kwargs.pop("out", [])
    outs = out if isinstance(out, (list, tuple)) else [out]
    outputs = _to_operands(outs, _builder)
    inputs = _args_to_operands(op, _builder, args, kwargs)
    # Setup attributes.
    attrs = _make_attrs(op, _builder)
    # Build IR for the custom op.
    res = _builder.create_custom_op(name, attrs, inputs, outputs)
    # Results with same types as outputs.
    res_types = [out.type for out in outs]
    return _to_result(res, res_types)


@core.builtin
def custom(name: str, *args, _semantic=None, **kwargs):
    """Invoke a custom operation with the given name and arguments."""
    return custom_semantic(name, *args, _semantic=_semantic, **kwargs)


def register_custom_op(op):
    """Register a custom operation so that we can invoke it using al.custom()."""
    assert inspect.isclass(op), "@register_custom_op should decorate on a class."
    # Use class name if name not set.
    if not hasattr(op, "name"):
        setattr(op, "name", op.__name__)
    # The op name should not be used.
    assert (
        op.name not in _custom_op_registry
    ), f"Custom op name '{op.name}' already used."
    # Check required core, pipe, mode fields.
    assert hasattr(op, "core"), "'core' field is required."
    assert hasattr(op, "pipe"), "'pipe' field is required."
    assert hasattr(op, "mode"), "'mode' field is required."
    assert isinstance(op.core, CORE), "Invalid 'core' field, CORE type is required."
    assert isinstance(op.pipe, PIPE), "Invalid 'pipe' field, PIPE type is required."
    assert isinstance(op.mode, MODE), "Invalid 'mode' field, MODE type is required."
    # Retrieve arguments signature from __init__ method and save it.
    signature = inspect.signature(op)
    setattr(op, "signature", signature)
    # Register the custom op configuration.
    _custom_op_registry[op.name] = op
    return op


# -----------------------
# SPMD Programming Model
# -----------------------
def _constexpr_to_value(v):
    if isinstance(v, tl.constexpr):
        return v.value
    return v


def _is_int_like_elem(x) -> bool:
    """Accept int / tl.constexpr(int) / tl.tensor(int*)."""
    if isinstance(x, int):
        return True
    if isinstance(x, tl.constexpr):
        # constexpr value should be python int
        return isinstance(x.value, int)
    if isinstance(x, tl.tensor):
        # Offsets/strides must be integer typed (i32/i64 etc.)
        return x.dtype.is_int()
    return False


def _assert_int_like_tuple(name: str, xs):
    assert isinstance(
        xs, (tuple, list)
    ), f"{name} should be a tuple/list, but got {type(xs)}"
    assert all(_is_int_like_elem(x) for x in xs), f"{name} should be integer"
