# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

__all__ = ["custom", "custom_semantic", "register_custom_op"]

import inspect
import types
import typing
import itertools
import triton.language.core as tl
from .core import CORE, PIPE, MODE


# Registry for custom op, mapping name to its configuration.
_custom_op_registry = {}


def _get_op_class(name):
    # Try to get op class in _custom_op_registry.
    op_class = _custom_op_registry.get(name)
    if op_class is None:
        # Allow builtin custom ops used without registry.
        assert name.startswith("__builtin_"), f"Custom Op '{name}' not registered."
        op_class = type(
            "_builtin_custom_op",
            (object,),
            {
                "name": name,
                "core": CORE.VECTOR,
                "pipe": PIPE.PIPE_V,
                "mode": MODE.SIMT,
                "signature": inspect.signature(object),
            },
        )
    return op_class


def _unwrap_constexpr(arg):
    if isinstance(arg, tl.constexpr):
        return arg.value
    if isinstance(arg, (tuple, tl.tuple)):
        return tuple(_unwrap_constexpr(x) for x in arg)
    if isinstance(arg, list):
        return [_unwrap_constexpr(x) for x in arg]
    if isinstance(arg, dict):
        return {k: _unwrap_constexpr(v) for k, v in arg.items()}
    return arg


def _to_value(value, _semantic=None, ty=None):
    # Try to use 'type' attribute if ty not set.
    ty = getattr(value, "type", ty) if ty is None else ty
    if isinstance(value, tl.tensor):
        if not value.type.is_block() and isinstance(ty, tl.dtype) and value.type != ty:
            return _semantic.cast(value, ty).handle
        return value.handle
    if isinstance(value, bool):
        return _semantic.builder.get_int1(value)
    if isinstance(value, int):
        if isinstance(ty, tl.dtype):
            if ty.is_int64():
                return _semantic.builder.get_int64(value)
            if ty.is_uint64():
                return _semantic.builder.get_uint64(value)
            if ty.is_int32():
                return _semantic.builder.get_int32(value)
            if ty.is_uint32():
                return _semantic.builder.get_uint32(value)
            if ty.is_int16():
                return _semantic.builder.get_int16(value)
            if ty.is_uint16():
                return _semantic.builder.get_uint16(value)
            if ty.is_int8():
                return _semantic.builder.get_int8(value)
            if ty.is_uint8():
                return _semantic.builder.get_uint8(value)
        # default int32
        return _semantic.builder.get_int32(value)
    if isinstance(value, float):
        if isinstance(ty, tl.dtype):
            if ty.is_fp64():
                return _semantic.builder.get_fp64(value)
            if ty.is_fp32():
                return _semantic.builder.get_fp32(value)
            if ty.is_fp16():
                return _semantic.builder.get_fp16(value)
            if ty.is_bf16():
                return _semantic.builder.get_bf16(value)
        # default float32
        return _semantic.builder.get_fp32(value)
    if isinstance(value, tl.constexpr):
        return _to_value(value.value, _semantic)
    raise TypeError(f"Unsupported argument type {value} : {type(value)}")


def _to_operands(args, _semantic=None):
    operands = []
    for value in args:
        if value is None:
            continue
        if isinstance(value, (list, tuple, tl.tuple)):
            for item in value:
                operands.append(_to_value(item, _semantic))
        else:
            operands.append(_to_value(value, _semantic))
    return operands


def _get_element_type(ty):
    if isinstance(ty, types.GenericAlias):
        return typing.get_args(ty)[0]
    return ty


def _args_to_operands(op, _semantic, args, kwargs):
    if not op.signature.parameters:
        # Without parameters in signature, use the actual parameter order.
        return _to_operands(itertools.chain(args, kwargs.values()), _semantic)

    # Convert arguments to operands according the signature.
    operands = []
    bind = op.signature.bind(*args, **kwargs)
    for param in op.signature.parameters.values():
        value = bind.arguments.get(param.name)
        if value is None:
            continue
        ty = op.arg_type.get(param.name, param.annotation)
        if isinstance(value, (list, tuple, tl.tuple)):
            ty = _get_element_type(ty)
            for item in value:
                operands.append(_to_value(item, _semantic, ty))
        else:
            operands.append(_to_value(value, _semantic, ty))
    return operands


def _bind_op_arguments(op, args, kwargs):
    if not op.signature.parameters:
        return None
    return op.signature.bind(*args, **kwargs)


def _make_align_dim_attrs(op, builder, arg_attrs):
    name = "align_dim"
    if not hasattr(op, name):
        return

    align_arg_indices = {}
    if hasattr(op, "signature"):
        param_names = list(op.signature.parameters.keys())
        for arg_name in op.align_dim.keys():
            if arg_name in param_names:
                align_arg_indices[arg_name] = param_names.index(arg_name)

    for arg, align_val in op.align_dim.items():
        if isinstance(arg, str) and arg in align_arg_indices:
            arg_attrs[align_arg_indices[arg]] = {name: builder.get_int_attr(align_val)}
        elif isinstance(arg, int):
            arg_attrs[arg] = {name: builder.get_int_attr(align_val)}
        else:
            assert False, f"{name}'s keys should be string or int"


def _make_arg_attrs(op, builder):
    num_args = len(op.signature.parameters) if hasattr(op, "signature") else 0
    arg_attrs = [{} for _ in range(num_args)]
    _make_align_dim_attrs(op, builder, arg_attrs)
    return arg_attrs


def _get_bitcode_search_paths():
    """Return list of directories to search for bitcode files, ordered by priority."""
    import os
    from pathlib import Path

    paths = []

    # 1. Environment variable override (colon-separated)
    env_path = os.environ.get("DLCOMPILER_BITCODE_PATH")
    if env_path:
        for p in env_path.split(":"):
            p = p.strip()
            if p and Path(p).is_dir():
                paths.append(p)

    # 2. Relative to this file: language/deeplink/bitcode/bc/
    #    Works both in source tree and after pip install.
    local_bc = Path(__file__).parent.parent.parent / "bitcode" / "bc"
    if local_bc.is_dir():
        paths.append(str(local_bc))

    # 3. DLCOMPILER_SOURCE: project source root (set by set_env.sh)
    #    Used when this module runs from site-packages but bitcode is in source tree.
    source_root = os.environ.get("DLCOMPILER_SOURCE")
    if source_root:
        builtin_bc = Path(source_root) / "language" / "deeplink" / "bitcode" / "bc"
        if builtin_bc.is_dir():
            paths.append(str(builtin_bc))
        # Also check legacy dlcompiler/ location
        legacy_bc = Path(source_root) / "dlcompiler" / "bitcode" / "bc"
        if legacy_bc.is_dir():
            paths.append(str(legacy_bc))

    # 4. Legacy dsl/ directory for backward compatibility
    if source_root:
        legacy_dsl = Path(source_root) / "dsl"
        if legacy_dsl.is_dir():
            paths.append(str(legacy_dsl))

    return paths


def _resolve_bitcode_path(bitcode_ref):
    """Resolve a user-provided bitcode reference to an absolute file path.

    Resolution logic:
      1. If bitcode_ref is an existing file -> return its absolute path
      2. If bitcode_ref is a name -> search for {name}.aiv.bc in search paths
    """
    from pathlib import Path

    bitcode = Path(bitcode_ref)

    # Case 1: Already a valid existing file path
    if bitcode.is_file():
        return str(bitcode.absolute())

    # Case 2: Name resolution
    if bitcode.suffix == ".bc":
        search_name = bitcode.name
    else:
        search_name = f"{bitcode.name}.aiv.bc"

    for search_dir in _get_bitcode_search_paths():
        candidate = Path(search_dir) / search_name
        if candidate.is_file():
            return str(candidate.absolute())

    searched_dirs = "\n  ".join(_get_bitcode_search_paths())
    raise FileNotFoundError(
        f"Cannot find bitcode file for '{bitcode_ref}'.\n"
        f"  Searched for '{search_name}' in:\n"
        f"  {searched_dirs}\n"
        f"  Set DLCOMPILER_BITCODE_PATH to add custom search paths, "
        f"or provide an absolute path to an existing .aiv.bc file."
    )


def _add_optional_attr(op, name, builder, attrs):
    if hasattr(op, name):
        attrs[name] = builder.get_string_attr(getattr(op, name))


def _add_bitcode_attr(op, builder, attrs):
    name = "bitcode"
    if not hasattr(op, name):
        return

    from pathlib import Path

    bitcode_path = _resolve_bitcode_path(getattr(op, name))
    attrs[name] = builder.get_string_attr(bitcode_path)


def _add_optional_extra_buffer_attr(op, builder, attrs):
    name = "extra_buffers"
    if not hasattr(op, name):
        return

    extra_buffers = getattr(op, name)
    if isinstance(extra_buffers, tuple):
        extra_buffers = [extra_buffers]

    extra_buffer_types, extra_buffer_sizes = zip(*extra_buffers)
    if hasattr(builder, "get_type_array_attr"):
        attrs[name + "_types"] = builder.get_type_array_attr(
            [ty.to_ir(builder) for ty in extra_buffer_types]
        )
        attrs[name + "_sizes"] = builder.get_i64_array_attr(list(extra_buffer_sizes))
    else:
        type_strs = [str(ty.to_ir(builder)) for ty in extra_buffer_types]
        attrs[name + "_types"] = builder.parse_attr("[" + ", ".join(type_strs) + "]")
        size_strs = [str(s) for s in extra_buffer_sizes]
        attrs[name + "_sizes"] = builder.parse_attr("[" + ", ".join(size_strs) + "]")


def _add_optional_indexing_map_attr(op, builder, attrs):
    name = "indexing_map"
    if not hasattr(op, name):
        return
    indexing_map = getattr(op, name)
    attrs[name] = builder.get_affine_map_array_attr(indexing_map)


def _add_optional_iterator_types_attr(op, builder, attrs):
    name = "iterator_types"
    if not hasattr(op, name):
        return
    attrs[name] = builder.get_iterator_types_attr(
        [iterator_type.value for iterator_type in getattr(op, name)]
    )


def _make_attrs(op, builder):
    attrs = {
        "hivm.tcore_type": builder.get_core_type_attr(op.core.value),
        "hivm.pipe": builder.get_pipe_attr(op.pipe.value),
        "hivm.vf_mode": builder.get_vf_mode_attr(op.mode.value),
    }

    if not op.name.startswith("__builtin_"):
        assert hasattr(op, "symbol"), "Non builtin custom op, symbol is required."
        assert hasattr(
            op, "bitcode"
        ), "Non builtin custom op, bitcode path is required."

    _add_bitcode_attr(op, builder, attrs)
    _add_optional_indexing_map_attr(op, builder, attrs)
    _add_optional_iterator_types_attr(op, builder, attrs)
    _add_optional_extra_buffer_attr(op, builder, attrs)
    _add_optional_attr(op, "symbol", builder, attrs)
    _add_optional_attr(op, "source", builder, attrs)
    _add_optional_attr(op, "compile", builder, attrs)
    _add_optional_attr(op, "extra_attr", builder, attrs)

    return attrs


def _to_result(res, res_types):
    assert len(res) == len(res_types)
    n_res = len(res)
    if n_res == 0:
        return None
    if n_res == 1:
        return tl.tensor(res[0], res_types[0])
    return tl.tuple(tl.tensor(res[i], res_types[i]) for i in range(n_res))


def _init_op(op_class, *args, **kwargs):
    op = op_class.__new__(op_class)
    setattr(op, "arg_type", {})
    if op_class.signature.parameters:
        op_class.__init__(op, *args, **kwargs)
    return op


def custom_semantic(name: str, *args, _semantic=None, **kwargs):
    name = _unwrap_constexpr(name)
    op_class = _get_op_class(name)
    args = _unwrap_constexpr(args)
    kwargs = _unwrap_constexpr(kwargs)
    op = _init_op(op_class, *args, **kwargs)
    out = kwargs.pop("out", [])
    outs = out if isinstance(out, (list, tuple, tl.tuple)) else [out]
    outputs = _to_operands(outs, _semantic)
    inputs = _args_to_operands(op, _semantic, args, kwargs)
    builder = _semantic.builder
    attrs = _make_attrs(op, builder)
    arg_attrs = _make_arg_attrs(op, builder)
    res = builder.create_custom_op(name, attrs, inputs, outputs, arg_attrs)
    res_types = [out.type for out in outs]
    return _to_result(res, res_types)


@tl.builtin
def custom(name: str, *args, _semantic=None, **kwargs):
    """Invoke a custom operation with the given name and arguments."""
    return custom_semantic(name, *args, _semantic=_semantic, **kwargs)


def register_custom_op(op):
    """Register a custom operation so that we can invoke it using dl.custom()."""
    assert inspect.isclass(op), "@register_custom_op should decorate on a class."
    if not hasattr(op, "name"):
        setattr(op, "name", op.__name__)
    assert (
        op.name not in _custom_op_registry
    ), f"Custom op name '{op.name}' already used."

    assert hasattr(op, "core"), "'core' field is required."
    assert hasattr(op, "pipe"), "'pipe' field is required."
    assert hasattr(op, "mode"), "'mode' field is required."
    assert isinstance(op.core, CORE), "Invalid 'core' field, CORE type is required."
    assert isinstance(op.pipe, PIPE), "Invalid 'pipe' field, PIPE type is required."
    assert isinstance(op.mode, MODE), "Invalid 'mode' field, MODE type is required."
    signature = inspect.signature(op)
    setattr(op, "signature", signature)
    _custom_op_registry[op.name] = op
    return op


_dtype_cname_dict = {
    "int1": "bool",
    "int8": "int8_t",
    "int16": "int16_t",
    "int32": "int32_t",
    "int64": "int64_t",
    "uint8": "uint8_t",
    "uint16": "uint16_t",
    "uint32": "uint32_t",
    "uint64": "uint64_t",
    "fp16": "half",
    "bf16": "bfloat16_t",
    "fp32": "float",
    "fp64": "double",
    "fp8e5": "float8_e5m2_t",
    "fp8e4nv": "float8_e4m3_t",
}


def _cname(self):
    """Return the corresponding C name of the given tl.dtype"""
    return _dtype_cname_dict.get(self.name, self.name)


# Add 'cname' property to tl.dtype class.
tl.dtype.cname = property(_cname, None)
