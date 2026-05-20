from triton._C.libtriton import ir, dicp_triton

import enum
from functools import wraps
from typing import Union, List, Optional

import triton.language.core as tl
from triton.language.core import builtin, constexpr, tensor, _unwrap_if_constexpr

# Re-export from core.py
from .core import (
    insert_slice,
    extract_slice,
    sync_block_all,
    set_cross_flag,
    wait_cross_flag,
    parallel,
    inline_lambda,
    alloc,
    compile_hint,
    multibuffer,
    ND,
    NZ,
    fragment,
    UB,
    L1,
    L0A,
    L0B,
    L0C,
    SyncFlag,
)

# Re-export semantic module (for test_subview / test_alloc)
from . import semantic

# Re-export from libdevice.py
from .libdevice import flip, isfinited, finitef, atan2

# Re-export tl.gather
from triton.language.core import gather

# ---------------------------------------------------------------------------
# MLIR Affine bindings (from ascend_ir)
# ---------------------------------------------------------------------------

affine_expr = dicp_triton.ir.affine_expr
affine_constant_expr = dicp_triton.ir.affine_constant_expr
affine_dim_expr = dicp_triton.ir.affine_dim_expr
affine_symbol_expr = dicp_triton.ir.affine_symbol_expr
affine_binary_op_expr = dicp_triton.ir.affine_binary_op_expr
affine_map = dicp_triton.ir.affine_map

AffineExpr = affine_expr
AffineConstantExpr = affine_constant_expr
AffineDimExpr = affine_dim_expr
AffineSymbolExpr = affine_symbol_expr
AffineBinaryOpExpr = affine_binary_op_expr
AffineMap = affine_map

# ---------------------------------------------------------------------------
# Enums (wrapper over C++ ascend_ir enums)
# ---------------------------------------------------------------------------


class CORE(enum.Enum):
    VECTOR = dicp_triton.ir.CoreType.VECTOR
    CUBE = dicp_triton.ir.CoreType.CUBE
    CUBE_OR_VECTOR = dicp_triton.ir.CoreType.CUBE_OR_VECTOR
    CUBE_AND_VECTOR = dicp_triton.ir.CoreType.CUBE_AND_VECTOR


class PIPE(enum.Enum):
    PIPE_S = dicp_triton.ir.PIPE.PIPE_S
    PIPE_V = dicp_triton.ir.PIPE.PIPE_V
    PIPE_M = dicp_triton.ir.PIPE.PIPE_M
    PIPE_MTE1 = dicp_triton.ir.PIPE.PIPE_MTE1
    PIPE_MTE2 = dicp_triton.ir.PIPE.PIPE_MTE2
    PIPE_MTE3 = dicp_triton.ir.PIPE.PIPE_MTE3
    PIPE_ALL = dicp_triton.ir.PIPE.PIPE_ALL
    PIPE_FIX = dicp_triton.ir.PIPE.PIPE_FIX


class MODE(enum.Enum):
    SIMD = dicp_triton.ir.MODE.SIMD
    SIMT = dicp_triton.ir.MODE.SIMT
    MIX = dicp_triton.ir.MODE.MIX


class IteratorType(enum.Enum):
    Parallel = dicp_triton.ir.IteratorType.Parallel
    Broadcast = dicp_triton.ir.IteratorType.Broadcast
    Transpose = dicp_triton.ir.IteratorType.Transpose
    Reduction = dicp_triton.ir.IteratorType.Reduction
    Interleave = dicp_triton.ir.IteratorType.Interleave
    Deinterleave = dicp_triton.ir.IteratorType.Deinterleave
    Inverse = dicp_triton.ir.IteratorType.Inverse
    Pad = dicp_triton.ir.IteratorType.Pad
    Concat = dicp_triton.ir.IteratorType.Concat
    Gather = dicp_triton.ir.IteratorType.Gather
    Cumulative = dicp_triton.ir.IteratorType.Cumulative
    Opaque = dicp_triton.ir.IteratorType.Opaque


class FixpipeDMAMode(enum.Enum):
    NZ2DN = dicp_triton.ir.FixpipeDMAMode.NZ2DN
    NZ2ND = dicp_triton.ir.FixpipeDMAMode.NZ2ND
    NZ2NZ = dicp_triton.ir.FixpipeDMAMode.NZ2NZ


class FixpipeDualDstMode(enum.Enum):
    NO_DUAL = dicp_triton.ir.FixpipeDualDstMode.NO_DUAL
    COLUMN_SPLIT = dicp_triton.ir.FixpipeDualDstMode.COLUMN_SPLIT
    ROW_SPLIT = dicp_triton.ir.FixpipeDualDstMode.ROW_SPLIT


class FixpipePreQuantMode(enum.Enum):
    NO_QUANT = dicp_triton.ir.FixpipePreQuantMode.NO_QUANT
    F322BF16 = dicp_triton.ir.FixpipePreQuantMode.F322BF16
    F322F16 = dicp_triton.ir.FixpipePreQuantMode.F322F16
    S322I8 = dicp_triton.ir.FixpipePreQuantMode.S322I8


class FixpipePreReluMode(enum.Enum):
    LEAKY_RELU = dicp_triton.ir.FixpipePreReluMode.LEAKY_RELU
    NO_RELU = dicp_triton.ir.FixpipePreReluMode.NO_RELU
    NORMAL_RELU = dicp_triton.ir.FixpipePreReluMode.NORMAL_RELU
    P_RELU = dicp_triton.ir.FixpipePreReluMode.P_RELU


class SYNC_IN_VF(enum.Enum):
    VV_ALL = enum.auto()
    VST_VLD = enum.auto()
    VLD_VST = enum.auto()
    VST_VST = enum.auto()
    VS_ALL = enum.auto()
    VST_LD = enum.auto()
    VLD_ST = enum.auto()
    VST_ST = enum.auto()
    SV_ALL = enum.auto()
    ST_VLD = enum.auto()
    LD_VST = enum.auto()
    ST_VST = enum.auto()


# ---------------------------------------------------------------------------
# Address Space
# ---------------------------------------------------------------------------


def _get_bl():
    from . import buffer_core as bl

    return bl


class _ascend_address_space_base:
    def __init__(self, address_space_value):
        self.real_address_space = address_space_value

    def to_ir(self, builder: ir.builder) -> ir.attribute:
        return builder.get_target_attribute(self.real_address_space)


class ascend_address_space:
    def __init__(self):
        for k, v in {
            k: v
            for k, v in dicp_triton.ir.AddressSpace.__dict__.items()
            if isinstance(v, dicp_triton.ir.AddressSpace)
        }.items():
            setattr(self, k, _ascend_address_space_base(v))


ascend_address_space = ascend_address_space()

# ---------------------------------------------------------------------------
# int64 helper
# ---------------------------------------------------------------------------


class int64(int):
    def __new__(cls, value):
        obj = int.__new__(cls, value)
        obj.type = tl.int64
        return obj


# ---------------------------------------------------------------------------
# Core Ops
# ---------------------------------------------------------------------------


@builtin
def copy_from_ub_to_l1(src, dst, _semantic=None):
    from warnings import warn

    warn("copy_from_ub_to_l1 is deprecated, please use copy instead.")
    bl = _get_bl()
    if isinstance(src, tl.tensor) or isinstance(dst, tl.tensor):
        raise TypeError("tensor not support yet")
    if src.shape != dst.shape:
        raise TypeError("src and dst must have same shape")
    if src.dtype != dst.dtype:
        raise TypeError("src and dst need to have the same type")
    if isinstance(src, bl.buffer) and isinstance(dst, bl.buffer):
        if src.space != ascend_address_space.UB:
            raise TypeError("src's AddressSpace must be UB")
        if dst.space != ascend_address_space.L1:
            raise TypeError("dst's AddressSpace must be L1")
        _semantic.builder.create_copy_buffer(src.handle, dst.handle)
    else:
        raise TypeError("src and dst must be tl.tensor or bl.buffer")


@builtin
def copy(src, dst, _semantic=None):
    bl = _get_bl()
    if isinstance(src, tl.tensor) or isinstance(dst, tl.tensor):
        raise TypeError("tensor not support yet")
    if src.shape != dst.shape:
        raise TypeError("src and dst must have same shape")
    if src.dtype != dst.dtype:
        raise TypeError("src and dst need to have the same type")
    if isinstance(src, bl.buffer) and isinstance(dst, bl.buffer):
        if src.space != ascend_address_space.UB:
            raise TypeError("src's AddressSpace must be UB")
        if dst.space not in (ascend_address_space.L1, ascend_address_space.UB):
            raise TypeError("dst's AddressSpace must be UB or L1")
        _semantic.builder.create_copy_buffer(src.handle, dst.handle)
    else:
        raise TypeError("src and dst must be tl.tensor or bl.buffer")


@builtin
def fixpipe(
    src,
    dst=None,
    dma_mode=FixpipeDMAMode.NZ2ND,
    dual_dst_mode=FixpipeDualDstMode.NO_DUAL,
    _semantic=None,
):
    pre_quant_mode = FixpipePreQuantMode.NO_QUANT
    pre_relu_mode = FixpipePreReluMode.NO_RELU
    dst_handle = dst.handle if dst is not None else None
    result = _semantic.builder.create_fixpipe(
        src.handle,
        dst_handle,
        dma_mode.value,
        dual_dst_mode.value,
        pre_quant_mode.value,
        pre_relu_mode.value,
    )
    if dst is None:
        if dual_dst_mode == FixpipeDualDstMode.ROW_SPLIT:
            new_shape = list(src.type.shape)
            if len(new_shape) >= 1 and new_shape[0] > 0:
                new_shape[0] = new_shape[0] // 2
            return tl.tensor(result, tl.block_type(src.type.scalar, new_shape))
        elif dual_dst_mode == FixpipeDualDstMode.COLUMN_SPLIT:
            new_shape = list(src.type.shape)
            if len(new_shape) >= 2 and new_shape[1] > 0:
                new_shape[1] = new_shape[1] // 2
            return tl.tensor(result, tl.block_type(src.type.scalar, new_shape))
        else:
            return tl.tensor(result, src.type)
    # When dst is provided, C++ returns None (in-place)


@builtin
def debug_barrier(sync_mode: SYNC_IN_VF, _semantic=None):
    target = tl.tensor(_semantic.builder.get_int64(0), tl.int64)
    attr = _semantic.builder.get_string_attr(sync_mode.name)
    _semantic.builder.create_debug_barrier(target.handle, "SYNC_IN_VF", attr)


@builtin
def sub_vec_id(_semantic=None):
    return tl.tensor(_semantic.builder.create_get_sub_vec_id(), tl.int64)


@builtin
def sub_vec_num(_semantic=None):
    from triton.backends.dicp_triton.utils import NPUUtils

    npuUtils = NPUUtils()
    cube_num = npuUtils.get_aivector_core_num()
    vector_num = npuUtils.get_aicore_num()
    const_val = cube_num // vector_num
    return tl.constexpr(const_val)


def _sync_block_event_id_handle(event_id, _semantic):
    if isinstance(event_id, int):
        return _semantic.to_tensor(tl.constexpr(event_id)).handle
    elif isinstance(event_id, constexpr):
        return _semantic.to_tensor(event_id).handle
    else:
        return event_id.handle


@builtin
def sync_block_set(
    sender, receiver, event_id, sender_pipe, receiver_pipe, _semantic=None
):
    sender = _unwrap_if_constexpr(sender)
    receiver = _unwrap_if_constexpr(receiver)
    id_handle = _sync_block_event_id_handle(event_id, _semantic)
    _semantic.builder.sync_block_set(
        sender, receiver, id_handle, sender_pipe, receiver_pipe
    )


@builtin
def sync_block_wait(
    sender, receiver, event_id, sender_pipe, receiver_pipe, _semantic=None
):
    sender = _unwrap_if_constexpr(sender)
    receiver = _unwrap_if_constexpr(receiver)
    id_handle = _sync_block_event_id_handle(event_id, _semantic)
    _semantic.builder.sync_block_wait(
        sender, receiver, id_handle, sender_pipe, receiver_pipe
    )


# ---------------------------------------------------------------------------
# Vec Ops
# ---------------------------------------------------------------------------


@builtin
def get_element(src, indice, _semantic=None, _generator=None):
    assert len(src.shape) > 0
    new_indice = [
        _semantic.to_tensor(i) if isinstance(i, constexpr) else i for i in indice
    ]
    new_indice_handles = []
    for i in new_indice:
        if isinstance(i, tensor):
            new_indice_handles.append(i.handle)
        elif isinstance(i, int):
            new_indice_handles.append(i)
        else:
            new_indice_handles.append(i.handle if hasattr(i, "handle") else i)
    result = _semantic.builder.create_extract_scalar(src.handle, new_indice_handles)
    return _semantic.wrap_tensor(result, src.type.scalar, None)


@builtin
def sort(ptr, dim=-1, descending=False, _semantic=None):
    dim = _unwrap_if_constexpr(dim)
    if hasattr(descending, "value"):
        descending = bool(descending.value)
    else:
        descending = bool(descending)
    sorted_vals = _semantic.builder.create_sort(ptr.handle, dim, descending)
    return tensor(sorted_vals, type=ptr.type)


# ---------------------------------------------------------------------------
# Mem Ops
# ---------------------------------------------------------------------------


@builtin
def index_select_simd(
    src, dim, index, src_shape, src_offset, read_shape, _semantic=None
):
    dim = _unwrap_if_constexpr(dim)
    newsrc_shape = [
        _semantic.to_tensor(o) if isinstance(o, constexpr) else o for o in src_shape
    ]
    newsrc_offset = [
        _semantic.to_tensor(o) if isinstance(o, constexpr) else o for o in src_offset
    ]
    assert len(index.shape) == 1, "index must be a 1D tensor"

    ndim = len(newsrc_shape)
    return_shape = [index.shape[0] if i == dim else read_shape[i] for i in range(ndim)]
    element_ty = src.type.element_ty
    output_ty = tl.block_type(element_ty, return_shape)

    newsrc_shape_handles = [
        s.handle if isinstance(s, tensor) else s for s in newsrc_shape
    ]
    newsrc_offset_handles = [
        s.handle if isinstance(s, tensor) else s for s in newsrc_offset
    ]
    out = _semantic.builder.create_index_select_simd(
        src.handle,
        index.handle,
        dim,
        newsrc_shape_handles,
        newsrc_offset_handles,
        read_shape,
        return_shape,
    )
    return tl.tensor(out, output_ty)


@builtin
def gather_out_to_ub(
    src,
    index,
    index_boundary,
    dim,
    src_stride,
    end_offset,
    start_offset,
    other=None,
    _semantic=None,
):
    dim = _unwrap_if_constexpr(dim)
    index_boundary = _unwrap_if_constexpr(index_boundary)

    src_stride_handles = [s.handle if isinstance(s, tensor) else s for s in src_stride]
    end_offset_handles = [s.handle if isinstance(s, tensor) else s for s in end_offset]
    start_offset_handles = [
        s.handle if isinstance(s, tensor) else s for s in start_offset
    ]
    other_handle = None
    if other is not None:
        other = _semantic.cast(other, src.dtype.element_ty)
        other_handle = other.handle if isinstance(other, tensor) else other

    ret = _semantic.builder.create_gather_out_to_ub(
        src.handle,
        index.handle,
        index_boundary,
        dim,
        src_stride_handles,
        end_offset_handles,
        start_offset_handles,
        other_handle,
    )
    ret_shape = [_unwrap_if_constexpr(s) for s in index.shape]
    return _semantic.wrap_tensor(ret, src.dtype.element_ty, ret_shape)


@builtin
def scatter_ub_to_out(
    ptr,
    value,
    index,
    index_boundary,
    dim,
    dst_stride,
    end_offset,
    start_offset,
    _semantic=None,
):
    dim = _unwrap_if_constexpr(dim)
    index_boundary = _unwrap_if_constexpr(index_boundary)

    dst_stride_handles = [s.handle if isinstance(s, tensor) else s for s in dst_stride]
    end_offset_handles = [s.handle if isinstance(s, tensor) else s for s in end_offset]
    start_offset_handles = [
        s.handle if isinstance(s, tensor) else s for s in start_offset
    ]

    return tl.tensor(
        _semantic.builder.create_scatter_ub_to_out(
            ptr.handle,
            value.handle,
            index.handle,
            index_boundary,
            dim,
            dst_stride_handles,
            end_offset_handles,
            start_offset_handles,
        ),
        tl.void,
    )


@builtin
def index_put(
    ptr,
    index,
    value,
    dim,
    index_boundary,
    end_offset,
    start_offset,
    dst_stride,
    _semantic=None,
):
    dim = _unwrap_if_constexpr(dim)
    index_boundary = _unwrap_if_constexpr(index_boundary)

    end_offset_handles = [s.handle if isinstance(s, tensor) else s for s in end_offset]
    start_offset_handles = [
        s.handle if isinstance(s, tensor) else s for s in start_offset
    ]
    dst_stride_handles = [s.handle if isinstance(s, tensor) else s for s in dst_stride]

    return tl.tensor(
        _semantic.builder.create_index_put(
            ptr.handle,
            index.handle,
            value.handle,
            dim,
            index_boundary,
            end_offset_handles,
            start_offset_handles,
            dst_stride_handles,
        ),
        tl.void,
    )


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


class scope:
    def __init__(self, core_mode: str, _builder=None, _semantic=None, **kwargs):
        self.core_mode = (
            _unwrap_if_constexpr(core_mode) if _builder is None else core_mode
        )
        self._builder = _builder
        self._semantic = _semantic
        self.disable_auto_sync = kwargs.get("disable_auto_sync", False)
        if self.core_mode not in ("cube", "vector"):
            raise ValueError(
                f'core_mode must be "cube" or "vector", got {self.core_mode}'
            )

    def __enter__(self):
        if self._builder is None:
            raise RuntimeError("scope can only be used inside a Triton kernel")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# ---------------------------------------------------------------------------
# Custom Op Framework
# ---------------------------------------------------------------------------

import inspect
import types
import typing
import itertools

_custom_op_registry = {}


def _get_op_class(name):
    op_class = _custom_op_registry.get(name)
    if op_class is None:
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
        return _to_operands(itertools.chain(args, kwargs.values()), _semantic)
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


def _add_optional_attr(op, name, builder, attrs):
    if hasattr(op, name):
        attrs[name] = builder.get_string_attr(getattr(op, name))


def _add_bitcode_attr(op, builder, attrs):
    name = "bitcode"
    if not hasattr(op, name):
        return
    from pathlib import Path

    bitcode = Path(getattr(op, name))
    assert bitcode.exists(), f"Provided bitcode ({name}) not exist"
    attrs[name] = builder.get_string_attr(str(bitcode.absolute()))


def _add_optional_extra_buffer_attr(op, builder, attrs):
    name = "extra_buffers"
    if not hasattr(op, name):
        return
    extra_buffers = getattr(op, name)
    if isinstance(extra_buffers, tuple):
        extra_buffers = [extra_buffers]
    extra_buffer_types, extra_buffer_sizes = zip(*extra_buffers)
    # Use parse_attr as fallback if get_type_array_attr/get_i64_array_attr
    # are not available on this builder (they require triton-ascend ir.cc patches)
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


@builtin
def custom(name: str, *args, _semantic=None, **kwargs):
    return custom_semantic(name, *args, _semantic=_semantic, **kwargs)


def register_custom_op(op):
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


# ---------------------------------------------------------------------------
# dtype.cname injection
# ---------------------------------------------------------------------------

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
    return _dtype_cname_dict.get(self.name, self.name)


tl.dtype.cname = property(_cname, None)

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # core re-exports
    "insert_slice",
    "extract_slice",
    "sync_block_all",
    "set_cross_flag",
    "wait_cross_flag",
    "parallel",
    "inline_lambda",
    "alloc",
    "compile_hint",
    "multibuffer",
    "ND",
    "NZ",
    "fragment",
    "UB",
    "L1",
    "L0A",
    "L0B",
    "L0C",
    "SyncFlag",
    # semantic module
    "semantic",
    # math re-exports
    "flip",
    "atan2",
    "isfinited",
    "finitef",
    # MLIR affine
    "affine_expr",
    "affine_constant_expr",
    "affine_dim_expr",
    "affine_symbol_expr",
    "affine_binary_op_expr",
    "affine_map",
    "AffineExpr",
    "AffineConstantExpr",
    "AffineDimExpr",
    "AffineSymbolExpr",
    "AffineBinaryOpExpr",
    "AffineMap",
    # enums
    "CORE",
    "PIPE",
    "MODE",
    "IteratorType",
    "FixpipeDMAMode",
    "FixpipeDualDstMode",
    "FixpipePreQuantMode",
    "FixpipePreReluMode",
    "SYNC_IN_VF",
    # address space
    "ascend_address_space",
    # int64
    "int64",
    # core ops
    "copy",
    "copy_from_ub_to_l1",
    "fixpipe",
    "debug_barrier",
    "sub_vec_id",
    "sub_vec_num",
    "sync_block_set",
    "sync_block_wait",
    # vec ops
    "get_element",
    "sort",
    "gather",
    # mem ops
    "index_put",
    "gather_out_to_ub",
    "scatter_ub_to_out",
    "index_select_simd",
    # scope
    "scope",
    # custom op
    "custom",
    "custom_semantic",
    "register_custom_op",
]
