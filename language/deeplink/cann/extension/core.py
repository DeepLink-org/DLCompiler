# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

__all__ = [
    "ascend_address_space",
    "builtin",
    "CORE",
    "copy_from_ub_to_l1",
    "copy",
    "debug_barrier",
    "fixpipe",
    "FixpipeDMAMode",
    "FixpipeDualDstMode",
    "FixpipePreQuantMode",
    "FixpipePreReluMode",
    "int64",
    "is_builtin",
    "MODE",
    "PIPE",
    "IteratorType",
    "sub_vec_id",
    "sub_vec_num",
    "sync_block_all",
    "sync_block_set",
    "sync_block_wait",
    "alloc",
    "SyncFlag",
    "set_cross_flag",
    "wait_cross_flag",
    "SYNC_IN_VF",
]

import enum
from typing import TypeVar, List, Union
from functools import wraps

from triton._C.libtriton import ir, dicp_triton
import triton.language.core as tl
from triton.language.core import _shape_check_impl, _unwrap_if_constexpr

from triton.backends.dicp_triton.npu_driver import NPUUtils

from . import semantic as semantic


T = TypeVar("T")

TRITON_BUILTIN = "__triton_builtin__"
ASCEND_BUILTIN = "__ascend_builtin__"


def _constexpr_to_value(v):
    if isinstance(v, tl.constexpr):
        return v.value
    return v


def builtin(fn: T) -> T:
    """Mark a function as a buffer language builtin."""
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "_semantic" not in kwargs or kwargs["_semantic"] is None:
            raise ValueError(
                "Did you forget to add @triton.jit ? "
                "(`_semantic` argument must be provided outside of JIT functions.)"
            )
        return fn(*args, **kwargs)

    setattr(wrapper, TRITON_BUILTIN, True)
    setattr(wrapper, ASCEND_BUILTIN, True)

    return wrapper


def is_builtin(fn) -> bool:
    """Is this a registered ascend language builtin function?"""
    return getattr(fn, ASCEND_BUILTIN, False)


class int64(int):
    def __new__(cls, value):
        obj = int.__new__(cls, value)
        obj.type = tl.int64
        return obj


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


class ascend_address_space_base:
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
            setattr(self, k, ascend_address_space_base(v))


ascend_address_space = ascend_address_space()


@builtin
def sub_vec_id(_semantic=None) -> tl.tensor:
    return semantic.sub_vec_id(_semantic)


@builtin
def copy_from_ub_to_l1(src, dst, _semantic=None):
    from warnings import warn
    warn("copy_from_ub_to_l1 is deprecated, please use copy instead.")
    return semantic.copy_from_ub_to_l1(src, dst, _semantic)


@builtin
def copy(src, dst, _semantic=None):
    return semantic.copy(src, dst, _semantic)


def create_sync_block(sender, receiver, event_id, is_set: bool,
                      sender_pipe=None, receiver_pipe=None,
                      _semantic=None):
    sender = _unwrap_if_constexpr(sender)
    receiver = _unwrap_if_constexpr(receiver)
    assert isinstance(sender, str) and sender in ("cube", "vector"), f"ERROR: sender = {sender}"
    assert isinstance(receiver, str) and receiver in ("cube", "vector"), f"ERROR: receiver = {receiver}"
    if isinstance(event_id, int):
        assert 0 <= event_id < 16, f"event_id: {event_id} should be 0 ~ 15"
    if sender == receiver:
        raise ValueError(f"Unexpected pair: {sender} -> {receiver}")
    if sender_pipe is None and receiver_pipe is None:
        if sender == "cube":
            sender_pipe = PIPE.PIPE_FIX
            receiver_pipe = PIPE.PIPE_MTE2
        if sender == "vector":
            sender_pipe = PIPE.PIPE_MTE3
            receiver_pipe = PIPE.PIPE_MTE2
    if not isinstance(sender_pipe, PIPE) or not isinstance(receiver_pipe, PIPE):
        raise TypeError("sender_pipe and receiver_pipe must be instances of PIPE enum")
    if is_set:
        return semantic.create_sync_block_set(sender, receiver, event_id, sender_pipe, receiver_pipe, _semantic)
    return semantic.create_sync_block_wait(sender, receiver, event_id, sender_pipe, receiver_pipe, _semantic)


@builtin
def sync_block_set(sender, receiver, event_id, sender_pipe=None, receiver_pipe=None, _semantic=None):
    return create_sync_block(sender, receiver, event_id, True, sender_pipe, receiver_pipe, _semantic)


@builtin
def sync_block_wait(sender, receiver, event_id, sender_pipe=None, receiver_pipe=None, _semantic=None):
    return create_sync_block(sender, receiver, event_id, False, sender_pipe, receiver_pipe, _semantic)


@builtin
def sync_block_all(mode, event_id, _semantic=None):
    mode = _unwrap_if_constexpr(mode)
    event_id = _unwrap_if_constexpr(event_id)
    assert isinstance(mode, str), f"mode: {mode} is not string"
    assert isinstance(event_id, int) and 0 <= event_id < 16, f"event_id: {event_id} should be 0 ~ 15"
    assert mode in ("all_cube", "all_vector", "all"), f"ERROR: mode = {mode}"
    semantic.custom_sync_op(_semantic.builder, "sync_block_all", mode=mode, event_id=event_id)


@builtin
def alloc(shape, value, dtype, layout=None, scope=None, _semantic=None):
    """
    Returns a tensor filled with the scalar value for the given shape and dtype.
    """
    shape = _shape_check_impl(shape)
    value = _constexpr_to_value(value)
    dtype = _constexpr_to_value(dtype)
    layout = _constexpr_to_value(layout)
    scope = _constexpr_to_value(scope)
    return semantic.alloc(shape, value, dtype, layout, scope, _semantic.builder)


class SyncFlagType:
    ASCEND = ["cube_to_vector", "vector_to_cube"]

    def __init__(self, name):
        name = _unwrap_if_constexpr(name)
        self.name = name
        assert name in SyncFlagType.ASCEND, name

    def __str__(self):
        return self.name

    def codegen_name(self):
        return self.name

    def sender(self):
        if self.name == "cube_to_vector":
            return "cube"
        if self.name == "vector_to_cube":
            return "vector"
        assert self.name in SyncFlagType.ASCEND

    @property
    def cache_key_part(self) -> str:
        return self.name

    def __repr__(self):
        return f"triton.language.{self.codegen_name()}"


class SyncFlag:
    C2V = SyncFlagType("cube_to_vector")
    V2C = SyncFlagType("vector_to_cube")


def _get_cross_flag_pipes(sender):
    if sender == "cube":
        return "vector", PIPE.PIPE_FIX, PIPE.PIPE_MTE2
    if sender == "vector":
        return "cube", PIPE.PIPE_MTE3, PIPE.PIPE_MTE2
    raise AssertionError(f"Unexpected sender: {sender}")


@builtin
def set_cross_flag(sync_flag_type: SyncFlagType, event_id: int, _semantic=None):
    sender = _unwrap_if_constexpr(sync_flag_type.sender())
    event_id = _unwrap_if_constexpr(event_id)
    assert isinstance(event_id, int) and 0 <= event_id < 16, f"event_id: {event_id} should be 0 ~ 15"
    receiver, sender_pipe, receiver_pipe = _get_cross_flag_pipes(sender)
    return sync_block_set(
        sender, receiver, event_id, sender_pipe, receiver_pipe, _semantic=_semantic
    )


@builtin
def wait_cross_flag(sync_flag_type: SyncFlagType, event_id: int, _semantic=None):
    sender = _unwrap_if_constexpr(sync_flag_type.sender())
    event_id = _unwrap_if_constexpr(event_id)
    assert isinstance(event_id, int) and 0 <= event_id < 16, f"event_id: {event_id} should be 0 ~ 15"
    receiver, sender_pipe, receiver_pipe = _get_cross_flag_pipes(sender)
    return sync_block_wait(
        sender, receiver, event_id, sender_pipe, receiver_pipe, _semantic=_semantic
    )


@builtin
def fixpipe(
    src,
    dst=None,
    dma_mode: FixpipeDMAMode = FixpipeDMAMode.NZ2ND,
    dual_dst_mode: FixpipeDualDstMode = FixpipeDualDstMode.NO_DUAL,
    _semantic=None,
):
    pre_quant_mode = FixpipePreQuantMode.NO_QUANT
    pre_relu_mode = FixpipePreReluMode.NO_RELU
    return semantic.fixpipe(
        src, dst, dma_mode, dual_dst_mode, pre_quant_mode, pre_relu_mode, _semantic
    )


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


@builtin
def debug_barrier(sync_mode: SYNC_IN_VF, _semantic=None):
    return semantic.debug_barrier(sync_mode.name, _semantic)


@builtin
def sub_vec_num(_semantic=None) -> tl.constexpr:
    npuUtils = NPUUtils()
    cube_num = npuUtils.get_aivector_core_num()
    vector_num = npuUtils.get_aicore_num()
    const_val = cube_num // vector_num
    return tl.constexpr(const_val)
