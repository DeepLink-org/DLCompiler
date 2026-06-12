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

from typing import List
from contextlib import contextmanager
from triton.language import core as tl
from triton.language import semantic as tl_semantic
from triton._C.libtriton import ir, dicp_triton

_dry_run = False


@contextmanager
def dry_run_context():
    global _dry_run
    _dry_run = True
    try:
        yield
    finally:
        _dry_run = False


_SENDER_RECEIVER_MAP = {
    "cube": ("vector", dicp_triton.ir.PIPE.PIPE_FIX, dicp_triton.ir.PIPE.PIPE_MTE2),
    "vector": ("cube", dicp_triton.ir.PIPE.PIPE_MTE3, dicp_triton.ir.PIPE.PIPE_MTE2),
}


def create_address_space(
    address_space: dicp_triton.ir.AddressSpace,
    builder,
) -> ir.attribute:
    return builder.get_target_attribute(address_space)


class PIPE:
    PIPE_S = dicp_triton.ir.PIPE.PIPE_S
    PIPE_V = dicp_triton.ir.PIPE.PIPE_V
    PIPE_M = dicp_triton.ir.PIPE.PIPE_M
    PIPE_MTE1 = dicp_triton.ir.PIPE.PIPE_MTE1
    PIPE_MTE2 = dicp_triton.ir.PIPE.PIPE_MTE2
    PIPE_MTE3 = dicp_triton.ir.PIPE.PIPE_MTE3
    PIPE_ALL = dicp_triton.ir.PIPE.PIPE_ALL
    PIPE_FIX = dicp_triton.ir.PIPE.PIPE_FIX


def insert_slice(
    ful: tl.tensor,
    sub: tl.tensor,
    offsets: List[tl.tensor],
    sizes: List[int],
    strides: List[int],
    _semantic: tl_semantic.TritonSemantic,
) -> tl.tensor:
    assert len(ful.shape) == len(offsets)
    assert len(ful.shape) == len(sizes)
    assert len(ful.shape) == len(strides)
    assert all([s >= 1 for s in sizes])
    assert all([s >= 0 for s in strides])
    new_offsets = [o.handle for o in offsets]
    ret_type = tl.block_type(ful.type.scalar, ful.shape)
    out = _semantic.builder.create_insert_slice(
        ful.handle, sub.handle, new_offsets, sizes, strides
    )
    return tl.tensor(out, ret_type)


def extract_slice(
    ful: tl.tensor,
    offsets: List[tl.tensor],
    sizes: List[int],
    strides: List[int],
    _semantic: tl_semantic.TritonSemantic,
) -> tl.tensor:
    assert len(ful.shape) == len(offsets)
    assert len(ful.shape) == len(sizes)
    assert len(ful.shape) == len(strides)
    assert all([s >= 1 for s in sizes])
    assert all([s >= 0 for s in strides])
    new_offsets = [o.handle for o in offsets]
    ret_type = tl.block_type(ful.type.scalar, sizes)
    out = _semantic.builder.create_extract_slice(
        ful.handle, new_offsets, sizes, strides
    )
    return tl.tensor(out, ret_type)


def compile_hint(ptr: tl.tensor, hint_name: str, hint_val, builder: ir.builder):
    if isinstance(hint_val, bool):
        hint_val = builder.get_bool_attr(hint_val)
    elif not hint_val:
        hint_val = builder.get_unit_attr()
    elif isinstance(hint_val, int):
        hint_val = builder.get_int32_attr(hint_val)
    elif isinstance(hint_val, tl.constexpr):
        hint_val = builder.get_string_attr(hint_val.value)
    elif isinstance(hint_val, (list, tl.tuple)):
        hint_val = builder.get_i64_array_attr(hint_val)
    else:
        raise ValueError(f"Unsupported hint value type: {type(hint_val)}")
    builder.create_annotation_mark(ptr.handle, hint_name, hint_val)


def alloc(
    shape: List[int], value, dtype: tl.dtype, layout, scope, builder: ir.builder
) -> tl.tensor:
    if isinstance(value, tl.tensor):
        assert value.numel.value == 1, "only accepts size-1 tensor"
        value = tl_semantic.cast(value, dtype, builder)
    else:
        if dtype is None:
            raise ValueError("dtype must be specified when value is not a tensor")
        if value == 0:
            value = builder.get_null_value(dtype.to_ir(builder))
        else:
            get_value_fn = getattr(builder, f"get_{dtype.name}")
            value = get_value_fn(value)
        value = tl.tensor(value, dtype)
    if len(shape) == 0:
        return value
    ret_ty = tl.block_type(value.dtype, shape)
    x = tl.tensor(builder.create_splat(value.handle, shape), ret_ty)
    if layout is not None:
        builder.create_annotation_mark(
            x.handle, "layout", builder.get_string_attr(str(layout))
        )
    if scope is not None:
        builder.create_annotation_mark(
            x.handle, "scope", builder.get_string_attr(str(scope))
        )
    return x


def custom_sync_op(builder: ir.builder, op_name: str, **kwargs):
    if _dry_run:
        return None
    if op_name == "sync_block_all":
        return builder.sync_block_all(kwargs["mode"], kwargs["event_id"])
    elif op_name == "sync_block_set":
        sender = kwargs["sender"]
        receiver, sender_pipe, receiver_pipe = _SENDER_RECEIVER_MAP[sender]
        event_id = kwargs["event_id"]
        id_value = builder.get_int64(event_id)
        return builder.sync_block_set(
            sender, receiver, id_value, sender_pipe, receiver_pipe
        )
    elif op_name == "sync_block_wait":
        sender = kwargs["sender"]
        receiver, sender_pipe, receiver_pipe = _SENDER_RECEIVER_MAP[sender]
        event_id = kwargs["event_id"]
        id_value = builder.get_int64(event_id)
        return builder.sync_block_wait(
            sender, receiver, id_value, sender_pipe, receiver_pipe
        )
    raise ValueError(f"Unsupported custom op: {op_name}")


def create_sync_block_set(
    sender, receiver, event_id, sender_pipe, receiver_pipe, _semantic=None
):
    if isinstance(event_id, int):
        _semantic.builder.sync_block_set(
            sender,
            receiver,
            _semantic.to_tensor(tl.constexpr(event_id)).handle,
            sender_pipe.value,
            receiver_pipe.value,
        )
    elif isinstance(event_id, tl.constexpr):
        _semantic.builder.sync_block_set(
            sender,
            receiver,
            _semantic.to_tensor(event_id).handle,
            sender_pipe.value,
            receiver_pipe.value,
        )
    else:
        _semantic.builder.sync_block_set(
            sender, receiver, event_id.handle, sender_pipe.value, receiver_pipe.value
        )


def create_sync_block_wait(
    sender, receiver, event_id, sender_pipe, receiver_pipe, _semantic=None
):
    if isinstance(event_id, int):
        _semantic.builder.sync_block_wait(
            sender,
            receiver,
            _semantic.to_tensor(tl.constexpr(event_id)).handle,
            sender_pipe.value,
            receiver_pipe.value,
        )
    elif isinstance(event_id, tl.constexpr):
        _semantic.builder.sync_block_wait(
            sender,
            receiver,
            _semantic.to_tensor(event_id).handle,
            sender_pipe.value,
            receiver_pipe.value,
        )
    else:
        _semantic.builder.sync_block_wait(
            sender, receiver, event_id.handle, sender_pipe.value, receiver_pipe.value
        )


def sub_vec_id(_semantic=None):
    return tl.tensor(_semantic.builder.create_get_sub_vec_id(), tl.int64)


def copy_from_ub_to_l1(src, dst, _semantic=None):
    from ..buffer.core import buffer as bl_buffer
    from . import core as _core

    if isinstance(src, tl.tensor) or isinstance(dst, tl.tensor):
        raise TypeError("tensor not support yet")
    if src.shape != dst.shape:
        raise TypeError("src and dst must have same shape")
    if src.dtype != dst.dtype:
        raise TypeError("src and dst need to have the same type")
    if isinstance(src, bl_buffer) and isinstance(dst, bl_buffer):
        if src.space != _core.ascend_address_space.UB:
            raise TypeError("src's AddressSpace must be UB")
        if dst.space != _core.ascend_address_space.L1:
            raise TypeError("dst's AddressSpace must be L1")
        _semantic.builder.create_copy_buffer(src.handle, dst.handle)
    else:
        raise TypeError("src and dst must be tl.tensor or bl.buffer")


def copy(src, dst, _semantic=None):
    from ..buffer.core import buffer as bl_buffer
    from . import core as _core

    if isinstance(src, tl.tensor) or isinstance(dst, tl.tensor):
        raise TypeError("tensor not support yet")
    if src.shape != dst.shape:
        raise TypeError("src and dst must have same shape")
    if src.dtype != dst.dtype:
        raise TypeError("src and dst need to have the same type")
    if isinstance(src, bl_buffer) and isinstance(dst, bl_buffer):
        if src.space != _core.ascend_address_space.UB:
            raise TypeError("src's AddressSpace must be UB")
        if dst.space not in (
            _core.ascend_address_space.L1,
            _core.ascend_address_space.UB,
        ):
            raise TypeError("dst's AddressSpace must be UB or L1")
        _semantic.builder.create_copy_buffer(src.handle, dst.handle)
    else:
        raise TypeError("src and dst must be tl.tensor or bl.buffer")


def fixpipe(
    src,
    dst,
    dma_mode,
    dual_dst_mode,
    pre_quant_mode,
    pre_relu_mode,
    _semantic=None,
):
    if dst is None:
        result = _semantic.builder.create_fixpipe(
            src.handle,
            None,
            dma_mode.value,
            dual_dst_mode.value,
            pre_quant_mode.value,
            pre_relu_mode.value,
        )
        if dual_dst_mode.value.name == "ROW_SPLIT":
            new_shape = list(src.type.shape)
            if len(new_shape) >= 1 and new_shape[0] > 0:
                new_shape[0] = new_shape[0] // 2
            new_type = tl.block_type(src.type.element_ty, new_shape)
            return tl.tensor(result, new_type)
        elif dual_dst_mode.value.name == "COLUMN_SPLIT":
            new_shape = list(src.type.shape)
            if len(new_shape) >= 2 and new_shape[1] > 0:
                new_shape[1] = new_shape[1] // 2
            new_type = tl.block_type(src.type.element_ty, new_shape)
            return tl.tensor(result, new_type)
        else:
            return tl.tensor(result, src.type)
    else:
        _semantic.builder.create_fixpipe(
            src.handle,
            dst.handle,
            dma_mode.value,
            dual_dst_mode.value,
            pre_quant_mode.value,
            pre_relu_mode.value,
        )


def debug_barrier(sync_mode: str, _semantic=None):
    target = tl.tensor(_semantic.builder.get_int64(0), tl.int64)
    attr = _semantic.builder.get_string_attr(sync_mode)
    _semantic.builder.create_debug_barrier(target.handle, "SYNC_IN_VF", attr)
