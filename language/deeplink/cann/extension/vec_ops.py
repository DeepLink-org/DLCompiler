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

import builtins

from triton.language.core import (
    _tensor_member_fn,
    _unwrap_if_constexpr,
    builtin,
    constexpr,
    slice,
    tensor,
)

import triton.language.core as tl
from . import semantic as dl_semantic


__all__ = ["insert_slice", "extract_slice", "get_element", "sort", "flip"]


def _constexpr_to_value(v):
    if isinstance(v, constexpr):
        return v.value
    return v


def _extract_slice(sl: slice, shape: constexpr):
    def constexpr_or_none_to_value(v, default: int):
        if v is None:
            return default
        assert isinstance(
            v, (constexpr, int)
        ), f"slice only can be constexpr or int, got: {v}"
        return _constexpr_to_value(v)

    start = constexpr_or_none_to_value(sl.start, 0)
    stop = constexpr_or_none_to_value(sl.stop, _constexpr_to_value(shape))
    step = constexpr_or_none_to_value(sl.step, 1)
    size = (stop - start + step - 1) // step
    assert (
        start >= 0 and stop >= 0 and step >= 0 and size >= 0
    ), "slice should be greater than 0"
    return start, size, step


@_tensor_member_fn
@builtin
def __getitem__(self, slices, _semantic=None):
    if isinstance(slices, (builtins.slice, slice, constexpr, tensor, int)) or slices is None:
        slices = [slices]
    if isinstance(slices, tuple):
        slices = slices.values
    ret = self
    offsets = []
    sizes = []
    strides = []
    dst_shape = []
    need_extract_slice = False
    for dim, sl in enumerate(slices):
        if sl is None or isinstance(sl, constexpr) and sl.value is None:
            ret = _semantic.expand_dims(ret, dim)
            offsets.append(_semantic.builder.get_int32(0))
            dst_shape.append(constexpr(1))
            sizes.append(constexpr(1))
            strides.append(constexpr(1))
        elif (
            isinstance(sl, slice)
            and sl.start is None
            and sl.stop is None
            and sl.step is None
        ):
            pass
        elif isinstance(sl, constexpr) and sl.value is not None:
            offsets.append(_semantic.builder.get_int32(_constexpr_to_value(sl)))
            need_extract_slice = True
            sizes.append(constexpr(1))
            strides.append(constexpr(1))
        elif isinstance(sl, int):
            offsets.append(_semantic.builder.get_int32(sl))
            need_extract_slice = True
            sizes.append(constexpr(1))
            strides.append(constexpr(1))
        elif isinstance(sl, tensor):
            offsets.append(sl.handle)
            sizes.append(constexpr(1))
            strides.append(constexpr(1))
            need_extract_slice = True
        elif isinstance(sl, (slice, builtins.slice)):
            start, size, step = _extract_slice(sl, ret.shape[dim])
            offsets.append(start)
            strides.append(constexpr(step))
            sizes.append(constexpr(size))
            dst_shape.append(constexpr(size))
            need_extract_slice = True
        else:
            raise ValueError(f"unsupported tensor index: {sl}")

    if need_extract_slice:
        new_offsets = [
            (_semantic.to_tensor(o) if not isinstance(o, tensor) else o)
            for o in offsets
        ]
        ret = dl_semantic.extract_slice(
            self, new_offsets, sizes, strides, _semantic=_semantic
        )
    return ret


@_tensor_member_fn
@builtin
def insert_slice(
    ful, sub, offsets, sizes, strides, _builder=None, _generator=None, _semantic=None
) -> tensor:
    """
    Insert a tensor to another tensor as specified by the operation's offsets, sizes and strides arguments.
    """
    assert len(ful.shape) > 0
    assert len(ful.shape) == len(sub.shape)
    new_offsets = [
        _semantic.to_tensor(o) if isinstance(o, constexpr) else o for o in offsets
    ]
    return dl_semantic.insert_slice(
        ful, sub, new_offsets, sizes, strides, _semantic=_semantic
    )


@_tensor_member_fn
@builtin
def extract_slice(
    ful, offsets, sizes, strides, _generator=None, _semantic=None
) -> tensor:
    """
    Extract a tensor from another tensor as specified by the operation's offsets, sizes and strides arguments.
    """
    assert len(ful.shape) > 0
    new_offsets = [
        _semantic.to_tensor(o) if isinstance(o, constexpr) else o for o in offsets
    ]
    return dl_semantic.extract_slice(
        ful, new_offsets, sizes, strides, _semantic=_semantic
    )


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


# flip is defined in libdevice; re-export here for alignment with triton-ascend
from ..libdevice import flip
