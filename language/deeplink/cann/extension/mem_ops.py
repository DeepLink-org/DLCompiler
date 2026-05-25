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

from triton.language.core import (
    _unwrap_if_constexpr,
    builtin,
    tensor,
)

import triton.language.core as tl


__all__ = [
    "index_select_simd",
    "gather_out_to_ub",
    "scatter_ub_to_out",
    "index_put",
]


@builtin
def index_select_simd(
    src, dim, index, src_shape, src_offset, read_shape, _semantic=None
):
    dim = _unwrap_if_constexpr(dim)
    newsrc_shape = [
        _semantic.to_tensor(o) if isinstance(o, tl.constexpr) else o for o in src_shape
    ]
    newsrc_offset = [
        _semantic.to_tensor(o) if isinstance(o, tl.constexpr) else o for o in src_offset
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
