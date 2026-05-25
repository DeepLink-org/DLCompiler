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
    constexpr,
    tensor,
)

import triton.language.core as tl


__all__ = ["get_element", "sort", "flip"]


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
