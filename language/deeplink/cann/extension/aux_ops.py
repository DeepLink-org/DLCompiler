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

from triton.language.core import _unwrap_if_constexpr, range, tensor
import triton.language.core as tl_core

from .core import builtin
from . import semantic

__all__ = ["parallel", "compile_hint", "multibuffer"]


class parallel(range):
    """
    Iterator that counts upward with parallel execution semantics.
    """

    def __init__(
        self,
        arg1,
        arg2=None,
        step=None,
        num_stages=None,
        loop_unroll_factor=None,
        bind_sub_block: bool = False,
    ):
        super().__init__(arg1, arg2, step, num_stages, loop_unroll_factor)
        self.bind_sub_block = bind_sub_block


@builtin
def compile_hint(ptr, hint_name, hint_val=None, _semantic=None):
    def _unwrap(val):
        return _unwrap_if_constexpr(val) if val else val

    hint_name = _unwrap_if_constexpr(hint_name)
    assert isinstance(hint_name, str), f"hint name: {hint_name} is not string"
    if isinstance(hint_val, (list, tl_core.tuple)):
        hint_val = [_unwrap(val) for val in hint_val]
    else:
        hint_val = _unwrap(hint_val)
    hint_val = _unwrap_if_constexpr(hint_val) if hint_val else hint_val
    semantic.compile_hint(ptr, hint_name, hint_val, _semantic.builder)


@builtin
def multibuffer(src: tensor, size, _semantic=None):
    """
    Set multi_buffer for an existing tensor.
    """
    buffer_size = _unwrap_if_constexpr(size)
    assert (
        isinstance(buffer_size, int) and buffer_size == 2
    ), "only support bufferize equals 2"
    semantic.compile_hint(src, "multi_buffer", buffer_size, _semantic.builder)
