################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
import triton
import triton.language as tl
from triton.language import core


@core.extern
def prefetch_async(ptr, nbytes, _semantic=None):
    return tl.inline_asm_elementwise(
        asm="""
        cp.async.bulk.prefetch.L2.global [$1], $2;
        mov.u32 $0, 0;
        """,
        constraints=("=r,l,r"),  # no use output
        args=[ptr, nbytes],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@triton.jit
def prefetch_task_compute(task_base_info, scoreboard):
    weight_tensor = task_base_info.get_tensor(0)
    weight_ptr = weight_tensor.data_ptr(tl.bfloat16)
    M = weight_tensor.size(0)
    N = weight_tensor.size(1, 32)

    elem_size = tl.constexpr(weight_ptr.dtype.element_ty.primitive_bitwidth) // 8
    nbytes = elem_size * M * N
    if task_base_info.tile_id_or_start == 0:
        prefetch_async(weight_ptr, nbytes)
