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
from .task_context import TaskBaseInfo, Scoreboard


@triton.jit
def add_task_compute(
    task_base_info: TaskBaseInfo,
    scoreboard: Scoreboard,
    BLOCK_SIZE: tl.constexpr,
):
    lhs_tensor = task_base_info.get_tensor(0)
    rhs_tensor = task_base_info.get_tensor(1)
    out_tensor = task_base_info.get_tensor(2)
    lhs_ptr = lhs_tensor.data_ptr(tl.bfloat16)
    rhs_ptr = rhs_tensor.data_ptr(tl.bfloat16)
    out_ptr = out_tensor.data_ptr(tl.bfloat16)

    n_elements = out_tensor.size(0)
    block_start = task_base_info.tile_id_or_start * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(lhs_ptr + offsets, mask=mask)
    y = tl.load(rhs_ptr + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)
    scoreboard.release_tile(task_base_info, task_base_info.tile_id_or_start)
