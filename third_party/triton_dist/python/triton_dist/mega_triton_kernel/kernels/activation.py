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
def act_mul_up_tile_compute(tile_id, input, output, M, N, ACT_FN, BLOCK_SIZE_M: tl.constexpr,
                            BLOCK_SIZE_N: tl.constexpr):
    tl.static_assert(ACT_FN == tl.constexpr("silu"))

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.where(offs_m < M, offs_m, 0)
    offs_n_gate = tl.where(offs_n < N, offs_n, 0)
    offs_n_up = tl.where(offs_n < N, offs_n, 0) + N
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_n_gate = tl.max_contiguous(tl.multiple_of(offs_n_gate, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_n_up = tl.max_contiguous(tl.multiple_of(offs_n_up, BLOCK_SIZE_N), BLOCK_SIZE_N)
    gate_ptrs = input + (offs_m[:, None] * N * 2 + offs_n_gate[None, :])
    up_ptrs = input + (offs_m[:, None] * N * 2 + offs_n_up[None, :])
    gate = tl.load(gate_ptrs)
    up = tl.load(up_ptrs)
    if ACT_FN == tl.constexpr("silu"):
        gate = gate.to(tl.float32)
        gate = gate * (1.0 / (1.0 + tl.exp((-gate))))
        gate = gate.to(gate_ptrs.dtype.element_ty)
    ret = gate * up
    ret = ret.to(output.dtype.element_ty)
    out_ptrs = output + (offs_m[:, None] * N + offs_n_gate[None, :])
    tl.store(out_ptrs, ret)


@triton.jit
def silu_mul_up_task_compute(task_base_info: TaskBaseInfo, scoreboard: Scoreboard, BLOCK_SIZE_M: tl.constexpr,
                             BLOCK_SIZE_N: tl.constexpr):
    # scoreboard.wait_deps(task_base_info)

    input = task_base_info.get_tensor(0)
    output = task_base_info.get_tensor(1)

    M = output.size(0, 16)
    N = output.size(1, 16)
    ACT_FN: tl.constexpr = tl.constexpr("silu")
    a_ptr = input.data_ptr(tl.bfloat16)
    b_ptr = output.data_ptr(tl.bfloat16)

    act_mul_up_tile_compute(task_base_info.tile_id_or_start, a_ptr, b_ptr, M, N, ACT_FN, BLOCK_SIZE_M, BLOCK_SIZE_N)
    scoreboard.release_tile(task_base_info, task_base_info.tile_id_or_start)
