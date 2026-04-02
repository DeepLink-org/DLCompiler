"""Triton vector addition kernel test on CPU verify mode."""

import os

# NOTE: Must set BEFORE importing triton, as triton reads this during import
os.environ.setdefault("DLC_CPU_VERIFY", "1")
os.environ.setdefault(
    "LLVM_BINARY_DIR",
    "/mnt/data01/kezengxiang/work/third_party/llvm-project/build_064f02dac0c81c19350a74415b3245f42fed09dc/bin",
)

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def test_vec_add():
    size = 1024
    x = torch.rand(size)
    y = torch.rand(size)

    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, size, BLOCK_SIZE=BLOCK_SIZE)

    assert torch.allclose(x + y, output)
