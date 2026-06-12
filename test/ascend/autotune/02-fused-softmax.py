import os

import torch
import torch_npu
import triton
import triton.language as tl
from backend.testing import do_bench_npu
import triton.backends.dicp_triton.ascend_autotune_hooks  # noqa: F401 — install proxy before @triton.autotune


@triton.autotune(
    configs=[],
    key=["n_rows", "n_cols"],
)
@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
):
    row_start = tl.program_id(0) * XBLOCK
    for row_idx in tl.range(0, XBLOCK, XBLOCK_SUB):
        row_offsets = row_start + row_idx + tl.arange(0, XBLOCK_SUB)[:, None]
        col_offsets = tl.arange(0, BLOCK_SIZE)[None, :]
        xmask = row_offsets < n_rows
        ymask = col_offsets < n_cols
        mask = xmask & ymask
        input_ptrs = input_ptr + (row_offsets * input_row_stride + col_offsets)
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row, axis=1).reshape(XBLOCK_SUB, 1).broadcast_to(
            XBLOCK_SUB, BLOCK_SIZE
        )
        numerator = tl.exp(row_minus_max)
        denominator = (
            tl.sum(numerator, axis=1)
            .reshape(XBLOCK_SUB, 1)
            .broadcast_to(XBLOCK_SUB, BLOCK_SIZE)
        )
        softmax_output = numerator / denominator
        output_ptrs = output_ptr + (row_offsets * output_row_stride + col_offsets)
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax_torch(x):
    return torch.softmax(x, axis=-1)


def softmax_autotune(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = n_cols
    y = torch.empty_like(x)
    softmax_kernel[lambda meta: (triton.cdiv(n_rows, meta["XBLOCK"]), 1, 1)](
        y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def test_softmax(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device="npu")
    y_torch = softmax_torch(x)
    y_triton = softmax_autotune(x)
    assert torch.allclose(y_triton, y_torch)
    print(f"Fused Softmax {shape} {dtype} PASSED!")


if __name__ == "__main__":
    test_softmax((16896, 1024), torch.float32)
