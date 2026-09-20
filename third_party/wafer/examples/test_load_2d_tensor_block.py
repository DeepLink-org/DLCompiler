import torch
import torch_txda  # noqa: F401

import triton
import triton.language as tl
"""

|-----|-----|-----|-----|
|     |     |     |     |
|-----|-----|-----|-----|
|     |     |     |     |
|-----|-----|-----|-----|

Each instance loads BLOCK_SIZE_ROW * BLOCK_SIZE_COL
"""


@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    stride_0,
    stride_1,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    input_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=[n_rows, n_cols],
        strides=[stride_0, stride_1],
        offsets=[pid0 * BLOCK_SIZE_ROW, pid1 * BLOCK_SIZE_COL],
        block_shape=[BLOCK_SIZE_ROW, BLOCK_SIZE_COL],
        order=[1, 0],
    )
    x = tl.load(input_ptr)
    x = (2 * x) + 1
    output_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=[n_rows, n_cols],
        strides=[stride_0, stride_1],
        offsets=[pid0 * BLOCK_SIZE_ROW, pid1 * BLOCK_SIZE_COL],
        block_shape=[BLOCK_SIZE_ROW, BLOCK_SIZE_COL],
        order=[1, 0],
    )
    tl.store(output_ptr, x)


def test(device):
    n_rows = 512
    n_cols = 256
    x = torch.arange(0, n_rows * n_cols, 1, device="cpu", dtype=torch.float32).reshape([n_rows, n_cols])
    output = torch.full([n_rows, n_cols], -1, device="cpu", dtype=x.dtype)
    BLOCK_SIZE_ROW = 4
    BLOCK_SIZE_COL = 2

    grid = lambda meta: (n_rows // BLOCK_SIZE_ROW, n_cols // BLOCK_SIZE_COL)

    x_txda = x.to("txda")
    output_txda = output.to("txda")
    kernel[grid](
        x_txda,
        output_txda,
        n_rows,
        n_cols,
        x_txda.stride(0),
        x_txda.stride(1),
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
    )
    with torch.no_grad():
        output.copy_(output_txda.cpu())
    expected = (2 * x) + 1

    torch.testing.assert_close(output, expected, rtol=0.001, atol=1e-5)
