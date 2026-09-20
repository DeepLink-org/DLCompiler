"""Reduction and softmax, with no torch_txda arithmetic in the oracle."""
import pytest
import torch
import torch_txda  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def row_kernel(X, Y, N: tl.constexpr, SOFTMAX: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    x = tl.load(X + row * N + i, i < N, other=0).to(tl.float32)
    if SOFTMAX:
        x = tl.where(i < N, x, float('-inf'))
        exp = tl.exp(x - tl.max(x, 0))
        value = exp / tl.sum(exp, 0)
        tl.store(Y + row * N + i, value, i < N)
    else:
        tl.store(Y + row, tl.sum(x, 0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("columns", [32, 127, 256])
@pytest.mark.parametrize("softmax", [False, True])
def test_row_reduce(dtype, columns, softmax):
    host = (torch.arange(4 * columns).reshape(4, columns) % 17 - 8).to(dtype) / 8
    expected = host.float().softmax(1).to(dtype) if softmax else host.float().sum(1)
    x = (host).to("txda")
    y = (torch.zeros_like(expected)).to("txda")
    row_kernel[(4,)](x, y, columns, softmax, triton.next_power_of_2(columns))
    tolerance = 1e-4 if dtype == torch.float32 else 1e-3
    torch.testing.assert_close(y.cpu(), expected, rtol=tolerance, atol=tolerance)

