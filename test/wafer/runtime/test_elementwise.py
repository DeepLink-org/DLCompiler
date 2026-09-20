"""Add/mul with CPU references, matching the Ascend comparison tolerances."""
import pytest
import torch
import torch_txda  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def binary_kernel(X, Y, Z, N: tl.constexpr, MUL: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x, y = tl.load(X + i, i < N, 0), tl.load(Y + i, i < N, 0)
    z = x * y if MUL else x + y
    tl.store(Z + i, z, i < N)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int32])
@pytest.mark.parametrize("size", [1, 31, 256, 257])
@pytest.mark.parametrize("multiply", [False, True])
def test_binary(dtype, size, multiply):
    a = (torch.arange(size) % 17 - 8).to(dtype)
    b = (torch.arange(size) % 7 - 3).to(dtype)
    x, y = (a).to("txda"), (b).to("txda")
    z = (torch.zeros_like(a)).to("txda")
    binary_kernel[(triton.cdiv(size, 256),)](x, y, z, size, multiply, 256)
    expected = a * b if multiply else a + b
    # These bounded integer-valued inputs are exactly representable in all dtypes.
    torch.testing.assert_close(z.cpu(), expected, rtol=0, atol=0)

