"""Native TXDA allocation, masked GEMM, FP32 accumulation and CPU comparison."""
import pytest
import torch
import torch_txda  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.program_id(0) * 16 + tl.arange(0, 16)
    n = tl.program_id(1) * 16 + tl.arange(0, 16)
    k = tl.arange(0, 16)
    acc = tl.full((16, 16), 0, tl.float32)
    for start in range(tl.cdiv(K, 16)):
        kk = start * 16 + k
        a = tl.load(A + m[:, None] * K + kk[None, :], (m[:, None] < M) & (kk[None, :] < K), 0)
        b = tl.load(B + kk[:, None] * N + n[None, :], (kk[:, None] < K) & (n[None, :] < N), 0)
        acc = tl.dot(a, b, acc)
    tl.store(C + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(16, 16, 16), (32, 64, 32), (17, 19, 33)])
def test_native_matmul(dtype, shape):
    m, n, k = shape
    a = (torch.arange(m * k).reshape(m, k) % 9 - 4).to(dtype) / 8
    b = (torch.arange(k * n).reshape(k, n) % 7 - 3).to(dtype) / 8
    x, y = (a).to("txda"), (b).to("txda")
    z = (torch.zeros((m, n), dtype=dtype)).to("txda")
    matmul_kernel[(triton.cdiv(m, 16), triton.cdiv(n, 16))](x, y, z, m, n, k)
    torch.testing.assert_close(z.cpu(), (a.float() @ b.float()).to(dtype), rtol=1e-3, atol=1e-3)

