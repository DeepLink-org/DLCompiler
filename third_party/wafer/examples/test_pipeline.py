"""Exercise software pipelining across full tiles, tails and short loops."""
import pytest
import torch
import torch_txda  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def pipelined_gemm(A, B, C, K: tl.constexpr, BK: tl.constexpr):
    m = tl.arange(0, 32)
    n = tl.arange(0, 32)
    k = tl.arange(0, BK)
    acc = tl.full((32, 32), 0, tl.float32)
    for block in tl.range(0, tl.cdiv(K, BK), num_stages=2):
        offsets = block * BK + k
        a = tl.load(A + m[:, None] * K + offsets[None, :], offsets[None, :] < K, other=0)
        b = tl.load(B + offsets[:, None] * 32 + n[None, :], offsets[:, None] < K, other=0)
        acc += tl.dot(a, b)
    tl.store(C + m[:, None] * 32 + n[None, :], acc)


@pytest.mark.parametrize("k", [16, 32, 33, 64, 96, 128])
def test_pipeline_gemm(device, k):
    a = torch.randn((32, k), dtype=torch.float16, device="cpu")
    b = torch.randn((k, 32), dtype=torch.float16, device="cpu")
    expected = a.float() @ b.float()
    for enabled in (False, True):
        output = torch.empty((32, 32), dtype=torch.float32, device="cpu")
        a_txda = a.to("txda")
        b_txda = b.to("txda")
        output_txda = output.to("txda")
        pipelined_gemm[(1,)](a_txda, b_txda, output_txda, k, 32, num_stages=2, enable_pipeline=enabled)
        with torch.no_grad():
            output.copy_(output_txda.cpu())
        torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)
