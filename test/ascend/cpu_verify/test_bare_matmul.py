"""Triton bare matmul kernel test on CPU verify mode."""

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


@triton.jit
def bare_matmul(X, Y, Z, M, N, K, BLOCK_SIZE: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    offs_x = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_y = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offs_x[:, None] * K + offs_y[None, :])
    y = tl.load(Y + offs_x[:, None] * N + offs_y[None, :])
    z = tl.dot(x, y)
    tl.store(Z + offs_x[:, None] * N + offs_y[None, :], z)


def test_bare_matmul():
    n = 128
    a = torch.randn((n, n), dtype=torch.float32)
    b = torch.randn((n, n), dtype=torch.float32)
    c = torch.empty((n, n), dtype=torch.float32)

    bare_matmul[(1,)](a, b, c, n, n, n, BLOCK_SIZE=n)

    assert torch.allclose(torch.matmul(a, b), c, atol=1e-2, rtol=0)
