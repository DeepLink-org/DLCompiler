import torch
import tilelang
import tilelang.language as T
import pytest
import triton
import triton.language as tl


@pytest.mark.skip("todo::zmz will remove this after fix pass")
def test_if_then_else_1d():
    N = 128
    block = 128
    dtype = "float32"

    def if_then_else(N, block_N, dtype="float32"):
        @T.prim_func
        def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype)):
            with T.Kernel(1, 1) as (bx, by):
                A_shared = T.alloc_shared((block_N,), dtype)
                B_shared = T.alloc_shared((block_N,), dtype)
                T.copy(A, A_shared)
                for i in T.Parallel(block_N):
                    B_shared[i] = T.if_then_else(A_shared[i] > 0.0, A_shared[i], 0.0)
                T.copy(B_shared, B)

        return main

    a = torch.randn(N, dtype=torch.float32).npu()
    b = torch.zeros(N, dtype=torch.float32).npu()
    func = if_then_else(N, block, dtype)
    mod = tilelang.compile(func)
    mod(a, b)

    expected = torch.where(a > 0, a, torch.zeros_like(a))
    torch.testing.assert_close(b, expected, atol=1e-5, rtol=1e-5)
    print("T.if_then_else test passed")


@pytest.mark.skip("todo::zmz will remove this after fix pass")
def test_if_then_else_2d():
    M, N = 128, 128
    BM, BN = 64, 64
    dtype = "float32"

    def if_then_else(M, BM, N, BN, dtype="float32"):
        @T.prim_func
        def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
            with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN)) as (bx, by):
                A_shared = T.alloc_shared((BM, BN), dtype)
                B_shared = T.alloc_shared((BM, BN), dtype)
                T.copy(A[bx * BM : (bx + 1) * BM, by * BN : (by + 1) * BN], A_shared)
                for i, j in T.Parallel(BM, BN):
                    B_shared[i, j] = T.if_then_else(
                        A_shared[i, j] > 0.0, A_shared[i, j], 0.0
                    )

                T.copy(B_shared, B[bx * BM : (bx + 1) * BM, by * BN : (by + 1) * BN])

        return main

    a = torch.randn((M, N), dtype=torch.float32).npu()
    b = torch.zeros((M, N), dtype=torch.float32).npu()
    func = if_then_else(M, BM, N, BN, dtype)
    mod = tilelang.compile(func)
    mod(a, b)

    expected = torch.where(a > 0, a, torch.zeros_like(a))
    torch.testing.assert_close(b, expected, atol=1e-5, rtol=1e-5)
    print("T.if_then_else test passed")


@triton.jit
def if_then_else_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.where(x > 0, x, 0)
    tl.store(y_ptr + offsets, y, mask=mask)


def test_triton_if_then_else():
    N = 128
    block = 128
    a = torch.randn(N, dtype=torch.float32).npu()
    b = torch.zeros(N, dtype=torch.float32).npu()
    n_elements = a.numel()
    grid = (triton.cdiv(n_elements, block),)
    if_then_else_kernel[grid](a, b, n_elements, BLOCK=block)
    expected = torch.where(a > 0, a, torch.zeros_like(a))
    torch.testing.assert_close(b, expected, atol=1e-5, rtol=1e-5)
