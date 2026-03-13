import torch
import tilelang
import tilelang.language as T
import pytest

@pytest.mark.skip("todu::zmz will remove this after fix pass")
def test_if_then_else_1d():
    N = 128
    block = 128
    dtype = "float32"
    
    def if_then_else(N, block_N, dtype="float32"):
        @T.prim_func
        def main(
            A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype)
        ):
            with T.Kernel(1, 1) as (bx, by):
                for i in T.Parallel(block_N):
                    # A[i] > 0 时输出 A[i]，否则输出 0（等价 ReLU）
                    B[i] = T.if_then_else(A[i] > 0.0, A[i], 0.0)

        return main

    a = torch.randn(N, dtype=torch.float32).npu()
    b = torch.zeros(N, dtype=torch.float32).npu()
    func = if_then_else(N, block, dtype)
    mod = tilelang.compile(func)
    mod(a, b)

    expected = torch.where(a > 0, a, torch.zeros_like(a))
    torch.testing.assert_close(b, expected, atol=1e-5, rtol=1e-5)
    print("T.if_then_else test passed")

@pytest.mark.skip("todu::zmz will remove this after fix pass")
def test_if_then_else_2d():
    M, N = 128, 128
    BM, BN = 64, 64
    dtype = "float32"
    
    def if_then_else(M, BM, N, BN, dtype="float32"):
        @T.prim_func
        def main(
            A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)
        ):
            with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN)) as (bx, by):
                for i, j in T.Parallel(BM, BN):
                    x = bx * BM + i
                    y = by * BN + j
                    B[x, y] = T.if_then_else(A[x, y] > 0.0, A[x, y], 0.0)
        return main

    a = torch.randn((M, N), dtype=torch.float32).npu()
    b = torch.zeros((M, N), dtype=torch.float32).npu()
    func = if_then_else(M, BM, N, BN, dtype)
    mod = tilelang.compile(func)
    mod(a, b)

    expected = torch.where(a > 0, a, torch.zeros_like(a))
    torch.testing.assert_close(b, expected, atol=1e-5, rtol=1e-5)
    print("T.if_then_else test passed")

