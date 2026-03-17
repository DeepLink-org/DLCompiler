import pytest
import torch
import tilelang
import tilelang.language as T

dtype = "float32"


def test_vec_sqrt():
    """测试：向量 sqrt"""
    N = 1024

    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(1, 1) as (tid, _):
            for i in T.Parallel(N):
                C[i] = T.sqrt(A[i])

    a = torch.abs(torch.randn(N, dtype=torch.float32)).npu() + 0.1  # 确保正数
    c = torch.zeros(N, dtype=torch.float32).npu()

    compiled = tilelang.compile(kernel)
    compiled(a, c)

    expected = torch.sqrt(a)
    max_diff = torch.max(torch.abs(c - expected))
    print(f"sqrt: max_diff = {max_diff}")
    torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-4)
    print("✓ test_vec_sqrt passed")


def test_vec_exp():
    """测试：向量 exp"""
    N = 512

    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(1, 1) as (tid, _):
            for i in T.Parallel(N):
                C[i] = T.exp(A[i])

    a = torch.randn(N, dtype=torch.float32).npu() * 2  # 限制范围避免溢出
    c = torch.zeros(N, dtype=torch.float32).npu()

    compiled = tilelang.compile(kernel)
    compiled(a, c)

    expected = torch.exp(a)
    torch.testing.assert_close(c, expected, atol=1e-4, rtol=1e-3)
    print("✓ test_vec_exp passed")


def test_vec_tanh():
    """测试：向量 tanh"""
    N = 256

    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(1, 1) as (tid, _):
            for i in T.Parallel(N):
                C[i] = T.tanh(A[i])

    a = torch.randn(N, dtype=torch.float32).npu()
    c = torch.zeros(N, dtype=torch.float32).npu()

    compiled = tilelang.compile(kernel)
    compiled(a, c)

    expected = torch.tanh(a)
    torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-4)
    print("✓ test_vec_tanh passed")


def test_vec_abs():
    """测试：向量 abs"""
    N = 512

    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(1, 1) as (tid, _):
            for i in T.Parallel(N):
                C[i] = T.abs(A[i])

    a = torch.randn(N, dtype=torch.float32).npu()
    c = torch.zeros(N, dtype=torch.float32).npu()

    compiled = tilelang.compile(kernel)
    compiled(a, c)

    expected = torch.abs(a)
    torch.testing.assert_close(c, expected, atol=1e-6, rtol=1e-5)
    print("✓ test_vec_abs passed")


def test_combined_unary():
    """测试：组合一元操作 exp(sqrt(abs(x)))"""
    N = 256

    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(1, 1) as (tid, _):
            for i in T.Parallel(N):
                temp = T.abs(A[i])
                temp2 = T.sqrt(temp)
                C[i] = T.exp(temp2)

    a = torch.randn(N, dtype=torch.float32).npu()
    c = torch.zeros(N, dtype=torch.float32).npu()

    compiled = tilelang.compile(kernel)
    compiled(a, c)

    expected = torch.exp(torch.sqrt(torch.abs(a)))
    torch.testing.assert_close(c, expected, atol=1e-3, rtol=1e-2)
    print("✓ test_combined_unary passed")


if __name__ == "__main__":
    print("Testing Unary Operations Vectorization")
    print("=" * 60)
    test_vec_sqrt()
    test_vec_exp()
    test_vec_tanh()
    test_vec_abs()
    test_combined_unary()
    print("=" * 60)
    print("All unary operation tests passed! ✓")
