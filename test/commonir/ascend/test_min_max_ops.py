import pytest
import torch
import tilelang
import tilelang.language as T

dtype = "float32"


@pytest.mark.skip
def test_elementwise_max():
    """测试：Element-wise max(A, B)"""
    N = 1024

    @T.prim_func
    def kernel(
        A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)
    ):
        with T.Kernel(1, 1) as (tid, _):
            for i in T.Parallel(N):
                C[i] = T.max(A[i], B[i])

    a = torch.randn(N, dtype=torch.float32).npu()
    b = torch.randn(N, dtype=torch.float32).npu()
    c = torch.zeros(N, dtype=torch.float32).npu()

    compiled = tilelang.compile(kernel)
    compiled(a, b, c)

    expected = torch.maximum(a, b)
    max_diff = torch.max(torch.abs(c - expected))
    print(f"elementwise_max: max_diff = {max_diff.item():.8f}")
    print(f"  result sample: {c[:5]}")
    print(f"  expected sample: {expected[:5]}")
    torch.testing.assert_close(c, expected, atol=1e-6, rtol=1e-5)
    print("✓ test_elementwise_max passed")


@pytest.mark.skip("todo::zmz will remove this after fix pass")
def test_elementwise_max_tilelang():
    """测试：Element-wise max(A, B)"""
    N = 128

    @T.prim_func
    def kernel(
        A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)
    ):
        with T.Kernel(1, 1) as (bx, _):
            a_local = T.alloc_fragment([N], dtype)
            b_local = T.alloc_fragment([N], dtype)
            c_local = T.alloc_fragment([N], dtype)
            T.copy(A[bx * N : (bx + 1) * N], a_local)
            T.copy(B[bx * N : (bx + 1) * N], b_local)
            for i in T.Parallel(N):
                c_local[i] = T.max(a_local[i], b_local[i])
            T.copy(c_local, C[bx * N])

    a = torch.randn(N, dtype=torch.float32).npu()
    b = torch.randn(N, dtype=torch.float32).npu()
    c = torch.zeros(N, dtype=torch.float32).npu()

    compiled = tilelang.compile(kernel)
    compiled(a, b, c)

    expected = torch.maximum(a, b)
    max_diff = torch.max(torch.abs(c - expected))
    print(f"elementwise_max: max_diff = {max_diff.item():.8f}")
    print(f"  result sample: {c[:5]}")
    print(f"  expected sample: {expected[:5]}")
    torch.testing.assert_close(c, expected, atol=1e-6, rtol=1e-5)
    print("✓ test_elementwise_max passed")


@pytest.mark.skip
def test_elementwise_min():
    """测试：Element-wise min(A, B)"""
    N = 1024

    @T.prim_func
    def kernel(
        A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)
    ):
        with T.Kernel(1, 1) as (tid, _):
            for i in T.Parallel(N):
                C[i] = T.min(A[i], B[i])

    a = torch.randn(N, dtype=torch.float32).npu()
    b = torch.randn(N, dtype=torch.float32).npu()
    c = torch.zeros(N, dtype=torch.float32).npu()

    compiled = tilelang.compile(kernel)
    compiled(a, b, c)

    expected = torch.minimum(a, b)
    max_diff = torch.max(torch.abs(c - expected))
    print(f"elementwise_min: max_diff = {max_diff.item():.8f}")
    print(f"  result sample: {c[:5]}")
    print(f"  expected sample: {expected[:5]}")
    torch.testing.assert_close(c, expected, atol=1e-6, rtol=1e-5)
    print("✓ test_elementwise_min passed")


@pytest.mark.skip
def test_relu():
    """测试：ReLU = max(x, 0)"""
    N = 512

    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(1, 1) as (tid, _):
            for i in T.Parallel(N):
                # 直接使用常量，避免局部变量的 codegen bug
                C[i] = T.max(A[i], T.float32(0.0))

    a = torch.randn(N, dtype=torch.float32).npu()
    c = torch.zeros(N, dtype=torch.float32).npu()

    compiled = tilelang.compile(kernel)
    compiled(a, c)

    expected = torch.relu(a)
    max_diff = torch.max(torch.abs(c - expected))
    print(f"relu: max_diff = {max_diff.item():.8f}")
    torch.testing.assert_close(c, expected, atol=1e-6, rtol=1e-5)
    print("✓ test_relu passed")


@pytest.mark.skip
def test_clip():
    """测试：clip(x, 0, 1) = min(max(x, 0), 1)"""
    N = 256

    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(1, 1) as (tid, _):
            for i in T.Parallel(N):
                # 避免局部变量，直接使用常量
                temp = T.max(A[i], T.float32(0.0))
                C[i] = T.min(temp, T.float32(1.0))

    a = torch.randn(N, dtype=torch.float32).npu()
    c = torch.zeros(N, dtype=torch.float32).npu()

    compiled = tilelang.compile(kernel)
    compiled(a, c)

    expected = torch.clip(a, 0.0, 1.0)
    max_diff = torch.max(torch.abs(c - expected))
    print(f"clip: max_diff = {max_diff.item():.8f}")
    torch.testing.assert_close(c, expected, atol=1e-6, rtol=1e-5)
    print("✓ test_clip passed")


if __name__ == "__main__":
    test_elementwise_max_tilelang()
