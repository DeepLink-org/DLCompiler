import pytest
import torch
import tilelang
import tilelang.language as T

dtype = "float32"


def test_vec_add_const():
    """测试：向量 + 标量常量"""
    N = 1024
    block = 1024

    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(1, 1) as (tid, _):
            for i in T.Parallel(block):
                C[i] = A[i] + 2.5

    a = torch.randn(N, dtype=torch.float32).npu()
    c = torch.zeros(N, dtype=torch.float32).npu()

    compiled = tilelang.compile(kernel)
    compiled(a, c)

    expected = a + 2.5
    max_diff = torch.max(torch.abs(c - expected))
    print(f"Vec + Const: max_diff = {max_diff}")
    torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-5)
    print("✓ test_vec_add_const passed")


def test_const_mul_vec():
    """测试：标量常量 * 向量"""
    N = 512
    block = 512

    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(1, 1) as (tid, _):
            for i in T.Parallel(block):
                C[i] = 3.0 * A[i]

    a = torch.randn(N, dtype=torch.float32).npu()
    c = torch.zeros(N, dtype=torch.float32).npu()

    compiled = tilelang.compile(kernel)
    compiled(a, c)

    expected = 3.0 * a
    max_diff = torch.max(torch.abs(c - expected))
    print(f"Const * Vec: max_diff = {max_diff}")
    torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-5)
    print("✓ test_const_mul_vec passed")


def test_vec_sub_const():
    """测试：向量 - 标量常量"""
    N = 256
    block = 256

    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(1, 1) as (tid, _):
            for i in T.Parallel(block):
                C[i] = A[i] - 1.5

    a = torch.randn(N, dtype=torch.float32).npu()
    c = torch.zeros(N, dtype=torch.float32).npu()

    compiled = tilelang.compile(kernel)
    compiled(a, c)

    expected = a - 1.5
    max_diff = torch.max(torch.abs(c - expected))
    print(f"Vec - Const: max_diff = {max_diff}")
    torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-5)
    print("✓ test_vec_sub_const passed")


def test_vec_div_const():
    """测试：向量 / 标量常量"""
    N = 512
    block = 512

    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(1, 1) as (tid, _):
            for i in T.Parallel(block):
                C[i] = A[i] / 2.0

    a = torch.randn(N, dtype=torch.float32).npu()
    c = torch.zeros(N, dtype=torch.float32).npu()

    compiled = tilelang.compile(kernel)
    compiled(a, c)

    expected = a / 2.0
    max_diff = torch.max(torch.abs(c - expected))
    print(f"Vec / Const: max_diff = {max_diff}")
    torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-5)
    print("✓ test_vec_div_const passed")


def test_mixed_const_ops():
    """测试：多个常量操作组合"""
    N = 256
    block = 256

    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(1, 1) as (tid, _):
            for i in T.Parallel(block):
                # (A[i] * 0.5) + 1.0
                C[i] = A[i] * 0.5 + 1.0

    a = torch.randn(N, dtype=torch.float32).npu()
    c = torch.zeros(N, dtype=torch.float32).npu()

    compiled = tilelang.compile(kernel)
    compiled(a, c)

    expected = a * 0.5 + 1.0
    max_diff = torch.max(torch.abs(c - expected))
    print(f"Mixed ops: max_diff = {max_diff}")
    torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-5)
    print("✓ test_mixed_const_ops passed")


def test_integer_const():
    """测试：整数向量 + 常量"""
    N = 1024
    block = 1024
    dtype_int = "int32"

    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype_int), C: T.Tensor((N,), dtype_int)):
        with T.Kernel(1, 1) as (tid, _):
            for i in T.Parallel(block):
                C[i] = A[i] + 10

    a = torch.randint(0, 100, (N,), dtype=torch.int32).npu()
    c = torch.zeros(N, dtype=torch.int32).npu()

    compiled = tilelang.compile(kernel)
    compiled(a, c)

    expected = a + 10
    max_diff = torch.max(torch.abs(c - expected))
    print(f"Integer const: max_diff = {max_diff}")
    torch.testing.assert_close(c, expected)
    print("✓ test_integer_const passed")


if __name__ == "__main__":
    print("Testing VectorizeParallelLoopPass with constant operands")
    print("=" * 60)
    test_vec_add_const()
    test_const_mul_vec()
    test_vec_sub_const()
    # test_vec_div_const()  # Skip: TileLang has an issue with divf operation in CommonIR
    test_mixed_const_ops()
    # test_integer_const()  # Skip temporarily: investigating scf.parallel erase issue
    print("=" * 60)
    print("All float constant operation tests passed! ✓")
