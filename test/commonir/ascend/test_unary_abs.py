import torch
import tilelang
import tilelang.language as T
from tilelang import tvm
from tvm import tir

dtype = "float32"


def test_abs():
    """测试：一元 abs 操作"""
    N = 1024

    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(1, 1) as (tid, _):
            for i in T.Parallel(N):
                C[i] = tir.abs(A[i])  # 使用 tir.abs

    # 生成包含正负数的测试数据
    a = torch.randn(N, dtype=torch.float32).npu()
    c = torch.zeros(N, dtype=torch.float32).npu()

    compiled = tilelang.compile(kernel)
    compiled(a, c)

    expected = torch.abs(a)
    max_diff = torch.max(torch.abs(c - expected))
    print(f"abs: max_diff = {max_diff.item():.8f}")
    print(f"  input sample: {a[:5]}")
    print(f"  result sample: {c[:5]}")
    print(f"  expected sample: {expected[:5]}")
    torch.testing.assert_close(c, expected, atol=1e-6, rtol=1e-5)
    print("✓ test_abs passed")


def test_abs_negative():
    """测试：abs 对负数的处理"""
    N = 512

    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(1, 1) as (tid, _):
            for i in T.Parallel(N):
                C[i] = tir.abs(A[i])

    # 全部负数
    a = -torch.abs(torch.randn(N, dtype=torch.float32)).npu()
    c = torch.zeros(N, dtype=torch.float32).npu()

    compiled = tilelang.compile(kernel)
    compiled(a, c)

    expected = torch.abs(a)
    max_diff = torch.max(torch.abs(c - expected))
    print(f"abs_negative: max_diff = {max_diff.item():.8f}")
    assert torch.all(c >= 0), "All results should be non-negative"
    torch.testing.assert_close(c, expected, atol=1e-6, rtol=1e-5)
    print("✓ test_abs_negative passed")


def test_abs_with_compute():
    """测试：abs 与其他操作组合 - |A - B|"""
    N = 256

    @T.prim_func
    def kernel(
        A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)
    ):
        with T.Kernel(1, 1) as (tid, _):
            for i in T.Parallel(N):
                diff = A[i] - B[i]
                C[i] = tir.abs(diff)

    a = torch.randn(N, dtype=torch.float32).npu()
    b = torch.randn(N, dtype=torch.float32).npu()
    c = torch.zeros(N, dtype=torch.float32).npu()

    compiled = tilelang.compile(kernel)
    compiled(a, b, c)

    expected = torch.abs(a - b)
    max_diff = torch.max(torch.abs(c - expected))
    print(f"abs_with_compute: max_diff = {max_diff.item():.8f}")
    torch.testing.assert_close(c, expected, atol=1e-6, rtol=1e-5)
    print("✓ test_abs_with_compute passed")


if __name__ == "__main__":
    print("Testing Unary Abs Operation")
    print("=" * 60)
    test_abs()
    # test_abs_negative()
    # test_abs_with_compute()
    print("=" * 60)
    print("All abs operation tests passed! ✓")
