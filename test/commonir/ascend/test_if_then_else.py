import torch
import tilelang
import tilelang.language as T
import pytest
import triton
import triton.language as tl
from triton.testing import do_bench


def pytorch_if_then_else(a):
    """PyTorch implementation of if_then_else operation"""
    return torch.where(a > 0, a, torch.zeros_like(a))


@triton.jit
def triton_if_then_else_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    output_vals = tl.where(input_vals > 0, input_vals, 0.0)
    tl.store(output_ptr + offsets, output_vals, mask=mask)


def triton_if_then_else(a):
    """Triton implementation of if_then_else operation"""
    n_elements = a.numel()
    block_size = 1024
    output = torch.zeros_like(a)
    grid = (triton.cdiv(n_elements, block_size),)

    triton_if_then_else_kernel[grid](a, output, n_elements, BLOCK_SIZE=block_size)
    return output


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
    print("T.if_then_else 1d test passed")


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
    print("T.if_then_else 2d test passed")


def performance_test_if_then_else_2d():
    """Separate performance test function for if_then_else 2d with larger tensors"""
    M, N = 1024, 1024
    BM, BN = 64, 64
    dtype = "float32"

    def if_then_else(M, BM, N, BN, dtype="float32"):
        @T.prim_func
        def tilelang_main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
            with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN)) as (bx, by):
                A_shared = T.alloc_shared((BM, BN), dtype)
                B_shared = T.alloc_shared((BM, BN), dtype)
                T.copy(A[bx * BM : (bx + 1) * BM, by * BN : (by + 1) * BN], A_shared)
                for i, j in T.Parallel(BM, BN):
                    B_shared[i, j] = T.if_then_else(
                        A_shared[i, j] > 0.0, A_shared[i, j], 0.0
                    )

                T.copy(B_shared, B[bx * BM : (bx + 1) * BM, by * BN : (by + 1) * BN])

        return tilelang_main

    print("=" * 60)
    print(f"PERFORMANCE TESTS - Matrix size: {M}x{N} ({M * N:,} elements)")
    print("=" * 60)

    # 创建测试数据
    a = torch.randn((M, N), dtype=torch.float32).npu()
    b = torch.zeros((M, N), dtype=torch.float32).npu()

    # TileLang implementation
    func = if_then_else(M, BM, N, BN, dtype)
    mod = tilelang.compile(func)

    def tilelang_benchmark():
        temp_b = torch.zeros_like(b)
        mod(a, temp_b)

    # PyTorch implementation
    def pytorch_benchmark():
        result = pytorch_if_then_else(a)

    # Triton implementation
    triton_output = torch.zeros_like(a)
    n_elements = a.numel()
    block_size = 1024
    triton_grid = (triton.cdiv(n_elements, block_size),)

    def triton_benchmark():
        triton_if_then_else_kernel[triton_grid](
            a, triton_output, n_elements, BLOCK_SIZE=block_size
        )

    # Run benchmarks using do_bench
    print("Running benchmarks... (this may take a moment)")

    tilelang_time = do_bench(tilelang_benchmark)
    pytorch_time = do_bench(pytorch_benchmark)
    triton_time = do_bench(triton_benchmark)

    # Verify correctness
    mod(a, b)  # Run once more to get the result for comparison
    expected = torch.where(a > 0, a, torch.zeros_like(a))
    torch.testing.assert_close(b, expected, atol=1e-5, rtol=1e-5)

    # Also verify Triton output
    triton_result = triton_if_then_else(a)
    torch.testing.assert_close(triton_result, expected, atol=1e-5, rtol=1e-5)

    print(f"\nAverage execution time over runs:")
    print(f"  TileLang: {tilelang_time*1000:.4f} ms")
    print(f"  PyTorch:  {pytorch_time*1000:.4f} ms")
    print(f"  Triton:   {triton_time*1000:.4f} ms")

    print(f"\nPerformance comparison relative to PyTorch:")
    if tilelang_time <= pytorch_time:
        speedup = pytorch_time / tilelang_time
        print(f"  TileLang is {speedup:.2f}x FASTER than PyTorch")
    else:
        slowdown = tilelang_time / pytorch_time
        print(f"  TileLang is {slowdown:.2f}x SLOWER than PyTorch")

    if triton_time <= pytorch_time:
        speedup = pytorch_time / triton_time
        print(f"  Triton is {speedup:.2f}x FASTER than PyTorch")
    else:
        slowdown = triton_time / pytorch_time
        print(f"  Triton is {slowdown:.2f}x SLOWER than PyTorch")

    if triton_time <= tilelang_time:
        speedup = tilelang_time / triton_time
        print(f"  Triton is {speedup:.2f}x FASTER than TileLang")
    else:
        slowdown = triton_time / tilelang_time
        print(f"  Triton is {slowdown:.2f}x SLOWER than TileLang")


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
    print("T.if_then_else test passed")


if __name__ == "__main__":
    test_if_then_else_1d()
    test_if_then_else_2d()
    test_triton_if_then_else()
    performance_test_if_then_else_2d()
