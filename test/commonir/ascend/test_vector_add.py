import os
import pytest
import tilelang
import tilelang.language as T

import triton
import triton.language as tl
from triton.testing import do_bench

import torch

dtype = "float32"
seq_len = 1024 * 1024  # 增大到1M个元素，更能体现kernel优化效果，1048576
block = 1024


def vec_add(N, block_N, dtype="float32"):
    n_num = (
        N // block_N
    )  # n_num是块的数量 32，block_N是每个块处理的元素数量 32k，N是总元素数量 1M

    @T.prim_func
    def main(
        A: T.Tensor((N), dtype),
        B: T.Tensor((N), dtype),
        C: T.Tensor((N), dtype),
    ):
        with T.Kernel(n_num, 1) as (tid, _):
            start_idx = tid * block_N
            for local_y in T.Parallel(block_N):
                y = start_idx + local_y
                C[y] = A[y] + B[y]

    return main


@tilelang.jit(out_idx=[-1])
def create_vector_add_developer(N, block_N, dtype="float32", threads=1):
    """
    TileLang developer mode implementation using standard pattern.

    This follows the standard TileLang developer mode pattern:
    - @tilelang.jit decorator with out_idx specification
    - T.alloc_shared for explicit shared memory allocation
    - T.alloc_fragment for register allocation
    - T.copy for explicit data copying between memory hierarchies
    - T.Parallel for explicit parallelization

    This pattern demonstrates the full memory hierarchy:
    Global → Shared → Fragment → Shared → Global

    VectorizeParallelLoopPass will optimize the T.Parallel loops.

    Args:
        N: Total number of elements
        block_N: Block size (number of elements per block)
        dtype: Data type
        threads: Number of threads per block

    Returns:
        Compiled kernel function (via @tilelang.jit)
    """

    @T.prim_func
    def vector_add_dev(
        A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as (bx,):
            # Allocate memory hierarchy
            A_shared = T.alloc_shared((block_N,), dtype)
            B_shared = T.alloc_shared((block_N,), dtype)
            C_local = T.alloc_fragment((block_N,), dtype)
            C_shared = T.alloc_shared((block_N,), dtype)

            # Copy: Global → Shared
            T.copy(A[bx * block_N], A_shared)
            T.copy(B[bx * block_N], B_shared)

            # Compute: Shared → Fragment
            for i in T.Parallel(block_N):
                C_local[i] = A_shared[i] + B_shared[i]

            # Copy: Fragment → Shared → Global
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[bx * block_N])

    return vector_add_dev


@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def create_test_data():
    """创建测试数据 - 使用更大的张量"""
    v1 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    v2 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    v3 = torch.zeros(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    return v1, v2, v3


def test_tilelang_add():
    """测试 TileLang 实现"""
    print("Testing TileLang implementation...")

    # 创建测试数据
    v1, v2, v3 = create_test_data()
    y_ref = v1 + v2

    # 编译 TileLang kernel
    # func = vec_add(seq_len, seq_len // 32)  # 使用更合适的块大小 1M, 32K
    func = vec_add(seq_len, seq_len // block)
    compiled_kernel = tilelang.compile(func)

    # 执行 TileLang kernel
    compiled_kernel(v1, v2, v3)

    # 验证结果
    max_diff = torch.max(torch.abs(y_ref - v3))
    print(f"The maximum difference between torch and TileLang is {max_diff}")

    torch.testing.assert_close(v3, y_ref, atol=1e-2, rtol=0)
    print("TileLang test passed!\n")


def test_triton_add():
    """测试 Triton 实现"""
    print("Testing Triton implementation...")

    # 创建测试数据
    v1, v2, v3 = create_test_data()
    y_ref = v1 + v2

    # 设置块大小和网格 - 适应更大的数据集
    block_size = block  # Triton常用的块大小
    grid = (triton.cdiv(seq_len, block_size),)  # 修正网格定义

    # 执行 Triton kernel
    add_kernel[grid](v1, v2, v3, seq_len, BLOCK_SIZE=block_size)

    # 验证结果
    max_diff = torch.max(torch.abs(y_ref - v3))
    print(f"The maximum difference between torch and Triton is {max_diff}")

    torch.testing.assert_close(v3, y_ref, atol=1e-2, rtol=0)
    print("Triton test passed!\n")


def test_tilelang_developer_mode():
    """
    测试 TileLang 开发者模式实现（标准模式）

    使用标准 TileLang 开发者模式 API：
    - @tilelang.jit 装饰器 (out_idx 指定输出参数)
    - T.alloc_shared 显式分配共享内存
    - T.alloc_fragment 显式分配寄存器
    - T.copy 显式数据拷贝（global→shared→fragment→shared→global）
    - T.Parallel 显式并行化

    这是固定的标准写法，VectorizeParallelLoopPass 需要能够处理这种模式。
    """
    print("Testing TileLang Developer Mode implementation...")

    # 创建测试数据
    v1, v2, v3 = create_test_data()
    y_ref = v1 + v2

    # 调用 @tilelang.jit 装饰的函数
    # 第一次调用会触发编译，返回 compiled kernel
    kernel_func = create_vector_add_developer(seq_len, block, dtype)

    # 执行 kernel
    kernel_func(v1, v2, v3)

    # 验证结果
    max_diff = torch.max(torch.abs(y_ref - v3))
    print(
        f"The maximum difference between torch and TileLang Developer Mode is {max_diff}"
    )

    torch.testing.assert_close(v3, y_ref, atol=1e-2, rtol=0)
    print("TileLang Developer Mode test passed!\n")


def run_performance_tests():
    """运行性能测试"""
    print("=" * 60)
    print(
        f"PERFORMANCE TESTS - Vector size: {seq_len:,} elements ({seq_len * 4 / 1e6:.2f} MB)"
    )
    print("=" * 60)

    # 创建测试数据
    v1, v2, v3 = create_test_data()

    # TileLang kernel
    func = vec_add(seq_len, seq_len // block)  # 使用更合适的块大小
    compiled_tilelang_kernel = tilelang.compile(func)

    def tilelang_benchmark():
        temp_v3 = torch.zeros_like(v3)
        compiled_tilelang_kernel(v1, v2, temp_v3)

    def triton_benchmark():
        temp_v3 = torch.zeros_like(v3)
        block_size = block  # Triton常用块大小
        grid = (triton.cdiv(seq_len, block_size),)
        add_kernel[grid](v1, v2, temp_v3, seq_len, BLOCK_SIZE=block_size)

    def torch_benchmark():
        temp_result = v1 + v2

    # 运行基准测试
    print("Running benchmarks... (this may take a moment)")

    tilelang_time = do_bench(tilelang_benchmark)
    triton_time = do_bench(triton_benchmark)
    torch_time = do_bench(torch_benchmark)

    print(f"\nAverage execution time over runs:")
    print(f"  TileLang: {tilelang_time*1000:.4f} ms")  # 转换为毫秒
    print(f"  Triton:   {triton_time*1000:.4f} ms")  # 转换为毫秒
    print(f"  PyTorch:  {torch_time*1000:.4f} ms")  # 转换为毫秒

    # 计算吞吐量
    total_elements = seq_len
    triton_throughput = (total_elements * 4 * 3 / 1e9) / (
        triton_time
    )  # GB/s (read two + write one)
    tilelang_throughput = (total_elements * 4 * 3 / 1e9) / (tilelang_time)  # GB/s
    torch_throughput = (total_elements * 4 * 3 / 1e9) / (torch_time)  # GB/s

    print(f"\nThroughput Analysis:")
    print(f"  Triton:   {triton_throughput:.2f} GB/s")
    print(f"  TileLang: {tilelang_throughput:.2f} GB/s")
    print(f"  PyTorch:  {torch_throughput:.2f} GB/s")

    print(f"\nPerformance comparison relative to PyTorch:")
    if triton_time <= torch_time:
        speedup = torch_time / triton_time
        print(f"  Triton is {speedup:.2f}x FASTER than PyTorch")
    else:
        slowdown = triton_time / torch_time
        print(f"  Triton is {slowdown:.2f}x SLOWER than PyTorch")

    if tilelang_time <= torch_time:
        speedup = torch_time / tilelang_time
        print(f"  TileLang is {speedup:.2f}x FASTER than PyTorch")
    else:
        slowdown = tilelang_time / torch_time
        print(f"  TileLang is {slowdown:.2f}x SLOWER than PyTorch")

    if triton_time <= tilelang_time:
        speedup = tilelang_time / triton_time
        print(f"  Triton is {speedup:.2f}x FASTER than TileLang")
    else:
        slowdown = triton_time / tilelang_time
        print(f"  Triton is {slowdown:.2f}x SLOWER than TileLang")


def main():
    """主函数"""
    print("Vector Addition Comparison: TileLang vs Triton")
    print("=" * 60)

    # 运行功能测试
    print("FUNCTIONALITY TESTS")
    print("-" * 20)
    test_tilelang_add()
    test_tilelang_developer_mode()
    test_triton_add()

    # 运行性能测试
    run_performance_tests()


if __name__ == "__main__":
    main()
