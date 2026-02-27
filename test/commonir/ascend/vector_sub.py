import os
import time

import tilelang
import tilelang.language as T

import triton
import triton.language as tl

import torch

dtype = "float32"
seq_len = 1024 * 1024  # 增大到1M个元素，更能体现kernel优化效果，1048576
block = 1024


def vec_sub(N, block_N, dtype="float32"):
    n_num = (
        N // block_N
    )  # n_num是块的数量 32，block_N是每个块处理的元素数量 32k，N是总元素数量 1M
    # print(f"zmz debug : n_num={n_num}, block_N={block_N}, N={N}")

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
                C[y] = A[y] - B[y]

    return main


@triton.jit
def sub_kernel(
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
    output = x - y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def create_test_data():
    """创建测试数据 - 使用更大的张量"""
    v1 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    v2 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    v3 = torch.zeros(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    return v1, v2, v3


def test_tilelang_sub():
    """测试 TileLang 实现"""
    print("Testing TileLang implementation...")

    # 创建测试数据
    v1, v2, v3 = create_test_data()
    y_ref = v1 - v2

    # 编译 TileLang kernel
    func = vec_sub(seq_len, seq_len // block)
    compiled_kernel = tilelang.compile(func, target="commonir")

    # 执行 TileLang kernel
    compiled_kernel(v1, v2, v3)

    # 验证结果
    max_diff = torch.max(torch.abs(y_ref - v3))
    print(f"The maximum difference between torch and TileLang is {max_diff}")

    torch.testing.assert_close(v3, y_ref, atol=1e-2, rtol=0)
    print("TileLang test passed!\n")

    return v1, v2, v3, y_ref


def test_triton_sub():
    """测试 Triton 实现"""
    print("Testing Triton implementation...")

    # 创建测试数据
    v1, v2, v3 = create_test_data()
    y_ref = v1 - v2

    # 设置块大小和网格 - 适应更大的数据集
    block_size = block  # Triton常用的块大小
    grid = (triton.cdiv(seq_len, block_size),)  # 修正网格定义

    # 执行 Triton kernel
    sub_kernel[grid](v1, v2, v3, seq_len, BLOCK_SIZE=block_size)

    # 验证结果
    max_diff = torch.max(torch.abs(y_ref - v3))
    print(f"The maximum difference between torch and Triton is {max_diff}")

    torch.testing.assert_close(v3, y_ref, atol=1e-2, rtol=0)
    print("Triton test passed!\n")

    return v1, v2, v3, y_ref


def benchmark_function(func, *args, num_runs=100, warmup_runs=10):
    """性能测试函数"""
    # 预热运行
    for _ in range(warmup_runs):
        func(*args)

    # 同步NPU
    if torch.npu.is_available():
        torch.npu.synchronize()

    # 正式测试
    start_time = time.time()
    for _ in range(num_runs):
        func(*args)

    # 同步NPU
    if torch.npu.is_available():
        torch.npu.synchronize()

    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    return avg_time * 1000  # 转换为毫秒


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
    func = vec_sub(seq_len, seq_len // block)  # 使用更合适的块大小
    compiled_tilelang_kernel = tilelang.compile(func, target="commonir")

    def tilelang_benchmark():
        temp_v3 = torch.zeros_like(v3)
        compiled_tilelang_kernel(v1, v2, temp_v3)

    def triton_benchmark():
        temp_v3 = torch.zeros_like(v3)
        block_size = block  # Triton常用块大小
        grid = (triton.cdiv(seq_len, block_size),)
        sub_kernel[grid](v1, v2, temp_v3, seq_len, BLOCK_SIZE=block_size)

    def torch_benchmark():
        temp_result = v1 - v2

    # 运行基准测试
    print("Running benchmarks... (this may take a moment)")

    tilelang_time = benchmark_function(tilelang_benchmark, num_runs=100, warmup_runs=10)
    triton_time = benchmark_function(triton_benchmark, num_runs=100, warmup_runs=10)
    torch_time = benchmark_function(torch_benchmark, num_runs=100, warmup_runs=10)

    print(f"\nAverage execution time over 100 runs:")
    print(f"  TileLang: {tilelang_time:.4f} ms")
    print(f"  Triton:   {triton_time:.4f} ms")
    print(f"  PyTorch:  {torch_time:.4f} ms")

    # 计算吞吐量
    total_elements = seq_len
    triton_throughput = (total_elements * 4 * 3 / 1e9) / (
        triton_time / 1000
    )  # GB/s (read two + write one)
    tilelang_throughput = (total_elements * 4 * 3 / 1e9) / (
        tilelang_time / 1000
    )  # GB/s
    torch_throughput = (total_elements * 4 * 3 / 1e9) / (torch_time / 1000)  # GB/s

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
    print("Vector Sub Comparison: TileLang vs Triton")
    print("=" * 60)

    # 运行功能测试
    print("FUNCTIONALITY TESTS")
    print("-" * 20)
    tilelang_data = test_tilelang_sub()
    triton_data = test_triton_sub()

    # 运行性能测试
    run_performance_tests()


if __name__ == "__main__":
    main()
