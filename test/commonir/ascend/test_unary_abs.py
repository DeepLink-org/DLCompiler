import torch
import tilelang
import tilelang.language as T
from tilelang import tvm
from tvm import tir
import triton
import triton.language as tl
from triton.testing import do_bench

dtype = "float32"
seq_len = 1024 * 1024
block = 1024


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


@triton.jit
def abs_kernel(
    x_ptr,  # *Pointer* to input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.abs(x)  # Apply absolute value
    # Write result back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def create_abs_test_data():
    """创建 abs 测试数据"""
    a = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    c = torch.zeros(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    return a, c


def run_abs_performance_tests():
    """运行 abs 算子的性能测试"""
    print("=" * 60)
    print(
        f"ABS PERFORMANCE TESTS - Vector size: {seq_len:,} elements ({seq_len * 4 / 1e6:.2f} MB)"
    )
    print("=" * 60)

    # 创建测试数据
    a, c = create_abs_test_data()

    # TileLang abs kernel
    @T.prim_func
    def tilelang_abs_kernel(
        A: T.Tensor((seq_len,), dtype), C: T.Tensor((seq_len,), dtype)
    ):
        with T.Kernel(T.ceildiv(seq_len, block), 1) as (tid, _):
            start_idx = tid * block
            for local_i in T.Parallel(block):
                i = start_idx + local_i
                C[i] = tir.abs(A[i])

    compiled_tilelang_kernel = tilelang.compile(tilelang_abs_kernel)

    def tilelang_benchmark():
        temp_c = torch.zeros_like(c)
        compiled_tilelang_kernel(a, temp_c)

    def triton_benchmark():
        temp_c = torch.zeros_like(c)
        block_size = block  # Triton常用块大小
        grid = (triton.cdiv(seq_len, block_size),)
        abs_kernel[grid](a, temp_c, seq_len, BLOCK_SIZE=block_size)

    def torch_benchmark():
        temp_result = torch.abs(a)

    # 运行基准测试
    print("Running benchmarks... (this may take a moment)")

    tilelang_time = do_bench(tilelang_benchmark)
    triton_time = do_bench(triton_benchmark)
    torch_time = do_bench(torch_benchmark)

    print(f"\nAverage execution time over 100 runs:")
    print(f"  TileLang: {tilelang_time:.4f} ms")
    print(f"  Triton:   {triton_time:.4f} ms")
    print(f"  PyTorch:  {torch_time:.4f} ms")

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
    print("Unary Abs Operation Tests")
    print("=" * 60)

    # 运行功能测试
    print("FUNCTIONALITY TESTS")
    print("-" * 20)
    test_abs()
    test_abs_negative()
    test_abs_with_compute()

    # 运行性能测试
    run_abs_performance_tests()


if __name__ == "__main__":
    main()
