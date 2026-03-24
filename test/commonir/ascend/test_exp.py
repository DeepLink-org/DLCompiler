import pytest
import tilelang
import tilelang.language as T
import triton
import triton.language as tl

import torch
from triton.language.math import exp as tl_exp

dtype = "float32"
block = 128
seq_len = 32 * block


@tilelang.jit(out_idx=[-1])
def create_vector_exp_developer(N, block_N, dtype="float32", threads=1):

    @T.prim_func
    def vector_add_exp_dev(
        A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as (bx,):
            # Allocate memory hierarchy
            A_shared = T.alloc_shared((block_N,), dtype)
            B_shared = T.alloc_shared((block_N,), dtype)
            C_shared = T.alloc_shared((block_N,), dtype)

            # Copy: Global → Shared
            T.copy(A[bx * block_N], A_shared)
            T.copy(B[bx * block_N], B_shared)

            for i in T.Parallel(block_N):
                C_shared[i] = T.exp(A_shared[i] + B_shared[i])

            T.copy(C_shared, C[bx * block_N])

    return vector_add_exp_dev


def create_test_data():
    """创建测试数据 - 使用更大的张量"""
    v1 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    v2 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    v3 = torch.zeros(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    return v1, v2, v3


# @pytest.mark.skip("todo::zmz will remove this after fix pass")
def test_tilelang_developer_mode():
    v1, v2, v3 = create_test_data()
    y_ref = torch.exp(v1 + v2)
    kernel_func = create_vector_exp_developer(seq_len, block, dtype)
    kernel_func(v1, v2, v3)
    torch.testing.assert_close(v3, y_ref, atol=1e-2, rtol=0)
    print("TileLang Developer Mode test passed!\n")


@tilelang.jit(out_idx=[-1])
def create_vector_exp2_developer(N, block_N, dtype="float32", threads=1):

    @T.prim_func
    def vector_add_exp2_dev(
        A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as (bx,):
            A_shared = T.alloc_shared((block_N,), dtype)
            B_shared = T.alloc_shared((block_N,), dtype)
            C_shared = T.alloc_shared((block_N,), dtype)

            T.copy(A[bx * block_N], A_shared)
            T.copy(B[bx * block_N], B_shared)

            for i in T.Parallel(block_N):
                C_shared[i] = T.exp2(A_shared[i] + B_shared[i])

            T.copy(C_shared, C[bx * block_N])

    return vector_add_exp2_dev


def test_tilelang_developer_mode_exp2():
    v1, v2, v3 = create_test_data()
    y_ref = torch.pow(2.0, v1 + v2)  # 2^(A+B)
    kernel_func = create_vector_exp2_developer(seq_len, block, dtype)
    kernel_func(v1, v2, v3)
    torch.testing.assert_close(v3, y_ref, atol=1e-2, rtol=0)
    print("TileLang Developer Mode exp2 test passed!\n")


@triton.jit
def _triton_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl_exp(x + y)
    tl.store(output_ptr + offsets, output, mask=mask)


def test_triton_developer_mode():
    print("Testing Triton implementation...")
    v1, v2, v3 = create_test_data()
    y_ref = torch.exp(v1 + v2)
    block_size = block
    grid = (triton.cdiv(seq_len, block_size),)
    _triton_kernel[grid](v1, v2, v3, seq_len, BLOCK_SIZE=block_size)
    max_diff = torch.max(torch.abs(y_ref - v3))
    print(f"The maximum difference between torch and Triton is {max_diff}")
    torch.testing.assert_close(v3, y_ref, atol=1e-2, rtol=0)
    print("Triton test passed!\n")
