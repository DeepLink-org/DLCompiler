import torch
import torch_npu
import tilelang
import tilelang.language as T
import pytest
import triton
import triton.language as tl


@triton.jit
def row_max_sum_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    stride_am,
    stride_an,
    stride_b,
    stride_c,
    N: tl.constexpr,
    BM: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)  # [BM]
    row_mask = rows < M
    cols = tl.arange(0, N)
    ptrs = A_ptr + rows[:, None] * stride_am + cols[None, :] * stride_an
    x = tl.load(ptrs, mask=row_mask[:, None], other=0.0)
    max_vals = tl.max(x, axis=1)  # [BM]
    sum_vals = tl.sum(x, axis=1)
    tl.store(B_ptr + rows * stride_b, max_vals, mask=row_mask)
    tl.store(C_ptr + rows * stride_c, sum_vals, mask=row_mask)


def reduce_triton(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, BM=64):
    M, N = a.shape
    grid = (triton.cdiv(M, BM),)
    row_max_sum_kernel[grid](
        a,
        b,
        c,
        M,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        c.stride(0),
        N=N,
        BM=BM,
    )


def test_reduce_triton():
    M, N = 1024, 32
    a = torch.randn((M, N), device="npu", dtype=torch.float32)
    b = torch.zeros((M,), device="npu", dtype=torch.float32)
    c = torch.zeros((M,), device="npu", dtype=torch.float32)
    reduce_triton(a, b, c, BM=64)
    max_ref = torch.max(a, dim=-1, keepdim=False)
    torch.testing.assert_close(b, max_ref[0])
    sum_ref = torch.sum(a, dim=-1, keepdim=False)
    torch.testing.assert_close(c, sum_ref)


def test_reduce_tilelang():
    M, N = 1024, 32
    BM, BN = 64, 64
    dtype = "float32"

    def if_then_else(M, BM, N, dtype="float32"):
        @T.prim_func
        def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M,), dtype),
            C: T.Tensor((M,), dtype),
        ):
            with T.Kernel(T.ceildiv(M, BM), 1) as (bx, _):
                acc_s = T.alloc_fragment([BM, N], dtype)
                scores_max = T.alloc_fragment([BM], dtype)
                scores_sum = T.alloc_fragment([BM], dtype)

                T.copy(A[bx * BM : (bx + 1) * BM, :], acc_s)
                T.fill(scores_max, -T.infinity(dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(scores_max, B[bx * BM])
                T.copy(scores_sum, C[bx * BM])

        return main

    a = torch.randn((M, N), dtype=torch.float32, device="npu")
    b = torch.zeros((M,), dtype=torch.float32, device="npu")
    c = torch.zeros((M,), dtype=torch.float32, device="npu")
    func = if_then_else(M, BM, N, dtype)
    mod = tilelang.compile(func)
    mod(a, b, c)

    max_ref = torch.max(a, dim=-1, keepdim=False)
    torch.testing.assert_close(b, max_ref[0])
    sum_ref = torch.sum(a, dim=-1, keepdim=False)
    torch.testing.assert_close(c, sum_ref)
