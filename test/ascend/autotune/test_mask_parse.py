import pytest
import triton
import triton.language as tl
from test_common import check_axes_parse_res, mock_autotuner


def test_triton_dot_case1(mock_autotuner):
    @triton.autotune(configs=[], key=["M", "N", "K"])
    @triton.jit
    def triton_dot_case1(
        A,
        B,
        C,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        MBLOCK: tl.constexpr,
        NBLOCK: tl.constexpr,
        MBLOCK_SUB: tl.constexpr,
        NBLOCK_SUB: tl.constexpr,
        KBLOCK_SUB: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        base_m = pid_m * MBLOCK
        base_n = pid_n * NBLOCK

        loops_m = (MBLOCK + MBLOCK_SUB - 1) // MBLOCK_SUB
        loops_n = (NBLOCK + NBLOCK_SUB - 1) // NBLOCK_SUB
        loops_k = (K + KBLOCK_SUB - 1) // KBLOCK_SUB

        for loop_m in range(loops_m):
            for loop_n in range(loops_n):
                acc = tl.zeros((MBLOCK_SUB, NBLOCK_SUB), dtype=tl.float32)

                mdx = base_m + loop_m * MBLOCK_SUB + tl.arange(0, MBLOCK_SUB)[:, None]
                ndx = base_n + loop_n * NBLOCK_SUB + tl.arange(0, NBLOCK_SUB)[None, :]

                for loop_k in range(loops_k):
                    kdx = loop_k * KBLOCK_SUB + tl.arange(0, KBLOCK_SUB)
                    kdx_m = kdx[None, :]
                    A_ptr = A + mdx * K + kdx_m
                    a_mask = (mdx < M) & (kdx_m < K)
                    a = tl.load(A_ptr, mask=a_mask, other=0.0)

                    kdx_n = kdx[:, None]
                    B_ptr = B + kdx_n * N + ndx
                    b_mask = (kdx_n < K) & (ndx < N)
                    b = tl.load(B_ptr, mask=b_mask, other=0.0)

                    acc += tl.dot(a, b)

                C_ptr = C + mdx * N + ndx
                c_mask = (mdx < M) & (ndx < N)
                tl.store(C_ptr, acc, mask=c_mask)

    ref_res = {
        "keys": {"x": "M", "y": "N", "z": "K"},
        "split_params": {"x": "MBLOCK", "y": "NBLOCK"},
        "tiling_params": {"x": "MBLOCK_SUB", "y": "NBLOCK_SUB", "z": "KBLOCK_SUB"},
        "low_dim_axes": ["y", "z"],
        "reduction_axes": [],
    }
    grid = lambda meta: (meta["MBLOCK"], meta["NBLOCK"])
    act_res = triton_dot_case1[grid]()

    check_axes_parse_res(act_res, ref_res)


@pytest.mark.skip
def test_triton_dot_case2(mock_autotuner):
    @triton.autotune(configs=[], key=["M", "N", "K"])
    @triton.jit
    def triton_dot_case2(
        A,
        B,
        C,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        MBLOCK: tl.constexpr,
        NBLOCK: tl.constexpr,
        MBLOCK_SUB: tl.constexpr,
        NBLOCK_SUB: tl.constexpr,
        KBLOCK_SUB: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        base_m = pid_m * MBLOCK
        base_n = pid_n * NBLOCK

        loops_m = (MBLOCK + MBLOCK_SUB - 1) // MBLOCK_SUB
        loops_n = (NBLOCK + NBLOCK_SUB - 1) // NBLOCK_SUB
        loops_k = (K + KBLOCK_SUB - 1) // KBLOCK_SUB

        for loop_m in range(loops_m):
            for loop_n in range(loops_n):
                acc = tl.zeros((MBLOCK_SUB, NBLOCK_SUB), dtype=tl.float32)

                mdx = base_m + loop_m * MBLOCK_SUB + tl.arange(0, MBLOCK_SUB)[:, None]
                ndx = base_n + loop_n * NBLOCK_SUB + tl.arange(0, NBLOCK_SUB)[None, :]

                for loop_k in range(loops_k):
                    kdx = loop_k * KBLOCK_SUB + tl.arange(0, KBLOCK_SUB)
                    A_ptr = A + mdx * K + kdx[None, :]
                    a_mask = (mdx < M) & (kdx[None, :] < K)
                    a = tl.load(A_ptr, mask=a_mask, other=0.0)

                    B_ptr = B + kdx[:, None] * N + ndx
                    b_mask = (kdx[:, None] < K) & (ndx < N)
                    b = tl.load(B_ptr, mask=b_mask, other=0.0)

                    acc += tl.dot(a, b)

                C_ptr = C + mdx * N + ndx
                c_mask = (mdx < M) & (ndx < N)
                tl.store(C_ptr, acc, mask=c_mask)

    ref_res = {
        "keys": {"x": "M", "y": "N", "z": "K"},
        "split_params": {"x": "MBLOCK", "y": "NBLOCK"},
        "tiling_params": {"x": "MBLOCK_SUB", "y": "NBLOCK_SUB", "z": "KBLOCK_SUB"},
        "low_dim_axes": ["y", "z"],
        "reduction_axes": [],
    }
    grid = lambda meta: (meta["MBLOCK"], meta["NBLOCK"])
    act_res = triton_dot_case2[grid]()

    check_axes_parse_res(act_res, ref_res)
