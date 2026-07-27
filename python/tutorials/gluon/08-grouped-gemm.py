"""
Grouped GEMM in Gluon for MetaX/MACA.

This keeps Triton's grouped-GEMM device scheduler and mirrors the original
Triton tutorial IR: pointer-table metadata loads, direct global loads, dot, and
store inside the device-side tile loop.
"""

import argparse

import pytest
import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


DEVICE = torch.device("cuda")
BENCHMARK_SHAPES = [(1, 2048, 2048, 2048), (2, 2048, 2048, 2048)]


def is_maca():
    try:
        target = triton.runtime.driver.active.get_current_target()
    except RuntimeError:
        return False
    return target.backend == "maca"


def num_sms():
    if is_maca():
        return 148
    try:
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    except Exception:
        return 148


@gluon.jit
def grouped_matmul_kernel(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_gemm_sizes,
    g_lds,
    group_size: gl.constexpr,
    NUM_SM: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
):
    tile_idx = gl.program_id(0)
    last_problem_end = 0

    for g in gl.static_range(0, group_size):
        gm = gl.load(group_gemm_sizes + g * 3)
        gn = gl.load(group_gemm_sizes + g * 3 + 1)
        gk = gl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = gl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = gl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            lda = gl.load(g_lds + g * 3)
            ldb = gl.load(g_lds + g * 3 + 1)
            ldc = gl.load(g_lds + g * 3 + 2)
            a_ptr = gl.load(group_a_ptrs + g).to(gl.pointer_type(gl.float16))
            b_ptr = gl.load(group_b_ptrs + g).to(gl.pointer_type(gl.float16))
            c_ptr = gl.load(group_c_ptrs + g).to(gl.pointer_type(gl.float16))

            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            offs_m = tile_m_idx * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M)
            offs_n = tile_n_idx * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N)
            offs_k = gl.arange(0, BLOCK_SIZE_K)

            a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
            accumulator = gl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), 0.0, gl.float32)

            for _ in range(0, gl.cdiv(gk, BLOCK_SIZE_K)):
                a_ptrs = gl.multiple_of(a_ptrs, [16, 16])
                b_ptrs = gl.multiple_of(b_ptrs, [16, 16])
                a = gl.load(a_ptrs)
                b = gl.load(b_ptrs)
                accumulator = gl.dot(a, b, accumulator, input_precision="tf32")
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb

            c = accumulator.to(gl.float16)
            c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
            gl.store(c_ptrs, c)

            tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles


def group_gemm_fn(
    group_A,
    group_B,
    BLOCK_SIZE_M=128,
    BLOCK_SIZE_N=128,
    BLOCK_SIZE_K=64,
    NUM_SM=None,
):
    assert len(group_A) == len(group_B)
    group_size = len(group_A)
    assert group_size > 0
    if NUM_SM is None:
        NUM_SM = num_sms()

    a_addrs = []
    b_addrs = []
    c_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []

    for A, B in zip(group_A, group_B):
        assert A.dtype is torch.float16
        assert B.dtype is torch.float16
        assert A.shape[1] == B.shape[0]
        M, K = A.shape
        _, N = B.shape
        assert M % BLOCK_SIZE_M == 0
        assert N % BLOCK_SIZE_N == 0
        assert K % BLOCK_SIZE_K == 0

        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        group_C.append(C)
        a_addrs.append(A.data_ptr())
        b_addrs.append(B.data_ptr())
        c_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]

    d_a_ptrs = torch.tensor(a_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(b_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(c_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

    grouped_matmul_kernel[(NUM_SM, )](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
        NUM_SM,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_warps=8,
        num_ctas=1,
    )
    return group_C


@pytest.mark.skipif(not is_maca(), reason="Requires MetaX/MACA target")
@pytest.mark.parametrize("group_size, M, N, K", [(1, 256, 256, 256), (2, 256, 256, 256)])
def test_maca_grouped_gemm(group_size, M, N, K):
    torch.manual_seed(0)
    group_A = [torch.randn((M, K), device=DEVICE, dtype=torch.float16) for _ in range(group_size)]
    group_B = [torch.randn((K, N), device=DEVICE, dtype=torch.float16) for _ in range(group_size)]
    group_C = group_gemm_fn(group_A, group_B)

    for A, B, C in zip(group_A, group_B, group_C):
        ref = torch.matmul(A, B).to(torch.float16)
        torch.testing.assert_close(C, ref, atol=1e-2, rtol=1e-2)


def _tflops(group_size, M, N, K, ms):
    return group_size * 2.0 * M * N * K * 1e-12 / (ms * 1e-3)


def _shape_name(group_size, M, N, K):
    return f"g{group_size}:{M}x{N}x{K}"


def _make_inputs(group_size, M, N, K):
    torch.manual_seed(0)
    group_A = [torch.randn((M, K), device=DEVICE, dtype=torch.float16) for _ in range(group_size)]
    group_B = [torch.randn((K, N), device=DEVICE, dtype=torch.float16) for _ in range(group_size)]
    return group_A, group_B


def _measure_accuracy(group_A, group_B):
    group_C = group_gemm_fn(group_A, group_B)
    max_abs = 0.0
    allclose = True
    for A, B, C in zip(group_A, group_B, group_C):
        ref = torch.matmul(A, B).to(torch.float16)
        diff = (C - ref).abs()
        max_abs = max(max_abs, diff.max().item())
        allclose = allclose and torch.allclose(C, ref, atol=1e-2, rtol=1e-2)
    return {"max_abs": max_abs, "allclose": allclose}


def _print_markdown_table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(str(item) for item in row) + " |")


def run_accuracy_cases(shapes=((1, 256, 256, 256), (2, 256, 256, 256))):
    rows = []
    for group_size, M, N, K in shapes:
        group_A, group_B = _make_inputs(group_size, M, N, K)
        result = _measure_accuracy(group_A, group_B)
        rows.append([_shape_name(group_size, M, N, K), f"{result['max_abs']:.6g}", result["allclose"]])
    _print_markdown_table(["shape", "max_abs", "allclose"], rows)


def run_benchmark(warmup=25, rep=100, check_correctness=True):
    rows = []
    for group_size, M, N, K in BENCHMARK_SHAPES:
        group_A, group_B = _make_inputs(group_size, M, N, K)
        accuracy = _measure_accuracy(group_A, group_B)
        if check_correctness and not accuracy["allclose"]:
            raise AssertionError(f"grouped GEMM failed correctness for {_shape_name(group_size, M, N, K)}")

        torch_ms = triton.testing.do_bench(
            lambda: [torch.matmul(A, B) for A, B in zip(group_A, group_B)],
            warmup=warmup,
            rep=rep,
        )
        gluon_ms = triton.testing.do_bench(lambda: group_gemm_fn(group_A, group_B), warmup=warmup, rep=rep)
        rows.append([
            _shape_name(group_size, M, N, K),
            f"{torch_ms:.4f}",
            f"{gluon_ms:.4f}",
            f"{_tflops(group_size, M, N, K, torch_ms):.2f}",
            f"{_tflops(group_size, M, N, K, gluon_ms):.2f}",
            f"{accuracy['max_abs']:.6g}",
            accuracy["allclose"],
        ])
    _print_markdown_table(["shape", "torch_ms", "gluon_ms", "torch_tflops", "gluon_tflops", "max_abs", "allclose"], rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(warmup=args.warmup, rep=args.rep, check_correctness=not args.no_check)
    else:
        run_accuracy_cases()
