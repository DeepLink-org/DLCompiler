import pytest
import torch
import numpy as np
import triton
import triton.language as tl


@triton.jit
def load_kernel_tma(Z, desc, SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    off_desc = 0
    z_ptrs = Z + tl.arange(0, BLOCK_SIZE)
    for k in range(0, tl.cdiv(SIZE, BLOCK_SIZE)):
        x = tl._experimental_descriptor_load(
            desc, [off_desc], [BLOCK_SIZE], Z.dtype.element_ty
        )
        tl.store(z_ptrs, x)
        z_ptrs += BLOCK_SIZE
        off_desc += BLOCK_SIZE


@triton.jit
def load_kernel_tl(Z, x, SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    x_ptrs = x + tl.arange(0, BLOCK_SIZE)
    z_ptrs = Z + tl.arange(0, BLOCK_SIZE)
    for k in range(0, tl.cdiv(SIZE, BLOCK_SIZE)):
        xd = tl.load(x_ptrs)
        tl.store(z_ptrs, xd)
        x_ptrs += BLOCK_SIZE
        z_ptrs += BLOCK_SIZE


def test_experimetal_descriptor_load():
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] == 9:
        pytest.skip("Test requires Hopper target.")
        return
    device = "cuda"
    SIZE = 256
    x = torch.randn(SIZE, dtype=torch.float32, device=device)
    desc = np.empty(SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_1d_tma_descriptor(
        x.data_ptr(), SIZE, SIZE, x.element_size(), desc
    )
    desc = torch.tensor(desc, device=device)
    z_tri = torch.empty_like(x)

    load_kernel_tma[(1,)](z_tri, desc, SIZE=SIZE, num_warps=4)
    assert torch.equal(x, z_tri)
    tl_res = torch.empty_like(x)
    load_kernel_tl[(1,)](tl_res, x, SIZE=SIZE, num_warps=4)
    assert torch.equal(x, tl_res)


@triton.jit
def matmul_kernel_tma(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,  #
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl._experimental_descriptor_load(
            a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], tl.float16
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr, [offs_k, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], tl.float16
        )
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
    accumulator = accumulator.to(tl.float16)
    tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am, offs_bn])


@triton.jit
def add_kernel_tma(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,  #
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_m = pid_m * BLOCK_SIZE_M
    offs_n = pid_n * BLOCK_SIZE_N
    a = tl._experimental_descriptor_load(
        a_desc_ptr, [offs_m, offs_n], [BLOCK_SIZE_M, BLOCK_SIZE_N], tl.float32
    )
    b = tl._experimental_descriptor_load(
        b_desc_ptr, [offs_m, offs_n], [BLOCK_SIZE_M, BLOCK_SIZE_N], tl.float32
    )
    tl._experimental_descriptor_store(c_desc_ptr, a + b, [offs_m, offs_n])


@triton.jit
def add_kernel_tl(
    a_ptr,
    b_ptr,
    c_ptr,  #
    M,
    N,
    stride_m,
    stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs = stride_m * offs_m[:, None] + stride_n * offs_n[None, :]
    mask_ = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a = tl.load(a_ptr + offs, mask=mask_)
    b = tl.load(b_ptr + offs, mask=mask_)
    tl.store(c_ptr + offs, a + b, mask=mask_)


@triton.jit
def matmul_kernel_tl(
    a_ptr,
    b_ptr,
    c_ptr,  #
    M,
    N,
    K,
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@pytest.mark.parametrize("num_stages", [1, 4])
@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_N, BLOCK_K",
    [(32, 32, 32), (128, 64, 64), (128, 128, 64), (128, 256, 64)],
)
def test_experimental_tma_matmul(num_stages, BLOCK_M, BLOCK_N, BLOCK_K):
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] == 9:
        pytest.skip("Test requires Hopper target.")
        return
    device = "cuda"
    M, N, K = 8192, 8192, 1024
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((K, N), dtype=torch.float16, device=device)
    C = torch.empty((M, N), dtype=torch.float16, device=device)
    TMA_SIZE = 128
    desc_a = np.empty(TMA_SIZE, dtype=np.int8)
    desc_b = np.empty(TMA_SIZE, dtype=np.int8)
    desc_c = np.empty(TMA_SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        A.data_ptr(), M, K, BLOCK_M, BLOCK_K, A.element_size(), desc_a
    )
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        B.data_ptr(), K, N, BLOCK_K, BLOCK_N, B.element_size(), desc_b
    )
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        C.data_ptr(), M, N, BLOCK_M, BLOCK_N, C.element_size(), desc_c
    )

    desc_a = torch.tensor(desc_a, device=device)
    desc_b = torch.tensor(desc_b, device=device)
    desc_c = torch.tensor(desc_c, device=device)
    # kernel = matmul_kernel_tma[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1,
    #                             1)](desc_a, desc_b, desc_c, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=8,
    #                                 num_stages=num_stages)
    kernel = matmul_kernel_tl[
        (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)
    ](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),  #
        B.stride(0),
        B.stride(1),  #
        C.stride(0),
        C.stride(1),  #
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps=8,
        num_stages=num_stages,
    )
    ref_out = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.float16)
    torch.testing.assert_close(ref_out, C, rtol=1e-3, atol=1e-3)
    if BLOCK_M >= 64 and BLOCK_N >= 64:
        assert "stmatrix.sync.aligned.m8n8.x4.shared.b16" in kernel.asm["ptx"]


def _test_tma():
    device = "cuda"
    M, N = 8192, 8192
    BLOCK_M, BLOCK_N = (
        128,
        64,
    )
    A = torch.randn((M, N), dtype=torch.float32, device=device)
    B = torch.randn((M, N), dtype=torch.float32, device=device)
    C = torch.empty((M, N), dtype=torch.float32, device=device)
    TMA_SIZE = 128
    desc_a = np.empty(TMA_SIZE, dtype=np.int8)
    desc_b = np.empty(TMA_SIZE, dtype=np.int8)
    desc_c = np.empty(TMA_SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        A.data_ptr(), M, N, BLOCK_M, BLOCK_N, A.element_size(), desc_a
    )
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        B.data_ptr(), M, N, BLOCK_M, BLOCK_N, B.element_size(), desc_b
    )
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        C.data_ptr(), M, N, BLOCK_M, BLOCK_N, C.element_size(), desc_c
    )

    desc_a = torch.tensor(desc_a, device=device)
    desc_b = torch.tensor(desc_b, device=device)
    desc_c = torch.tensor(desc_c, device=device)

    kernel = add_kernel_tma[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)](
        desc_a, desc_b, desc_c, M, N, BLOCK_M, BLOCK_N, num_warps=8, num_stages=1
    )
    assert torch.allclose(C, torch.add(A, B), rtol=1e-3, atol=1e-3)
    A_clone, B_clone, C_clone = A.clone(), B.clone(), C.clone()
    add_kernel_tl[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)](
        A_clone,
        B_clone,
        C_clone,
        M,
        N,
        A.stride(0),
        A.stride(1),
        BLOCK_M,
        BLOCK_N,
        num_warps=8,
        num_stages=1,
    )
    assert torch.allclose(C_clone, torch.add(A, B), rtol=1e-3, atol=1e-3)

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["op"],
            x_vals=["add"],
            line_arg="provider",
            line_vals=["tma", "tl"],
            line_names=["tma", "tl"],
            ylabel="ms",
            plot_name=f"tma()",
            args={},
        )
    )

    @triton.testing.perf_report(configs)
    def bench_fn(op, provider, device=device):
        warmup = 200
        rep = 200
        if "tma" in provider:
            fn = lambda: add_kernel_tma[
                (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)
            ](desc_a, desc_b, desc_c, M, N, BLOCK_M, BLOCK_N, num_warps=8, num_stages=1)
        if "tl" in provider:
            fn = lambda: add_kernel_tl[
                (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)
            ](
                A_clone,
                B_clone,
                C_clone,
                M,
                N,
                A.stride(0),
                A.stride(1),
                BLOCK_M,
                BLOCK_N,
                num_warps=8,
                num_stages=1,
            )
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)

    device = "cuda"
    SIZE = 81920
    BLOCK_SIZE = 256
    x = torch.randn(SIZE, dtype=torch.float32, device=device)
    desc = np.empty(SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_1d_tma_descriptor(
        x.data_ptr(), SIZE, BLOCK_SIZE, x.element_size(), desc
    )
    desc = torch.tensor(desc, device=device)
    z_tri = torch.empty_like(x)

    load_kernel_tma[(1,)](z_tri, desc, SIZE, BLOCK_SIZE, num_warps=1)
    assert torch.equal(x, z_tri)
    tl_res = torch.empty_like(x)
    load_kernel_tl[(1,)](tl_res, x, SIZE, BLOCK_SIZE, num_warps=1)
    assert torch.equal(x, tl_res)

    device = "cuda"
    M, N, K = 8192, 8192, 1024
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 64
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((K, N), dtype=torch.float16, device=device)
    C = torch.empty((M, N), dtype=torch.float16, device=device)
    TMA_SIZE = 128
    desc_a = np.empty(TMA_SIZE, dtype=np.int8)
    desc_b = np.empty(TMA_SIZE, dtype=np.int8)
    desc_c = np.empty(TMA_SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        A.data_ptr(), M, K, BLOCK_M, BLOCK_K, A.element_size(), desc_a
    )
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        B.data_ptr(), K, N, BLOCK_K, BLOCK_N, B.element_size(), desc_b
    )
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        C.data_ptr(), M, N, BLOCK_M, BLOCK_N, C.element_size(), desc_c
    )

    desc_a = torch.tensor(desc_a, device=device)
    desc_b = torch.tensor(desc_b, device=device)
    desc_c = torch.tensor(desc_c, device=device)
    kernel = matmul_kernel_tma[
        (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)
    ](
        desc_a,
        desc_b,
        desc_c,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps=8,
        num_stages=1,
    )
    A_tl = A.clone()
    B_tl = B.clone()
    C_tl = torch.empty_like(C)
    kernel = matmul_kernel_tl[
        (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)
    ](
        A_tl,
        B_tl,
        C_tl,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),  #
        B.stride(0),
        B.stride(1),  #
        C.stride(0),
        C.stride(1),  #
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps=8,
        num_stages=1,
    )
    ref_out = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.float16)
    torch.testing.assert_close(ref_out, C, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ref_out, C_tl, rtol=1e-3, atol=1e-3)
    if BLOCK_M >= 64 and BLOCK_N >= 64:
        assert "stmatrix.sync.aligned.m8n8.x4.shared.b16" in kernel.asm["ptx"]

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["op"],
            x_vals=["copy", "matmul"],
            line_arg="provider",
            line_vals=["tma", "tl"],
            line_names=["tma", "tl"],
            ylabel="ms",
            plot_name=f"tma()",
            args={},
        )
    )

    @triton.testing.perf_report(configs)
    def bench_fn(op, provider, device=device):
        warmup = 200
        rep = 200

        if "tma" in provider:
            if "copy" == op:
                fn = lambda: load_kernel_tma[(1,)](
                    z_tri, desc, SIZE, BLOCK_SIZE, num_warps=1
                )
            elif "matmul" == op:
                fn = lambda: matmul_kernel_tma[
                    (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)
                ](
                    desc_a,
                    desc_b,
                    desc_c,
                    M,
                    N,
                    K,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    num_warps=8,
                    num_stages=1,
                )

        if "tl" in provider:
            if "copy" == op:
                fn = lambda: load_kernel_tl[(1,)](
                    tl_res, x, SIZE, BLOCK_SIZE, num_warps=1
                )
            elif "matmul" == op:
                fn = lambda: matmul_kernel_tl[
                    (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)
                ](
                    A_tl,
                    B_tl,
                    C_tl,
                    M,
                    N,
                    K,
                    A.stride(0),
                    A.stride(1),  #
                    B.stride(0),
                    B.stride(1),  #
                    C.stride(0),
                    C.stride(1),  #
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    num_warps=8,
                    num_stages=1,
                )

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] == 9:
        print("Test requires Hopper target.")
        quit()
    _test_tma()
    print("sucessfully!")
