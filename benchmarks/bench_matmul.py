
import triton
import triton.language as tl
import torch
import triton.backends.dicp_triton.driver as dicp
import triton.language.extra.deeplink as dl
import pytest
triton.runtime.driver.set_active(dicp.DICPDriver('ascend'))

DEV = "npu"
activation = "leaky_relu_custom"


def get_autotune_config():
    return [
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    ACTIVATION: tl.constexpr,  #
    VEC_PARALLEL: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    GROUP_SIZE_M: tl.constexpr = 1
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs_base = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs_base = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    msk_m = offs_am < M
    msk_n = offs_bn < N
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a_ptrs = a_ptrs_base + k * BLOCK_SIZE_K * stride_ak
        b_ptrs = b_ptrs_base + k * BLOCK_SIZE_K * stride_bk
        a = tl.load(
            a_ptrs,
            mask=msk_m[:, None] and (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=msk_n[None, :] and (offs_k[:, None] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)

    if VEC_PARALLEL == 1:
        if ACTIVATION == "leaky_relu_custom":
            accumulator = leaky_relu_custom(accumulator)
        c = accumulator.to(tl.float16)
        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        SUB_BLK_M: tl.constexpr = BLOCK_SIZE_M // 2
        for s in dl.parallel(0, 2, bind_sub_block=True):
            vec_sub_blk = dl.extract_slice(
                accumulator, (s * SUB_BLK_M, 0), (SUB_BLK_M, BLOCK_SIZE_N), (1, 1)
            )
            if ACTIVATION == "leaky_relu_custom":
                vec_sub_blk = leaky_relu_custom(vec_sub_blk)
            c_sub_blk = vec_sub_blk.to(tl.float16)
            # -----------------------------------------------------------
            # Write back the block of the output matrix C with masks.
            offs_cm = pid_m * BLOCK_SIZE_M + s * SUB_BLK_M + tl.arange(0, SUB_BLK_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, c_sub_blk, mask=c_mask)


# We can fuse `leaky_relu_custom` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu_custom(x):
    return tl.where(x >= 0, x, 0.01 * x) + 1.0


def torch_matmul(a, b, activation=""):
    c = torch.matmul(a, b)
    if activation == "leaky_relu_custom":
        c = torch.where(c >= 0, c, 0.01 * c) + 1.0
    return c


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def triton_matmul(a, b, activation="", vec_parallel=2):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        ACTIVATION=activation,  #
        VEC_PARALLEL=vec_parallel,
    )
    return c


def main():
    torch.npu.set_device(1)
    torch.manual_seed(0)
    a = torch.randn((512, 512), device=DEV, dtype=torch.float16)
    b = torch.randn((512, 512), device=DEV, dtype=torch.float16)
    triton_output = triton_matmul(a, b, activation)
    torch_output = torch_matmul(a, b, activation)
    torch.testing.assert_close(triton_output, torch_output, rtol=0.01, atol=1e-03)


    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 6)],  # Different possible values for `x_name`
            line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=['triton_no_parallel','triton-parallel'],  # Label name for the lines
            line_names=['triton_no_parallel', 'triton-parallel'],  # Line styles
            styles=[('green', '-'), ('blue', '-')],
            ylabel='TFLOPS',  # Label name for the y-axis
            plot_name='matmul-performance-fp16',
            args={},
        ))

    @triton.testing.perf_report(configs)
    def benchmark(M, N, K, provider):
        warmup = 500
        rep = 500
        a = torch.randn((M, K), device=DEV, dtype=torch.float16)
        b = torch.randn((K, N), device=DEV, dtype=torch.float16)
        
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'triton_no_parallel':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_matmul(a, b, activation, 1),
                                                         quantiles=quantiles,
                                                         warmup=warmup,
                                                         rep=rep)
        if provider == 'triton-parallel':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_matmul(a, b, activation, 2),
                                                         quantiles=quantiles,
                                                         warmup=warmup,
                                                         rep=rep)
        return ms, max_ms, min_ms

    benchmark.run(show_plots=False, print_data=True)


if __name__ == '__main__':
    main()
