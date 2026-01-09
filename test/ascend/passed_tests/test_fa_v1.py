"""
i
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch
import torch_npu
import triton
import triton.language as tl

DEVICE = "npu"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    start_m,
    qk_scale: tl.constexpr,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
):
    # range of values handled by this stage
    # causal = true
    # stage = 1
    # 因果注意力，顾名思义，它在计算时会限制信息的流动，只允许模型看到当前位置及之前的位置
    # 的信息。也就是说，当前位置的输出只能依赖于该位置及其之前的输入，而不能访问当前位置
    # 之后的信息。因果注意力保证了数据的顺序性，避免了“未来信息”的泄露。
    # 但是后面的逻辑也会触发
    if STAGE == 1:
        # tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    # k 之前的版本，随路做转置的版本
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))

    # 修改后不转的版本
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)

        # 修改K
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)
        # ------------------------------

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            qk = qk * qk_scale
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

        # p = tl.math.exp2(qk)
        p = tl.math.exp(qk)
        p_cast = p.to(tl.float16)
        v = tl.load(V_block_ptr)
        # [bm, head_dim] * [bn, head_dim].transpose
        pv = tl.dot(p_cast, v)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp(m_i - m_ij)
        # alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None] + pv

        # m_i = m_ij * qk_scale / 0.72134752
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    M,
    Out,
    sm_scale: tl.constexpr,  #
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,  #
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,  #
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vk: tl.constexpr,  #
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,  #
    Z: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    NUM_BLOCKS_PER_CORE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_BLOCKS_M: tl.constexpr,
    # NUM_BLOCKS_N: tl.constexpr,
):
    # ???, why
    # tl.static_assert(BLOCK_N <= HEAD_DIM)
    pid = tl.program_id(0)
    block_start = pid * NUM_BLOCKS_PER_CORE
    NUM_BLOCKS_hz = NUM_BLOCKS // NUM_BLOCKS_M
    task_m_idx = 0
    task_hz_idx = 0

    for block_idx in range(pid, NUM_BLOCKS, 20):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            # doesn't matter
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            # doesn't matter
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            # why block_n??
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        # initialize offsets
        # qk_sclae = sm_scale * 1.4
        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        # load scales

        # load q: it will stay in SRAM throughout
        q = tl.load(Q_block_ptr)
        # stage 1: off-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
        # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE

        if STAGE & 1:
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,  #
                task_m_idx,
                sm_scale,  #
                BLOCK_M,
                HEAD_DIM,
                BLOCK_N,  #
                4 - STAGE,
                offs_m,
                offs_n,
                N_CTX,
                V.dtype.element_ty == tl.float8e5,  #
            )
        # stage 2: on-band

        if STAGE & 2:
            # barrier makes it easier for compielr to schedule the
            # two loops independently
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,  #
                task_m_idx,
                sm_scale,  #
                BLOCK_M,
                HEAD_DIM,
                BLOCK_N,  #
                2,
                offs_m,
                offs_n,
                N_CTX,
                V.dtype.element_ty == tl.float8e5,  #
            )
        # epilogue
        # m_i += tl.math.log2(l_i)
        m_i += tl.math.log(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + task_hz_idx * N_CTX + offs_m

        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, BM, BN):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        o = torch.empty_like(q)

        # stage = 3
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        # if is_hip():
        #     waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
        #     extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}
        num_cores = 20
        NUM_BLOCKS_M = triton.cdiv(q.shape[2], BM)
        NUM_BLOCKS = NUM_BLOCKS_M * q.shape[0] * q.shape[1]
        NUM_BLOCKS_PER_CORE = triton.cdiv(NUM_BLOCKS, num_cores)
        # grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        # grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        # (1, 2, 1024)
        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        _attn_fwd[(num_cores,)](
            q,
            k,
            v,
            M,
            o,
            sm_scale,  #
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),  #
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),  #
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),  #
            q.shape[0],
            q.shape[1],
            N_CTX=q.shape[2],  # why varidic??
            HEAD_DIM=HEAD_DIM_K,  # 64
            BLOCK_M=BM,  # 32
            BLOCK_N=BN,  # 32
            STAGE=stage,
            NUM_BLOCKS_PER_CORE=NUM_BLOCKS_PER_CORE,
            NUM_BLOCKS=NUM_BLOCKS,
            NUM_BLOCKS_M=NUM_BLOCKS_M,
            debug=True,
            multibuffer=True,  # 控制开double_buffer
            unit_flag=True,  # cube搬出的一个优化项
            limit_auto_multi_buffer_only_for_local_buffer=False,
            set_workspace_multibuffer=4,
            tile_mix_vector_loop=4,
            tile_mix_cube_loop=4,
            **extra_kern_args
        )

        ctx.save_for_backward(q, k, v, o, M)
        # ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o


import math

attention = _attention.apply


@pytest.mark.parametrize(
    "Z,H,N_CTX,HEAD_DIM,causal,dtype,BM,BN",
    [
        (1, 2, 1024, 64, False, torch.float16, 64, 32),
        (1, 1, 64, 64, False, torch.float16, 64, 16),
        (4, 32, 64, 64, False, torch.float16, 64, 32),
        (4, 32, 64, 64, False, torch.float16, 64, 16),
        (1, 1, 128, 128, False, torch.float16, 16, 16),
        (1, 1, 128, 128, False, torch.float16, 64, 64),
        # (4, 32, 128, 128, False, torch.float16, 32, 64),
        # (1, 2, 1024, 64, False, torch.float16, 128, 32),
    ],
)
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype, BM, BN):
    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    sm_scale = 0.5

    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    faq = q * torch.sqrt(torch.tensor(HEAD_DIM, dtype=torch.float32)) * sm_scale

    shape_str = "x".join(map(str, (Z, H, N_CTX, HEAD_DIM)))

    stream = torch.npu.current_stream()
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )

    torch_path = "./FA_FWD_PROF/" + shape_str + "/" + str(causal) + "/torch/"
    LOOP = 20
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=1, active=LOOP, repeat=1, skip_first=1
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:

        stream.synchronize()
        prof.step()
        for i in range(LOOP + 2):
            ref_out = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                H,
                padding_mask=None,
                atten_mask=None,
                scale=sm_scale,
                keep_prob=1.0,
                input_layout="BNSD",
                pre_tockens=65535,
                next_tockens=65535,
                sparse_mode=0,
            )[0]
            prof.step()
            if i == 0:
                stream.synchronize()
        stream.synchronize()

    tilling_str = "x".join(map(str, (BM, BN)))
    torch_path = (
        "./FA_FWD_PROF/" + shape_str + "/" + str(causal) + "/triton/" + tilling_str
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=1, active=LOOP, repeat=1, skip_first=1
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:

        stream.synchronize()
        prof.step()
        for i in range(LOOP + 2):
            tri_out = attention(q, k, v, causal, sm_scale, BM, BN).half()
            prof.step()
            if i == 0:
                stream.synchronize()
        stream.synchronize()
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_out, tri_out, atol=atol, rtol=rtol)

    # 增加性能测试
    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["op"],
            x_vals=["fwd"],
            line_arg="provider",
            line_vals=["triton", "pytorch"],
            line_names=["Triton", "PyTorch"],
            ylabel="ms",
            plot_name="",
            args={},
        )
    )

    @triton.testing.perf_report(configs)
    def bench_fn(op, provider):
        warmup = 500
        rep = 500

        if "triton" in provider:
            ms = triton.testing.do_bench(
                lambda: attention(q, k, v, causal, sm_scale, BM, BN),
                warmup=warmup,
                rep=rep,
            )
        if "pytorch" in provider:
            ms = triton.testing.do_bench(
                lambda: torch_npu.npu_fusion_attention(
                    q,
                    k,
                    v,
                    H,
                    padding_mask=None,
                    atten_mask=None,
                    scale=sm_scale,
                    keep_prob=1.0,
                    input_layout="BNSD",
                    pre_tockens=65535,
                    next_tockens=65535,
                    sparse_mode=0,
                )[0],
                warmup=warmup,
                rep=rep,
            )
        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    # test_op(1,8,8192,128, causal=True, dtype=torch.float16, BM = 32,BN = 32)
    test_op(1, 2, 2048, 64, causal=False, dtype=torch.float16, BM=128, BN=512)
    # test_op(4,32,1024,64, causal=False, dtype=torch.float16, BM = 64,BN = 256)
    # test_op(4,32,4096,64, causal=False, dtype=torch.float16, BM = 64,BN = 256)
    # test_op(4,32,8192,64, causal=False, dtype=torch.float16, BM =64,BN = 256)
    # test_op(4,32,16384,64, causal=False, dtype=torch.float16, BM=64,BN = 256)
