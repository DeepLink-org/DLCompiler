import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.deeplink as dl
import torch_npu
import triton.testing

DEVICE = "npu"


def require_npu():
    try:
        torch.empty(1, device=DEVICE)
    except Exception:
        pytest.skip("npu device not available")


# ------------------- Triton kernels (kept same functional implementation) -------------------
@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    start_m,
    qk_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            qk = qk * qk_scale
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

        p = tl.math.exp(qk)
        p_cast = p.to(tl.float16)
        v = tl.load(V_block_ptr)
        pv = tl.dot(p_cast, v)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + pv

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
    sm_scale: tl.constexpr,
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    Z: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    NUM_BLOCKS_PER_CORE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_BLOCKS_M: tl.constexpr,
):
    pid = tl.program_id(0)
    for block_idx in range(pid, NUM_BLOCKS, 24):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
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

        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        q = tl.load(Q_block_ptr)

        if STAGE & 1:
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,
                task_m_idx,
                sm_scale,
                BLOCK_M,
                HEAD_DIM,
                BLOCK_N,
                4 - STAGE,
                offs_m,
                offs_n,
                N_CTX,
                V.dtype.element_ty == tl.float8e5,
            )

        if STAGE & 2:
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,
                task_m_idx,
                sm_scale,
                BLOCK_M,
                HEAD_DIM,
                BLOCK_N,
                2,
                offs_m,
                offs_n,
                N_CTX,
                V.dtype.element_ty == tl.float8e5,
            )

        m_i += tl.math.log(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + task_hz_idx * N_CTX + offs_m

        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.jit
def _attn_fwd_split_cv(
    Q,
    K,
    V,
    M,
    Out,
    acc,
    sm_scale,
    workspace_1,
    workspace_2,
    workspace_3,
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    w1_stride_nb: tl.constexpr,
    w1_stride_bm: tl.constexpr,
    w1_stride_bn: tl.constexpr,
    w2_stride_nb: tl.constexpr,
    w2_stride_bm: tl.constexpr,
    w2_stride_bn: tl.constexpr,
    w3_stride_nb: tl.constexpr,
    w3_stride_bm: tl.constexpr,
    w3_stride_dm: tl.constexpr,
    Z: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_CORES: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H
    pid = tl.program_id(0)
    for block_idx in tl.range(pid, NUM_BLOCKS, NUM_CORES):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
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

        q = tl.load(Q_block_ptr)
        K_block_ptr = tl.advance(K_block_ptr, (0, 0))
        V_block_ptr = tl.advance(V_block_ptr, (0, 0))
        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        acc_ptr = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0

        lo, hi = 0, N_CTX
        for start_n in range(lo, hi, BLOCK_N * NUM_STAGES):
            for i in tl.range(0, NUM_STAGES, num_stages=NUM_STAGES):
                workspace_1_ptr = tl.make_block_ptr(
                    base=workspace_1
                    + (NUM_STAGES * block_idx.to(tl.int64) + i) * w1_stride_nb,
                    shape=(BLOCK_M, BLOCK_N),
                    strides=(w1_stride_bm, w1_stride_bn),
                    offsets=(0, 0),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                workspace_2_ptr = tl.make_block_ptr(
                    base=workspace_2
                    + (NUM_STAGES * block_idx.to(tl.int64) + i) * w2_stride_nb,
                    shape=(BLOCK_M, BLOCK_N),
                    strides=(w2_stride_bm, w2_stride_bn),
                    offsets=(0, 0),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                workspace_3_ptr = tl.make_block_ptr(
                    base=workspace_3
                    + (NUM_STAGES * block_idx.to(tl.int64) + i) * w3_stride_nb,
                    shape=(BLOCK_M, HEAD_DIM),
                    strides=(w3_stride_bm, w3_stride_dm),
                    offsets=(0, 0),
                    block_shape=(BLOCK_M, HEAD_DIM),
                    order=(1, 0),
                )
                with dl.async_task(scope=dl.async_task.cube):
                    k = tl.load(K_block_ptr)
                    trans_k = tl.trans(k)
                    qk = tl.dot(q, trans_k)
                    tl.store(workspace_1_ptr, qk)

                    dl.set_cross_flag(dl.SyncFlag.C2V, 0)
                    dl.wait_cross_flag(dl.SyncFlag.V2C, 1)

                    p_cast = tl.load(workspace_2_ptr)
                    v = tl.load(V_block_ptr)
                    acc_l0c = tl.dot(p_cast, v)
                    tl.store(workspace_3_ptr, acc_l0c)
                    dl.set_cross_flag(dl.SyncFlag.C2V, 2)
                    V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
                    K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))

                with dl.async_task(scope=dl.async_task.vector):
                    dl.wait_cross_flag(dl.SyncFlag.C2V, 0)

                    qk = tl.load(workspace_1_ptr)
                    qk = qk * sm_scale
                    m_ij = tl.maximum(m_i, tl.max(qk, 1))
                    qk = qk - m_ij[:, None]
                    p = tl.math.exp(qk)
                    p_cast = p.to(Q.type.element_ty)
                    tl.store(workspace_2_ptr, p_cast)
                    dl.set_cross_flag(dl.SyncFlag.V2C, 1)
                    l_ij = tl.sum(p, 1)
                    alpha = tl.math.exp(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    dl.wait_cross_flag(dl.SyncFlag.C2V, 2)
                    acc_ptr = acc_ptr * alpha[:, None]
                    acc_o_ub = tl.load(workspace_3_ptr)
                    acc_ptr = acc_ptr + acc_o_ub
                    m_i = m_ij

        m_i += tl.math.log(l_i)
        accumulator = acc_ptr / l_i[:, None]
        m_ptrs = M + task_hz_idx * N_CTX + offs_m

        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, accumulator.to(Out.type.element_ty))


# ------------------- Python wrappers and Function classes -------------------
class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, BM, BN, causal=False):
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        o = torch.empty_like(q)
        stage = 3 if causal else 1
        num_cores = 24
        NUM_BLOCKS_M = triton.cdiv(q.shape[2], BM)
        NUM_BLOCKS = NUM_BLOCKS_M * q.shape[0] * q.shape[1]
        NUM_BLOCKS_PER_CORE = triton.cdiv(NUM_BLOCKS, num_cores)

        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        _attn_fwd[(num_cores,)](
            q,
            k,
            v,
            M,
            o,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            q.shape[0],
            q.shape[1],
            N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_M=BM,
            BLOCK_N=BN,
            STAGE=stage,
            NUM_BLOCKS_PER_CORE=NUM_BLOCKS_PER_CORE,
            NUM_BLOCKS=NUM_BLOCKS,
            NUM_BLOCKS_M=NUM_BLOCKS_M,
            multibuffer=True,
            unit_flag=True,
            debug=False,
        )
        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o


@triton.jit
def _attn_fwd_split_cv_launcher(
    Q,
    K,
    V,
    M,
    o,
    acc,
    sm_scale,
    workspace_1,
    workspace_2,
    workspace_3,
    q_stride0,
    q_stride1,
    q_stride2,
    q_stride3,
    k_stride0,
    k_stride1,
    k_stride2,
    k_stride3,
    v_stride0,
    v_stride1,
    v_stride2,
    v_stride3,
    o_stride0,
    o_stride1,
    o_stride2,
    o_stride3,
    w1_nb,
    w1_bm,
    w1_bn,
    w2_nb,
    w2_bm,
    w2_bn,
    w3_nb,
    w3_bm,
    w3_dm,
    Z: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_CORES: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    _attn_fwd_split_cv(
        Q,
        K,
        V,
        M,
        o,
        acc,
        sm_scale,
        workspace_1,
        workspace_2,
        workspace_3,
        q_stride0,
        q_stride1,
        q_stride2,
        q_stride3,
        k_stride0,
        k_stride1,
        k_stride2,
        k_stride3,
        v_stride0,
        v_stride1,
        v_stride2,
        v_stride3,
        o_stride0,
        o_stride1,
        o_stride2,
        o_stride3,
        w1_nb,
        w1_bm,
        w1_bn,
        w2_nb,
        w2_bm,
        w2_bn,
        w3_nb,
        w3_bm,
        w3_dm,
        Z=Z,
        H=H,
        N_CTX=N_CTX,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        NUM_CORES=NUM_CORES,
        NUM_STAGES=NUM_STAGES,
    )


class AttentionSplitCV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, BM, BN, causal=False):
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        extra_kern_args = {}

        o = torch.empty_like(q)
        N_CTX = q.shape[2]
        Z, H = q.shape[0], q.shape[1]
        NUM_BLOCKS_M = N_CTX // BM
        NUM_BLOCKS = NUM_BLOCKS_M * Z * H
        DIM = q.shape[-1]
        NUM_CORES = 24
        NUM_STAGES = 4
        acc = torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2], HEAD_DIM_K),
            dtype=torch.float32,
            device=q.device,
        )
        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        workspace_1 = torch.empty(
            (NUM_STAGES, NUM_BLOCKS, BM, BN), device=q.device, dtype=torch.float32
        )
        workspace_2 = torch.empty(
            (NUM_STAGES, NUM_BLOCKS, BM, BN), device=q.device, dtype=q.dtype
        )
        workspace_3 = torch.empty(
            (NUM_STAGES, NUM_BLOCKS, BM, DIM), device=q.device, dtype=torch.float32
        )

        _attn_fwd_split_cv_launcher[(NUM_CORES,)](
            q,
            k,
            v,
            M,
            o,
            acc,
            sm_scale,
            workspace_1,
            workspace_2,
            workspace_3,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            workspace_1.stride(1),
            workspace_1.stride(2),
            workspace_1.stride(3),
            workspace_2.stride(1),
            workspace_2.stride(2),
            workspace_2.stride(3),
            workspace_3.stride(1),
            workspace_3.stride(2),
            workspace_3.stride(3),
            q.shape[0],
            q.shape[1],
            N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_M=BM,
            BLOCK_N=BN,
            NUM_CORES=NUM_CORES,
            NUM_STAGES=NUM_STAGES,
            disable_auto_inject_block_sync=True,
            disable_auto_cv_work_space_manage=True,
            **extra_kern_args,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        return o


attention_base = _attention.apply
attention_split_cv = AttentionSplitCV.apply

# ------------------- Tests (expanded to include all original test_op cases) -------------------
ALL_CASES = [
    (1, 2, 2048, 128, 64, 128, False),
    (4, 32, 1024, 64, 64, 256, False),
    # 超长序列
    (1, 1, 1024 * 32, 128, 64, 128, False),
    # 中等规模
    (4, 4, 512, 128, 16, 128, False),
    (8, 32, 512, 256, 16, 128, False),
    # 小序列 / tile 多样性
    (32, 32, 64, 64, 64, 16, False),
    (32, 32, 128, 128, 64, 32, False),
    (32, 32, 256, 128, 64, 64, False),
    # 常见 LLM 配置
    (1, 8, 1024, 64, 64, 128, False),
    (8, 12, 512, 64, 128, 128, False),
    # 长上下文
    (1, 16, 2048, 128, 64, 128, False),
    (1, 32, 4096, 128, 64, 128, False),
]


@pytest.mark.xdist_group(name="attention_ref_group")
@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM,BM,BN,causal", ALL_CASES)
def test_attention_matches_reference_all(Z, H, N_CTX, HEAD_DIM, BM, BN, causal):
    require_npu()
    torch.manual_seed(20)
    dtype = torch.float16

    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(
        mean=0.0, std=0.5
    )
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(
        mean=0.0, std=0.5
    )
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(
        mean=0.0, std=0.5
    )

    sm_scale = 0.5

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

    tri_out_base = attention_base(q, k, v, sm_scale, BM, BN, causal).half()
    tri_out_cv = attention_split_cv(q, k, v, sm_scale, BM, BN, causal).half()

    atol = 1e-3
    rtol = 0.0

    assert torch.allclose(ref_out, tri_out_base, atol=atol, rtol=rtol)
    assert torch.allclose(ref_out, tri_out_cv, atol=atol, rtol=rtol)


# # Performance test for the ultra-long sequence; asserts  torch_time < tri_time* 0.30
# @pytest.mark.xdist_group(name="test_perf_long_sequence")
# def test_perf_long_sequence():
#     require_npu()
#     torch.manual_seed(20)
#     Z, H, N_CTX, HEAD_DIM, BM, BN, causal = (1, 1, 1024 * 64, 128, 64, 256, False)
#     dtype = torch.float16

#     q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(
#         mean=0.0, std=0.5
#     )
#     k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(
#         mean=0.0, std=0.5
#     )
#     v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(
#         mean=0.0, std=0.5
#     )

#     sm_scale = 0.5

#     # Warmup & rep counts: keep modest to avoid extremely long CI runs; adjust as needed.
#     warmup = 50
#     rep = 50

#     # measure torch_npu fused attention
#     torch_fn = lambda: torch_npu.npu_fusion_attention(
#         q,
#         k,
#         v,
#         H,
#         padding_mask=None,
#         atten_mask=None,
#         scale=sm_scale,
#         keep_prob=1.0,
#         input_layout="BNSD",
#         pre_tockens=65535,
#         next_tockens=65535,
#         sparse_mode=0,
#     )[0]

#     tri_fn = lambda: attention_split_cv(q, k, v, sm_scale, BM, BN, causal)

#     # triton.testing.do_bench returns ms
#     torch_ms = triton.testing.do_bench(torch_fn, warmup=warmup, rep=rep)
#     tri_ms = triton.testing.do_bench(tri_fn, warmup=warmup, rep=rep)

#     # print for visibility when running tests
#     print(
#         f"torch_npu fusion ms: {torch_ms:.3f} ms; triton split_cv ms: {tri_ms:.3f} ms"
#     )

#     # require triton to be faster than 30% of torch time
#     assert (
#         torch_ms > tri_ms * 0.30
#     ), f"triton({torch_ms :.3f}ms) must be < 30% of triton({tri_ms:.3f}ms)"
