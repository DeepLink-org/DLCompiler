"""
Flash Attention Test Suite

Compares Triton attention implementations against references:
- CPU flash attention (PyTorch reference)
- NPU torch_npu.npu_fusion_attention (NPU reference)

Usage:
    pytest test_fa.py -v
    python test_fa.py  # Uses DLC_CPU_VERIFY=1 by default
"""

import os

os.environ.setdefault("DLC_CPU_VERIFY", "1")
os.environ.setdefault(
    "LLVM_BINARY_DIR",
    "/mnt/data01/kezengxiang/work/third_party/llvm-project/build_064f02dac0c81c19350a74415b3245f42fed09dc/bin",
)

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.deeplink as dl
import torch_npu

CPU_VERIFY = os.environ.get("DLC_CPU_VERIFY", "0") == "1"
DEVICE = "cpu" if CPU_VERIFY else "npu"
ATOL = 1e-3
RTOL = 0.0

# =============================================================================
# Triton Kernels
# =============================================================================


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
        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        acc_ptr = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0

        for start_n in range(0, N_CTX, BLOCK_N * NUM_STAGES):
            for i in tl.range(0, NUM_STAGES, num_stages=NUM_STAGES):
                ws1_ptr = tl.make_block_ptr(
                    base=workspace_1
                    + (NUM_STAGES * block_idx.to(tl.int64) + i) * w1_stride_nb,
                    shape=(BLOCK_M, BLOCK_N),
                    strides=(w1_stride_bm, w1_stride_bn),
                    offsets=(0, 0),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                ws2_ptr = tl.make_block_ptr(
                    base=workspace_2
                    + (NUM_STAGES * block_idx.to(tl.int64) + i) * w2_stride_nb,
                    shape=(BLOCK_M, BLOCK_N),
                    strides=(w2_stride_bm, w2_stride_bn),
                    offsets=(0, 0),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                ws3_ptr = tl.make_block_ptr(
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
                    tl.store(ws1_ptr, qk)
                    dl.set_cross_flag(dl.SyncFlag.C2V, 0)

                with dl.async_task(scope=dl.async_task.vector):
                    dl.wait_cross_flag(dl.SyncFlag.C2V, 0)
                    qk = tl.load(ws1_ptr)
                    qk = qk * sm_scale
                    m_ij = tl.maximum(m_i, tl.max(qk, 1))
                    qk = qk - m_ij[:, None]
                    p = tl.math.exp(qk)
                    p_cast = p.to(Q.type.element_ty)
                    tl.store(ws2_ptr, p_cast)
                    dl.set_cross_flag(dl.SyncFlag.V2C, 1)
                    dl.wait_cross_flag(dl.SyncFlag.V2C, 1)
                with dl.async_task(scope=dl.async_task.cube):
                    p_cast = tl.load(ws2_ptr)
                    v = tl.load(V_block_ptr)
                    acc_l0c = tl.dot(p_cast, v)
                    tl.store(ws3_ptr, acc_l0c)
                    dl.set_cross_flag(dl.SyncFlag.C2V, 2)
                    V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
                    K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
                with dl.async_task(scope=dl.async_task.vector):
                    l_ij = tl.sum(p, 1)
                    alpha = tl.math.exp(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    dl.wait_cross_flag(dl.SyncFlag.C2V, 2)
                    acc_ptr = acc_ptr * alpha[:, None]
                    acc_o_ub = tl.load(ws3_ptr)
                    acc_ptr = acc_ptr + acc_o_ub
                    m_i = m_ij

        m_i += tl.math.log(l_i)
        accumulator = acc_ptr / l_i[:, None]
        m_ptrs = M + task_hz_idx * N_CTX + offs_m

        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, accumulator.to(Out.type.element_ty))


# =============================================================================
# Python Wrappers
# =============================================================================


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


class AttentionSplitCV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, BM, BN, causal=False):
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

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

        _attn_fwd_split_cv[(NUM_CORES,)](
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
            stride_qz=q.stride(0),
            stride_qh=q.stride(1),
            stride_qm=q.stride(2),
            stride_qk=q.stride(3),
            stride_kz=k.stride(0),
            stride_kh=k.stride(1),
            stride_kn=k.stride(2),
            stride_kk=k.stride(3),
            stride_vz=v.stride(0),
            stride_vh=v.stride(1),
            stride_vn=v.stride(2),
            stride_vk=v.stride(3),
            stride_oz=o.stride(0),
            stride_oh=o.stride(1),
            stride_om=o.stride(2),
            stride_on=o.stride(3),
            w1_stride_nb=workspace_1.stride(1),
            w1_stride_bm=workspace_1.stride(2),
            w1_stride_bn=workspace_1.stride(3),
            w2_stride_nb=workspace_2.stride(1),
            w2_stride_bm=workspace_2.stride(2),
            w2_stride_bn=workspace_2.stride(3),
            w3_stride_nb=workspace_3.stride(1),
            w3_stride_bm=workspace_3.stride(2),
            w3_stride_dm=workspace_3.stride(3),
            Z=q.shape[0],
            H=q.shape[1],
            N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_M=BM,
            BLOCK_N=BN,
            NUM_CORES=NUM_CORES,
            NUM_STAGES=NUM_STAGES,
            disable_auto_inject_block_sync=True,
            disable_auto_cv_work_space_manage=True,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        return o


attention_base = _attention.apply
attention_split_cv = AttentionSplitCV.apply


# =============================================================================
# Reference Implementations
# =============================================================================


def flash_attention_cpu(q, k, v, sm_scale, causal=False):
    """Pure PyTorch CPU flash attention for reference."""
    q_fp32, k_fp32, v_fp32 = q.float(), k.float(), v.float()
    scores = torch.matmul(q_fp32, k_fp32.transpose(-2, -1)) * sm_scale

    if causal:
        seq_len = q.shape[-2]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device), diagonal=1
        ).bool()
        scores = scores.masked_fill(mask, float("-inf"))

    max_score = torch.max(scores, dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores - max_score)
    attn_weights = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)
    out = torch.matmul(attn_weights, v_fp32)

    return out.to(q.dtype)


def torch_npu_fusion_attention(q, k, v, num_heads, scale):
    """Wrapper for torch_npu.npu_fusion_attention."""
    return torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        num_heads,
        padding_mask=None,
        atten_mask=None,
        scale=scale,
        keep_prob=1.0,
        input_layout="BNSD",
        pre_tockens=65535,
        next_tockens=65535,
        sparse_mode=0,
    )[0]


# =============================================================================
# Error Metrics
# =============================================================================


def compute_error_metrics(ref, out, name):
    """Compute and print error metrics."""
    diff = (ref - out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ref_max = ref.abs().max().item()
    rel_max = max_diff / ref_max if ref_max > 0 else float("inf")
    print(
        f"  [{name}] max_abs={max_diff:.6e}, mean_abs={mean_diff:.6e}, max_rel={rel_max:.6e}"
    )
    return max_diff, mean_diff, rel_max


def compare_results(ref_cpu, ref_npu, base, cv, test_name):
    """
    Compare results according to the rules:
    - base is compared against both ref_cpu and ref_npu
    - cv is only compared against base (not against references)
    """
    print(f"\n[{test_name}] Precision Comparison:")
    print("-" * 80)

    # base vs CPU reference
    compute_error_metrics(ref_cpu, base, "base_vs_cpu_ref")

    # base vs NPU reference
    compute_error_metrics(ref_npu, base, "base_vs_npu_ref")

    # cv vs base (cv is only compared to base, not to references)
    compute_error_metrics(base, cv, "cv_vs_base")


# =============================================================================
# Test Data Generation
# =============================================================================


def generate_test_data(Z, H, N_CTX, HEAD_DIM, dtype=torch.float16):
    """Generate test tensors on CPU with fixed seed."""
    torch.manual_seed(20)
    shape = (Z, H, N_CTX, HEAD_DIM)
    q = torch.empty(shape, dtype=dtype, device="cpu").normal_(mean=0.0, std=0.5)
    k = torch.empty(shape, dtype=dtype, device="cpu").normal_(mean=0.0, std=0.5)
    v = torch.empty(shape, dtype=dtype, device="cpu").normal_(mean=0.0, std=0.5)
    return q, k, v


# =============================================================================
# Pytest Test Cases
# =============================================================================

ALL_CASES = [
    (2, 2, 1024, 128, 64, 128, False),
    (1, 1, 1024 * 4, 128, 64, 128, False),
]


@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM,BM,BN,causal", ALL_CASES)
def test_attention_precision(Z, H, N_CTX, HEAD_DIM, BM, BN, causal):
    """
    Test precision of Triton attention kernels.

    Comparison rules:
    - attention_base (_attention) is compared against:
        * CPU flash attention reference
        * NPU torch_npu.npu_fusion_attention reference
    - attention_split_cv is only compared against attention_base
      (split_cv is not compared directly to references)
    """
    q, k, v = generate_test_data(Z, H, N_CTX, HEAD_DIM)
    sm_scale = 0.5

    # CPU reference
    ref_cpu = flash_attention_cpu(q, k, v, sm_scale, causal)

    # NPU reference (runs on NPU, result moved to CPU)
    q_npu, k_npu, v_npu = q.to("npu"), k.to("npu"), v.to("npu")
    ref_npu = torch_npu_fusion_attention(q_npu, k_npu, v_npu, H, sm_scale).cpu()

    # Run Triton kernels on target device
    dev = "cpu" if CPU_VERIFY else "npu"
    q_dev, k_dev, v_dev = q.to(dev), k.to(dev), v.to(dev)

    tri_base = attention_base(q_dev, k_dev, v_dev, sm_scale, BM, BN, causal).cpu()
    tri_cv = attention_split_cv(q_dev, k_dev, v_dev, sm_scale, BM, BN, causal).cpu()

    # Compare and print results
    mode_str = "CPU_VERIFY" if CPU_VERIFY else "NPU"
    compare_results(
        ref_cpu, ref_npu, tri_base, tri_cv, f"{mode_str}_Z{Z}_H{H}_N{N_CTX}_D{HEAD_DIM}"
    )

    # Validate assertions
    assert torch.allclose(
        ref_cpu, ref_npu, atol=ATOL, rtol=RTOL
    ), "CPU ref vs NPU ref mismatch!"
    assert torch.allclose(
        ref_npu, tri_base, atol=ATOL, rtol=RTOL
    ), "base vs NPU ref mismatch!"
    assert torch.allclose(
        ref_npu, tri_cv, atol=ATOL, rtol=RTOL
    ), "cv vs NPU ref mismatch!"
    assert torch.allclose(
        tri_cv, tri_base, atol=ATOL, rtol=RTOL
    ), "base vs cv ref mismatch!"


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
