import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.deeplink as dl

DEVICE = "npu"


@triton.jit
def _attn_fwd_base_inner(
    acc_ptr,
    l_i,
    m_i,
    q,  # Accumulator, local l, local m, query vector
    K_block_ptr,
    V_block_ptr,  # Key and value block pointers for current stage
    start_m,
    qk_scale,  # Starting position of current query block, qk scale factor
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  # Block size constants
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  # Current stage flag, m and n offset indices
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
):  # Total context length, whether to enable FP8 for value precision
    # Set the processing range [lo, hi) for the current stage (in column block units)
    # causal = true
    # stage = 1
    # Causal attention, as the name implies, restricts the flow of information during computation,
    # only allowing the model to see the current and previous positions.
    # In other words, the output at the current position can only depend on the input at or before this position,
    # and cannot access information from future positions.
    # Causal attention ensures sequential order and prevents "leakage of future information."
    # But the following logic will also be triggered
    if STAGE == 1:
        # Stage 1: process all tokens before the query block
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # Stage 2: process the current query block
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)  # Align starting position
    # causal = False (no need for masking)
    else:
        lo, hi = 0, N_CTX  # Process the entire context

    # Adjust K and V block pointers to the starting position `lo`
    K_block_ptr = tl.advance(
        K_block_ptr, (lo, 0)
    )  # K is [HEAD_DIM, N_CTX], shift along the second dim by lo
    V_block_ptr = tl.advance(
        V_block_ptr, (lo, 0)
    )  # V is [N_CTX, HEAD_DIM], shift along the first dim by lo

    # Index mapping for the accumulator , used for slicing when HEAD_DIM >= 256
    row = tl.arange(0, BLOCK_M)[:, None]
    col_head_dim = tl.arange(0, HEAD_DIM)[None, :]
    block2d_acc = row * HEAD_DIM + col_head_dim

    # Iterate over all k, v blocks in the current stage and accumulate the output
    for start_n in range(lo, hi, BLOCK_N):  # Process BLOCK_N columns at a time
        start_n = tl.multiple_of(start_n, BLOCK_N)  # Align column start position
        # -- Compute qk ----
        k = tl.load(K_block_ptr)
        # Modify K
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)
        # Apply causal mask for STAGE 2
        if STAGE == 2:
            mask = offs_m[:, None] >= (
                start_n + offs_n[None, :]
            )  # Construct upper triangular mask
            qk = qk * qk_scale + tl.where(
                mask, 0, -1.0e6
            )  # Set invalid positions to -∞
            m_ij = tl.maximum(m_i, tl.max(qk, 1))  # Update m_ij = max(m_i, max(qk))
            qk -= m_ij[:, None]  # Subtract max for softmax stability
        else:
            qk = qk * qk_scale
            m_ij = tl.maximum(m_i, tl.max(qk, 1))  # Scaled max
            qk = qk - m_ij[:, None]  # Stabilize

        # Softmax weights p = exp(qk)
        p = tl.math.exp(qk)

        # Convert softmax weight type depending on FP8 usage
        if fp8_v:
            p_cast = p.to(tl.float8e5)  # Convert to FP8 format (save memory)
        else:
            p_cast = p.to(k.dtype)

        v = tl.load(V_block_ptr)  # Load corresponding V block
        pv = tl.dot(p_cast, v)
        l_ij = tl.sum(p, 1)  # Softmax denominator (sum of each row)
        # -- Update m_i and l_i
        alpha = tl.math.exp(
            m_i - m_ij
        )  # Update factor: exp difference between old and new max
        l_i = l_i * alpha + l_ij  # Update softmax denominator
        # -- Update output accumulator --
        acc_ptr = acc_ptr * alpha[:, None]
        acc_ptr = tl.dot(p_cast, v, acc_ptr)

        m_i = m_ij  # Update current block max
        # Advance V and K block pointers to next BLOCK_N range
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    # Return accumulated output acc_ptr, softmax denominator l_i, and max value m_i
    return acc_ptr, l_i, m_i


@triton.jit
def _attn_fwd_base(
    Q,
    K,
    V,
    M,
    Out,
    acc,
    sm_scale,
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
):
    # Total number of blocks in sequence dimension (M)
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    # Total tasks = number of sequence blocks × batch size (Z) × number of attention heads (H)
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    # Current M-dimension block index
    pid = tl.program_id(0)

    for block_idx in range(pid, NUM_BLOCKS, 20):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        # Create block pointers for Q, K, V, Output
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
        # Initialize offsets
        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0

        # Initialize accumulator
        acc_ptr = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        # load q: it will stay in SRAM throughout
        q = tl.load(Q_block_ptr)

        # stage 1: off-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
        # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
        if STAGE & 1:
            acc_ptr, l_i, m_i = _attn_fwd_base_inner(
                acc_ptr,
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
            acc_ptr, l_i, m_i = _attn_fwd_base_inner(
                acc_ptr,
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
        m_i += tl.math.log(l_i)
        accumulator = acc_ptr / l_i[:, None]
        m_ptrs = M + task_hz_idx * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, accumulator.to(Out.type.element_ty))


class AttentionBase(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, BM, BN):
        """
        Forward computation interface:
        Args:
            ctx: Context object
            q: Query tensor (Q), shape [Z, H, N_CTX, HEAD_DIM]
            k: Key tensor (K), shape [Z, H, N_CTX, HEAD_DIM]
            v: Value tensor (V), shape [Z, H, N_CTX, HEAD_DIM]
            sm_scale: Scaling factor for QK product
            BM: Q block size (BLOCK_M)
            BN: K/V block size (BLOCK_N)
        Returns:
            o: Attention output tensor, shape [Z, H, N_CTX, HEAD_DIM]
        """
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        extra_kern_args = {}
        # Number of NPU cores (adjust based on hardware)
        num_cores = 20
        acc = torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2], HEAD_DIM_K),
            dtype=torch.float32,
            device=q.device,
        )
        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        _attn_fwd_base[(num_cores,)](
            q,
            k,
            v,
            M,
            o,
            acc,
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
            STAGE=1,
            **extra_kern_args,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        return o


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
):
    # Total number of blocks in sequence dimension (M)
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    # Total tasks = number of sequence blocks × batch size (Z) × number of attention heads (H)
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H
    # Current M-dimension block index
    pid = tl.program_id(0)
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        # Create block pointers for Q, K, V, Output
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
        workspace_1_ptr = tl.make_block_ptr(
            base=workspace_1 + block_idx.to(tl.int64) * w1_stride_nb,
            shape=(BLOCK_M, BLOCK_N),
            strides=(w1_stride_bm, w1_stride_bn),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        workspace_2_ptr = tl.make_block_ptr(
            base=workspace_2 + block_idx.to(tl.int64) * w2_stride_nb,
            shape=(BLOCK_M, BLOCK_N),
            strides=(w2_stride_bm, w2_stride_bn),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        workspace_3_ptr = tl.make_block_ptr(
            base=workspace_3 + block_idx.to(tl.int64) * w3_stride_nb,
            shape=(BLOCK_M, HEAD_DIM),
            strides=(w3_stride_bm, w3_stride_dm),
            offsets=(0, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        with dl.async_task(scope=dl.async_task.cube):
            q = tl.load(Q_block_ptr)
            lo, hi = 0, N_CTX  # Process the entire context
            K_block_ptr = tl.advance(
                K_block_ptr, (lo, 0)
            )  # K is [HEAD_DIM, N_CTX], shift along the second dim by lo
            V_block_ptr = tl.advance(
                V_block_ptr, (lo, 0)
            )  # V is [N_CTX, HEAD_DIM], shift along the first dim by lo
            for start_n in range(lo, hi, BLOCK_N):  # Process BLOCK_N columns at a time
                start_n = tl.multiple_of(
                    start_n, BLOCK_N
                )  # Align column start position
                k = tl.load(K_block_ptr)
                trans_k = tl.trans(k)
                qk = tl.dot(q, trans_k)  # shape: [BLOCK_M, BLOCK_N]
                tl.store(workspace_1_ptr, qk)

                dl.set_cross_flag(dl.SyncFlag.C2V, 0)
                dl.wait_cross_flag(dl.SyncFlag.V2C, 1)

                p_cast = tl.load(workspace_2_ptr)  # shape: [BLOCK_M, BLOCK_N]
                v = tl.load(V_block_ptr)  # shape: [BLOCK_N, HEAD_DIM]
                acc_l0c = tl.dot(p_cast, v)  # shape: [BLOCK_M, HEAD_DIM]
                tl.store(workspace_3_ptr, acc_l0c)
                dl.set_cross_flag(dl.SyncFlag.C2V, 2)
                # dl.wait_cross_flag(dl.SyncFlag.V2C, 3)
                V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
                K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))

        with dl.async_task(scope=dl.async_task.vector):
            offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
            m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
            acc_ptr = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
            lo, hi = 0, N_CTX  # Process the entire context
            for start_n in range(lo, hi, BLOCK_N):  # Process BLOCK_N columns at a time
                start_n = tl.multiple_of(
                    start_n, BLOCK_N
                )  # Align column start position
                dl.wait_cross_flag(dl.SyncFlag.C2V, 0)

                qk = tl.load(workspace_1_ptr)
                qk = qk * sm_scale
                m_ij = tl.maximum(m_i, tl.max(qk, 1))  # Scaled max
                qk = qk - m_ij[:, None]  # Stabilize
                p = tl.math.exp(qk)
                p_cast = p.to(Q.type.element_ty)
                tl.store(workspace_2_ptr, p_cast)
                dl.set_cross_flag(dl.SyncFlag.V2C, 1)
                l_ij = tl.sum(p, 1)  # Softmax denominator (sum of each row)
                alpha = tl.math.exp(
                    m_i - m_ij
                )  # Update factor: exp difference between old and new max
                l_i = l_i * alpha + l_ij  # Update softmax denominator
                acc_ptr = acc_ptr * alpha[:, None]
                dl.wait_cross_flag(dl.SyncFlag.C2V, 2)
                acc_o_ub = tl.load(workspace_3_ptr)
                acc_ptr = acc_ptr + acc_o_ub
                m_i = m_ij  # Update current block max

            m_i += tl.math.log(l_i)
            accumulator = acc_ptr / l_i[:, None]
            m_ptrs = M + task_hz_idx * N_CTX + offs_m

            tl.store(m_ptrs, m_i)
            tl.store(O_block_ptr, accumulator.to(Out.type.element_ty))


class AttentionSplitCV(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, BM, BN):
        """
        Forward computation interface:
        Args:
            ctx: Context object
            q: Query tensor (Q), shape [Z, H, N_CTX, HEAD_DIM]
            k: Key tensor (K), shape [Z, H, N_CTX, HEAD_DIM]
            v: Value tensor (V), shape [Z, H, N_CTX, HEAD_DIM]
            sm_scale: Scaling factor for QK product
            BM: Q block size (BLOCK_M)
            BN: K/V block size (BLOCK_N)
        Returns:
            o: Attention output tensor, shape [Z, H, N_CTX, HEAD_DIM]
        """
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        o = torch.empty_like(q)
        extra_kern_args = {}

        N_CTX = q.shape[2]
        Z, H = q.shape[0], q.shape[1]
        NUM_BLOCKS_M = N_CTX // BM
        NUM_BLOCKS = NUM_BLOCKS_M * Z * H
        DIM = q.shape[-1]
        # Number of NPU cores (adjust based on hardware)
        NUM_CORES = 20
        acc = torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2], HEAD_DIM_K),
            dtype=torch.float32,
            device=q.device,
        )
        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        workspace_1 = torch.empty(
            (NUM_BLOCKS, BM, BN), device=q.device, dtype=torch.float32
        )
        workspace_2 = torch.empty((NUM_BLOCKS, BM, BN), device=q.device, dtype=q.dtype)
        workspace_3 = torch.empty(
            (NUM_BLOCKS, BM, DIM), device=q.device, dtype=torch.float32
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
            workspace_1.stride(0),
            workspace_1.stride(1),
            workspace_1.stride(2),
            workspace_2.stride(0),
            workspace_2.stride(1),
            workspace_2.stride(2),
            workspace_3.stride(0),
            workspace_3.stride(1),
            workspace_3.stride(2),
            q.shape[0],
            q.shape[1],
            N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_M=BM,
            BLOCK_N=BN,
            NUM_CORES=NUM_CORES,
            disable_auto_inject_block_sync=True,
            disable_auto_cv_work_space_manage=True,
            **extra_kern_args,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        return o


attention_base = AttentionBase.apply
attention_split_cv = AttentionSplitCV.apply


@pytest.mark.parametrize(
    "Z, H, N_CTX, HEAD_DIM, dtype, BM, BN",
    [
        (1, 1, 128, 128, torch.float16, 32, 128),
        (1, 1, 128, 128, torch.bfloat16, 64, 128),
        (1, 2, 256, 256, torch.bfloat16, 32, 256),
        (2, 2, 128, 256, torch.float16, 64, 128),
        (4, 32, 64, 64, torch.float16, 32, 64),
        (4, 32, 1024, 64, torch.bfloat16, 64, 128),
    ],
)
def test_op(Z, H, N_CTX, HEAD_DIM, dtype, BM, BN):
    # Filter out non-integer cases; N_CTX must be divisible by BM and BN, and HEAD_DIM must be divisible by 16.
    if N_CTX % BM != 0 or N_CTX % BN != 0 or HEAD_DIM % 16 != 0:
        pytest.skip("Skipping non-divisible case")

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

    tri_base_out = attention_base(q, k, v, sm_scale, BM, BN)
    tri_split_cv_out = attention_split_cv(q, k, v, sm_scale, BM, BN)
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

    torch.testing.assert_close(
        ref_out, tri_base_out, atol=1e-2, rtol=1e-2, equal_nan=True
    )
    torch.testing.assert_close(
        ref_out, tri_split_cv_out, atol=1e-2, rtol=1e-2, equal_nan=True
    )
    print(
        f"[PASSED] Attention shape:({Z}, {H}, {N_CTX}, {HEAD_DIM}), BM: {BM}, BN: {BN}, dtype: {dtype}"
    )


def benchmark(Z, H, N_CTX, HEAD_DIM, dtype, BM, BN):
    if N_CTX % BM != 0 or N_CTX % BN != 0 or HEAD_DIM % 16 != 0:
        pytest.skip("Skipping non-divisible case")

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
    tri_base_out = attention_base(q, k, v, sm_scale, BM, BN)
    tri_split_cv_out = attention_split_cv(q, k, v, sm_scale, BM, BN)
    torch.testing.assert_close(
        tri_split_cv_out, tri_base_out, atol=1e-2, rtol=1e-2, equal_nan=True
    )

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["cnt"],  # Argument names to use as an x-axis for the plot
            # x_vals=[128 * i for i in range(10, 15)],  # Different possible values for `x_name`
            x_vals=[1],  # NOTE: the tunning framework specialized to one shape
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["fa_base", "fa_split_cv"],  # Label name for the lines
            line_names=["fa_base", "fa_split_cv"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="flash-attention-"
            + (
                f"{dtype}-[Batch={Z} H={H} N_CTX={N_CTX} HEAD_DIM={HEAD_DIM}]"
            ),  # Name for the plot, used also as a file name for saving the plot.
            args={},
        )
    )

    @triton.testing.perf_report(configs)
    def benchmark(cnt, provider):
        warmup = 500
        rep = 500
        quantiles = [0.5, 0.2, 0.8]
        if provider == "fa_base":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: attention_base(q, k, v, sm_scale, BM, BN),
                quantiles=quantiles,
                warmup=warmup,
                rep=rep,
            )
        if provider == "fa_split_cv":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: attention_split_cv(q, k, v, sm_scale, BM, BN),
                quantiles=quantiles,
                warmup=warmup,
                rep=rep,
            )

        return ms, max_ms, min_ms

    benchmark.run(show_plots=False, print_data=True)


if __name__ == "__main__":
    benchmark(1, 1, 128, 128, dtype=torch.float16, BM=32, BN=128)
    benchmark(1, 1, 512, 128, dtype=torch.float16, BM=32, BN=128)
    benchmark(1, 1, 1024, 128, dtype=torch.float16, BM=32, BN=128)
    benchmark(1, 1, 128, 128, dtype=torch.bfloat16, BM=64, BN=128)
    benchmark(1, 2, 256, 256, dtype=torch.bfloat16, BM=32, BN=256)
    benchmark(2, 2, 128, 256, dtype=torch.float16, BM=64, BN=128)
    benchmark(4, 32, 64, 64, dtype=torch.float16, BM=32, BN=64)
    benchmark(4, 32, 1024, 64, dtype=torch.bfloat16, BM=64, BN=128)
