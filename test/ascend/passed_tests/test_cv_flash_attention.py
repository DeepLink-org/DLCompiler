import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.deeplink as dl

DEVICE = "npu"

@triton.jit
def _attn_fwd(Q, K, V, M, Out, acc, sm_scale, workspace_1, workspace_2, workspace_3,
              stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,
              stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr,
              stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr,
              stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,
              w1_stride_nb: tl.constexpr, w1_stride_bm: tl.constexpr, w1_stride_bn: tl.constexpr,
              w2_stride_nb: tl.constexpr, w2_stride_bm: tl.constexpr, w2_stride_bn: tl.constexpr,
              w3_stride_nb: tl.constexpr, w3_stride_bm: tl.constexpr, w3_stride_dm: tl.constexpr,
              Z: tl.constexpr, H: tl.constexpr,
              N_CTX: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              NUM_CORES: tl.constexpr
              ):
    # Total number of blocks in sequence dimension (M)
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    # Total tasks = number of sequence blocks × batch size (Z) × number of attention heads (H)
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    # Current M-dimension block index
    pid = tl.program_id(0)
    with dl.async_task(scope=dl.async_task.cube):
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
            q = tl.load(Q_block_ptr)
            lo, hi = 0, N_CTX  # Process the entire context
            K_block_ptr = tl.advance(K_block_ptr, (lo, 0))  # K is [HEAD_DIM, N_CTX], shift along the second dim by lo
            V_block_ptr = tl.advance(V_block_ptr, (lo, 0))  # V is [N_CTX, HEAD_DIM], shift along the first dim by lo
            for start_n in range(lo, hi, BLOCK_N):  # Process BLOCK_N columns at a time
                start_n = tl.multiple_of(start_n, BLOCK_N)  # Align column start position
                k = tl.load(K_block_ptr)
                trans_k = tl.trans(k)
                qk = tl.dot(q, trans_k) # shape: [BLOCK_M, BLOCK_N]
                tl.store(workspace_1_ptr, qk)

                dl.set_cross_flag(dl.SyncFlag.C2V, 0)
                dl.wait_cross_flag(dl.SyncFlag.V2C, 1)
                
                p_cast = tl.load(workspace_2_ptr)  # shape: [BLOCK_M, BLOCK_N]
                v = tl.load(V_block_ptr)  # shape: [BLOCK_N, HEAD_DIM]
                acc_l0c = tl.dot(p_cast, v)  # shape: [BLOCK_M, HEAD_DIM]
                tl.store(workspace_3_ptr, acc_l0c)
                dl.set_cross_flag(dl.SyncFlag.C2V, 2)
                dl.wait_cross_flag(dl.SyncFlag.V2C, 3)
                V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
                K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))

    with dl.async_task(scope=dl.async_task.vector):
        for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
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
            offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
            m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
            acc_ptr = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
            lo, hi = 0, N_CTX  # Process the entire context
            for start_n in range(lo, hi, BLOCK_N):  # Process BLOCK_N columns at a time
                start_n = tl.multiple_of(start_n, BLOCK_N)  # Align column start position
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
                alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max
                l_i = l_i * alpha + l_ij  # Update softmax denominator
                
                acc_ptr = acc_ptr * alpha[:, None]
                # acc_ptr = tl.dot(p_cast, v, acc_ptr)
            
            
                dl.wait_cross_flag(dl.SyncFlag.C2V, 2)
                acc_o_ub = tl.load(workspace_3_ptr)
                acc_ptr = acc_ptr + acc_o_ub
                dl.set_cross_flag(dl.SyncFlag.V2C, 3)
                m_i = m_ij  # Update current block max
            

            m_i += tl.math.log(l_i)
            accumulator = acc_ptr / l_i[:, None]
            m_ptrs = M + task_hz_idx * N_CTX + offs_m

            tl.store(m_ptrs, m_i)
            tl.store(O_block_ptr, accumulator.to(Out.type.element_ty))


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, BM, BN):
        """
        Forward computation interface:
        Args:
            ctx: Context object
            q: Query tensor (Q), shape [Z, H, N_CTX, HEAD_DIM]
            k: Key tensor (K), shape [Z, H, N_CTX, HEAD_DIM]
            v: Value tensor (V), shape [Z, H, N_CTX, HEAD_DIM]
            causal: Whether to enable causal attention
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
        stage = 3 if causal else 1
        extra_kern_args = {}
        
        N_CTX=q.shape[2]
        Z, H = q.shape[0], q.shape[1]
        NUM_BLOCKS_M = N_CTX // BM
        NUM_BLOCKS = NUM_BLOCKS_M * Z * H
        DIM = q.shape[-1]
        # Number of NPU cores (adjust based on hardware)
        NUM_CORES = 20
        acc = torch.zeros((q.shape[0], q.shape[1], q.shape[2], HEAD_DIM_K), dtype=torch.float32, device=q.device)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        workspace_1 = torch.empty((NUM_BLOCKS, BM, BN), device=q.device, dtype=torch.float32)
        workspace_2 = torch.empty((NUM_BLOCKS, BM, BN), device=q.device, dtype=q.dtype)
        workspace_3 = torch.empty((NUM_BLOCKS, BM, DIM), device=q.device, dtype=torch.float32)

        _attn_fwd[(NUM_CORES,)](
            q, k, v, M, o, acc, sm_scale, workspace_1, workspace_2, workspace_3,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            workspace_1.stride(0), workspace_1.stride(1), workspace_1.stride(2),
            workspace_2.stride(0), workspace_2.stride(1), workspace_2.stride(2),
            workspace_3.stride(0), workspace_3.stride(1), workspace_3.stride(2),
            q.shape[0], q.shape[1], N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_M=BM,
            BLOCK_N=BN,
            NUM_CORES=NUM_CORES,
            disable_auto_inject_block_sync=True,
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

attention = _attention.apply


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM, causal, dtype, BM, BN", [
    (1, 1, 128, 128, False, torch.float16, 32, 128),
    (1, 1, 128, 128, False, torch.bfloat16, 64, 128),
    (1, 2, 256, 256, False, torch.bfloat16, 32, 256),
    (2, 2, 128, 256, False, torch.float16, 64, 128),
    (4, 32, 64, 64, False, torch.float16, 32, 64),
    (4, 32, 1024, 64, False, torch.bfloat16, 64, 128),
])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype, BM, BN):
    # Filter out non-integer cases; N_CTX must be divisible by BM and BN, and HEAD_DIM must be divisible by 16.
    if N_CTX % BM != 0 or N_CTX % BN != 0 or HEAD_DIM % 16 != 0:
        pytest.skip("Skipping non-divisible case")

    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())

    sm_scale = 0.5

    tri_out = attention(q, k, v, causal, sm_scale, BM, BN)
    ref_out = torch_npu.npu_fusion_attention(
            q, k, v, H,
            padding_mask=None,
            atten_mask=None,
            scale=sm_scale,
            keep_prob=1.0,
            input_layout="BNSD",
            pre_tockens=65535,
            next_tockens=65535,
            sparse_mode=0,
            )[0]

    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2, equal_nan=True)
    print(f"[PASSED] Attention shape:({Z}, {H}, {N_CTX}, {HEAD_DIM}), BM: {BM}, BN: {BN}, dtype: {dtype}")


if __name__ == "__main__":
    test_op(1, 1, 128, 128, causal=False, dtype=torch.float16, BM=32, BN=128)
    test_op(1, 1, 128, 128, causal=False, dtype=torch.bfloat16, BM=64, BN=128)
    test_op(1, 2, 256, 256, causal=False, dtype=torch.bfloat16, BM=32, BN=256)
    test_op(2, 2, 128, 256, causal=False, dtype=torch.float16, BM=64, BN=128)
    test_op(4, 32, 64, 64, causal=False, dtype=torch.float16, BM=32, BN=64)
    test_op(4, 32, 1024, 64, causal=False, dtype=torch.bfloat16, BM=64, BN=128)
  
