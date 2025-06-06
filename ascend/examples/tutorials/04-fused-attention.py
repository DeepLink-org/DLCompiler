"""
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

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  # Accumulator, local l, local m, query vector
                    K_block_ptr, V_block_ptr,  # Key and value block pointers for current stage
                    start_m, qk_scale,  # Starting position of current query block, qk scale factor
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  # Block size constants
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  # Current stage flag, m and n offset indices
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):  # Total context length, whether to enable FP8 for value precision
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
        lo, hi = 0, start_m * BLOCK_M  # Stage 1: process all tokens before the query block
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M  # Stage 2: process the current query block
        lo = tl.multiple_of(lo, BLOCK_M)  # Align starting position
    # causal = False (no need for masking)
    else:
        lo, hi = 0, N_CTX  # Process the entire context

    # Adjust K and V block pointers to the starting position `lo`
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))  # K is [HEAD_DIM, N_CTX], shift along the second dim by lo
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))  # V is [N_CTX, HEAD_DIM], shift along the first dim by lo

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
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])  # Construct upper triangular mask
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)  # Set invalid positions to -∞
            m_ij = tl.maximum(m_i, tl.max(qk, 1))  # Update m_ij = max(m_i, max(qk))
            qk -= m_ij[:, None]  # Subtract max for softmax stability
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)  # Scaled max
            qk = qk * qk_scale - m_ij[:, None]  # Stabilize
        # Softmax weights p = exp(qk)
        p = tl.math.exp2(qk)  # Use base-2 exponent for better numerical stability
        l_ij = tl.sum(p, 1)  # Softmax denominator (sum of each row)
        # -- Update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)  # Update factor: exp difference between old and new max
        l_i = l_i * alpha + l_ij  # Update softmax denominator
        # -- Update output accumulator --
        acc = acc * alpha[:, None]  # Scale accumulator by alpha to maintain consistency
        # update acc
        v = tl.load(V_block_ptr)  # Load corresponding V block
        # Convert softmax weight type depending on FP8 usage
        if fp8_v:
            p = p.to(tl.float8e5)  # Convert to FP8 format (save memory)
        else:
            p = p.to(tl.float16) 
        # -------------------------------
        acc = tl.dot(p, v, acc)  # Multiply softmax weights with V and accumulate to acc
        m_i = m_ij  # Update current block max
        # Advance V and K block pointers to next BLOCK_N range
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i  # Return accumulated output acc, softmax denominator l_i, and max value m_i

@triton.jit
def _attn_fwd(Q, K, V, M, Out, sm_scale,  
              stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr, 
              stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr, 
              stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr, 
              stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr, 
              Z: tl.constexpr, H: tl.constexpr,
              N_CTX: tl.constexpr, 
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr
              ):
    # Assert that BLOCK_N does not exceed HEAD_DIM
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    # Current M-dimension block index
    start_m = tl.program_id(0)

    # Loop through all (Z * H) attention groups
    for off_hz in range(0,Z*H):
        # Compute batch and head index
        off_z = off_hz // H
        off_h = off_hz % H

        # Offset for current batch and head
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

        # Construct Q block pointer (BLOCK_M, HEAD_DIM)
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        # Construct V block pointer (starts from 0,0; advanced in inner)
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )

        # Construct K block pointer
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )

        # Construct Out block pointer
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        # Construct offset indices along M and N
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        # Initialize m_i and l_i for softmax normalization
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Apply softmax scale and convert to log2 base
        qk_scale = sm_scale
        qk_scale *= 1.44269504  # 1 / log(2)

        # Load current Q block
        q = tl.load(Q_block_ptr)

        # STAGE controls whether to run stage 1 (off-band) or stage 2 (on-band)
        # stage 1: off-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1
        # For causal = False, STAGE = 1 and _attn_fwd_inner gets 3
        if STAGE & 1:
            # Off-diagonal attention computation
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  
                                            start_m, qk_scale,  
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  
                                            4 - STAGE,  # STAGE=1 -> inner=3, STAGE=3 -> inner=1
                                            offs_m, 
                                            offs_n, 
                                            N_CTX, 
                                            V.dtype.element_ty == tl.float8e5  # whether to use fp8
                                            )

        # stage 2: on-band
        if STAGE & 2:
            # Diagonal block attention computation
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                                            start_m, qk_scale,
                                            BLOCK_M, HEAD_DIM, BLOCK_N,
                                            2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5
                                            )

        # Epilogue: normalize and compute logsumexp
        m_i += tl.math.log2(l_i)  # compute logsumexp
        acc = acc / l_i[:, None]  # normalize output

        # Store logsumexp to M and final output to Out
        m_ptrs = M + off_hz * N_CTX + offs_m
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

        # grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        grid = (triton.cdiv(q.shape[2], BM), 1, 1)

        # (1, 2, 1024)
        # Allocate a temporary buffer M used during softmax computation
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        _attn_fwd[grid](
            q, k, v, M, o, sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_M=BM,
            BLOCK_N=BN,
            STAGE=stage,
            debug=True,
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

attention = _attention.apply

@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM, causal, dtype, BM , BN",[
    (4, 32, 32, 64, False, torch.float16, 32, 32),
    (4, 32, 64, 64, False, torch.float16, 32, 64),
    (1, 2, 128, 128, False, torch.float16, 32, 128),
    (1, 1, 128, 128, False, torch.float16, 64, 128),
    (1, 1, 128, 128, False, torch.float16, 32, 128),
    (2, 2, 128, 256, False, torch.float16, 32, 128),
    (1, 2, 256, 256, False, torch.float16, 32, 256),
])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype,BM ,BN ):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
  
    sm_scale = 0.5

    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    tri_out = attention(q, k, v, causal, sm_scale,BM,BN ).half()
    
    try:
        torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2)
        print(f"Test Fused-Attention PASS!")
    except AssertionError as e:
        print(f"Test Fused-Attention FAILED with ({Z},{H},{N_CTX},{HEAD_DIM}), causal={causal}, dtype={dtype}, BM={BM}, BN={BN} ! ERROR:{e}")
   
if __name__ == "__main__":
   test_op(4,32,32,64, causal=False, dtype=torch.float16, BM = 32,BN = 32)
   test_op(4,32,64,64, causal=False, dtype=torch.float16, BM = 32,BN = 64)
   test_op(1,2,128,128, causal=False, dtype=torch.float16, BM = 32,BN = 128)
   test_op(1,1,128,128, causal=False, dtype=torch.float16, BM = 64,BN = 128)
   test_op(1,1,128,128, causal=False, dtype=torch.float16, BM = 32,BN = 128)
   test_op(2,2,128,256, causal=False, dtype=torch.float16, BM = 32,BN = 128)
   test_op(1,2,256,256, causal=False, dtype=torch.float16, BM = 32,BN = 256)