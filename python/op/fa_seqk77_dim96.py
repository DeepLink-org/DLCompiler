import math
import argparse

import torch
import triton
import triton.language as tl

if triton.__version__ >= "3.0.0":
    from triton.language.extra.cuda.libdevice import fast_expf as tl_exp
    from triton.language.extra.cuda.libdevice import fast_logf as tl_log
else:
    from triton.language.math import fast_expf as tl_exp
    from triton.language.math import fast_logf as tl_log

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@triton.autotune(
    configs = [
        triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
        for BM in [64] \
        for BN in [32] \
        for s in [2] \
        for w in [4] \
    ],
    key=['seqlen_q', 'seqlen_k', 'seqlen_q_rounded'],
)

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel_seqk77_hdim96(
    Q, K, V, Bias,
    Out, Lse, TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_bb, stride_bh, stride_bm,
    stride_ob, stride_oh, stride_om,
    nheads,
    seqlen_q, seqlen_k, seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(1)
    off_hb = tl.program_id(0) * tl.num_programs(2) + tl.program_id(2)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    # off_hb = off_b * nheads + off_h
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_d0 = tl.arange(0, 32)
    offs_d1 = 32 + tl.arange(0, 32)
    offs_d2 = 64 + tl.arange(0, 32)
    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)

    q_off = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm)
    )

    k_off = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[None, :] * stride_kn)
    )
    v_off = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn)
    )

    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize pointer to m and l
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    acc_o0 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
    acc_o1 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
    acc_o2 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        q0 = tl.load(q_off + offs_d0[None, :])
        q1 = tl.load(q_off + offs_d1[None, :])
        q2 = tl.load(q_off + offs_d2[None, :])

    else:
        q0 = tl.load(q_off + offs_d0[None, :], mask=offs_m[:, None] < seqlen_q, other=0.0)
        q1 = tl.load(q_off + offs_d1[None, :], mask=offs_m[:, None] < seqlen_q, other=0.0)
        q2 = tl.load(q_off + offs_d2[None, :], mask=offs_m[:, None] < seqlen_q, other=0.0)
    
    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    iteration = 0
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:
            k0 = tl.load(k_off + offs_d0[:, None] + start_n * stride_kn)
        else:
            k0 = tl.load(
                        k_off + offs_d0[:, None] + start_n * stride_kn,
                        mask=(start_n + offs_n)[None, :] < seqlen_k,
                        other=0.0
                    )
        qk = tl.dot(q0, k0)
        if EVEN_N & EVEN_M:
            k1 = tl.load(k_off + offs_d1[:, None] + start_n * stride_kn)
        else:
            k1 = tl.load(
                        k_off + offs_d1[:, None] + start_n * stride_kn,
                        mask=(start_n + offs_n)[None, :] < seqlen_k,
                        other=0.0
                    )
        qk += tl.dot(q1, k1)
        if EVEN_N & EVEN_M:
            k2 = tl.load(k_off + offs_d2[:, None] + start_n * stride_kn)
        else:
            k2 = tl.load(
                        k_off + offs_d2[:, None] + start_n * stride_kn,
                        mask=(start_n + offs_n)[None, :] < seqlen_k,
                        other=0.0
                    )
        qk += tl.dot(q2, k2)
        
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:
            # Need to mask out otherwise the softmax is wrong;
            # seems ok
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))

        

        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl_exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl_exp(qk - m_ij[:, None])
        else:
            # if iteration == 2:
            #     p = qk
            #     p_tmp0, _ = tl.split(tl.reshape(p, (128, 16, 2)))

            #     m_ij = tl.maximum(lse_i, tl.max(p_tmp0, 1) * softmax_scale)
            #     p_tmp0 = p_tmp0*softmax_scale - m_ij[:, None]
            # else:
            m_ij = tl.maximum(lse_i, tl.max(qk, 1) * softmax_scale)
            qk = qk*softmax_scale - m_ij[:, None]
            p = qk
            # p = tl_exp(qk)
            # p = tl_exp(qk)
            # p = qk

        if iteration == 2:
            p_tmp, _ = tl.split(tl.reshape(p, (BLOCK_M, 16, 2)))
            p_tmp = tl_exp(p_tmp)
        else:
            p = tl_exp(p)
        
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl_exp(m_i - m_ij)
        # acc_o_scale = tl.math.exp2(m_i - m_ij)

        # # -- update output accumulator --
        # acc_o = acc_o * acc_o_scale[:, None]
        acc_o0 = acc_o0 * acc_o_scale[:, None]
        acc_o1 = acc_o1 * acc_o_scale[:, None]
        acc_o2 = acc_o2 * acc_o_scale[:, None]
        
        # update acc_o
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            v0 = tl.load(v_off + offs_d0[None, :] + start_n * stride_vn)
        else:
            v0 = tl.load(
                v_off + offs_d0[None, :] + start_n * stride_vn,
                mask=(start_n + offs_n)[:, None] < seqlen_k,
                other=0.0,
            )
        p = p.to(v0.dtype)
        p0, p1 = tl.split(tl.reshape(p, (BLOCK_M, 16, 2)))
        if iteration == 2:
            v00, v01 = tl.split(tl.reshape(tl.trans(v0), (32, 16, 2)))
            acc_o0 += tl.dot(p0, tl.trans(v00))
        else:
            acc_o0 += tl.dot(p, v0)
        # acc_o0 += tl.dot(p, v0)

        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            v0 = tl.load(v_off + offs_d1[None, :] + start_n * stride_vn)
        else:
            v0 = tl.load(
                v_off + offs_d1[None, :] + start_n * stride_vn,
                mask=(start_n + offs_n)[:, None] < seqlen_k,
                other=0.0,
            )
        if iteration == 2:
            v00, v01 = tl.split(tl.reshape(tl.trans(v0), (32, 16, 2)))
            acc_o1 += tl.dot(p0, tl.trans(v00))
        else:
            acc_o1 += tl.dot(p, v0)
        # acc_o1 += tl.dot(p, v1)

        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            v0 = tl.load(v_off + offs_d2[None, :] + start_n * stride_vn)
        else:
            v0 = tl.load(
                v_off + offs_d2[None, :] + start_n * stride_vn,
                mask=(start_n + offs_n)[:, None] < seqlen_k,
                other=0.0,
            )
        if iteration == 2:
            v00, v01 = tl.split(tl.reshape(tl.trans(v0), (32, 16, 2)))
            acc_o2 += tl.dot(p0, tl.trans(v00))
        else:
            acc_o2 += tl.dot(p, v0)
        # acc_o2 += tl.dot(p, v2)
        
        # -- update statistics
        m_i = m_ij
        l_i_new = tl_exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl_log(l_i_new)
        iteration += 1

    o_scale = tl_exp(m_i - lse_i)
    acc_o0 = acc_o0 * o_scale[:, None]
    acc_o1 = acc_o1 * o_scale[:, None]
    acc_o2 = acc_o2 * o_scale[:, None]

    #
    # store
    # rematerialize offsets to save registers
    #
    # start_m = tl.program_id(1)
    # offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output
    
    out_off = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om)
    )
    if EVEN_M:
        tl.store(out_off + offs_d0[None, :], acc_o0)
        tl.store(out_off + offs_d1[None, :], acc_o1)
        tl.store(out_off + offs_d2[None, :], acc_o2)
    else:
        tl.store(out_off + offs_d0[None, :], acc_o0, mask=offs_m[:, None] < seqlen_q)
        tl.store(out_off + offs_d1[None, :], acc_o1, mask=offs_m[:, None] < seqlen_q)
        tl.store(out_off + offs_d2[None, :], acc_o2, mask=offs_m[:, None] < seqlen_q)
        
@triton.autotune(
    configs = [
        triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
        for BM in [128] \
        for BN in [32] \
        for s in [2] \
        for w in [4] \
    ],
    key=['seqlen_q', 'seqlen_k', 'seqlen_q_rounded'],
)

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel_hdim96_fp16(
    Q, K, V, Bias,
    Out, Lse, TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_bb, stride_bh, stride_bm,
    stride_ob, stride_oh, stride_om,
    nheads,
    seqlen_q, seqlen_k, seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(1)
    off_hb = tl.program_id(0)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    # off_hb = off_b * nheads + off_h
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_d0 = tl.arange(0, 32)
    offs_d1 = 32 + tl.arange(0, 32)
    offs_d2 = 64 + tl.arange(0, 32)
    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)

    q_off = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm)
    )

    k_off = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[None, :] * stride_kn)
    )
    v_off = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn)
    )

    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize pointer to m and l
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    acc_o0 = tl.zeros([BLOCK_M, 32], dtype=tl.float16)
    acc_o1 = tl.zeros([BLOCK_M, 32], dtype=tl.float16)
    acc_o2 = tl.zeros([BLOCK_M, 32], dtype=tl.float16)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        q0 = tl.load(q_off + offs_d0[None, :])
        q1 = tl.load(q_off + offs_d1[None, :])
        q2 = tl.load(q_off + offs_d2[None, :])

    else:
        q0 = tl.load(q_off + offs_d0[None, :], mask=offs_m[:, None] < seqlen_q, other=0.0)
        q1 = tl.load(q_off + offs_d1[None, :], mask=offs_m[:, None] < seqlen_q, other=0.0)
        q2 = tl.load(q_off + offs_d2[None, :], mask=offs_m[:, None] < seqlen_q, other=0.0)
    
    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:
            k0 = tl.load(k_off + offs_d0[:, None] + start_n * stride_kn)
        else:
            k0 = tl.load(
                        k_off + offs_d0[:, None] + start_n * stride_kn,
                        mask=(start_n + offs_n)[None, :] < seqlen_k,
                        other=0.0
                    )
        qk = tl.dot(q0, k0, out_dtype = tl.float16)
        if EVEN_N & EVEN_M:
            k1 = tl.load(k_off + offs_d1[:, None] + start_n * stride_kn)
        else:
            k1 = tl.load(
                        k_off + offs_d1[:, None] + start_n * stride_kn,
                        mask=(start_n + offs_n)[None, :] < seqlen_k,
                        other=0.0
                    )
        qk += tl.dot(q1, k1, out_dtype = tl.float16)
        if EVEN_N & EVEN_M:
            k2 = tl.load(k_off + offs_d2[:, None] + start_n * stride_kn)
        else:
            k2 = tl.load(
                        k_off + offs_d2[:, None] + start_n * stride_kn,
                        mask=(start_n + offs_n)[None, :] < seqlen_k,
                        other=0.0
                    )
        qk += tl.dot(q2, k2, out_dtype = tl.float16)

        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:
            # Need to mask out otherwise the softmax is wrong;
            # seems ok
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))

        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl_exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl_exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(lse_i, tl.max(qk, 1) * softmax_scale)
            qk = qk*softmax_scale - m_ij[:, None]
            p = tl_exp(qk)

        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl_exp(m_i - m_ij).to(tl.float16)
        # acc_o_scale = tl.math.exp2(m_i - m_ij)

        # # -- update output accumulator --
        # acc_o = acc_o * acc_o_scale[:, None]
        acc_o0 = acc_o0 * acc_o_scale[:, None]
        acc_o1 = acc_o1 * acc_o_scale[:, None]
        acc_o2 = acc_o2 * acc_o_scale[:, None]
        
        # update acc_o
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            v0 = tl.load(v_off + offs_d0[None, :] + start_n * stride_vn)
        else:
            v0 = tl.load(
                v_off + offs_d0[None, :] + start_n * stride_vn,
                mask=(start_n + offs_n)[:, None] < seqlen_k,
                other=0.0,
            )
        p = p.to(v0.dtype)
        acc_o0 += tl.dot(p, v0, out_dtype = tl.float16)

        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            v1 = tl.load(v_off + offs_d1[None, :] + start_n * stride_vn)
        else:
            v1 = tl.load(
                v_off + offs_d1[None, :] + start_n * stride_vn,
                mask=(start_n + offs_n)[:, None] < seqlen_k,
                other=0.0,
            )
        acc_o1 += tl.dot(p, v1, out_dtype = tl.float16)

        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            v2 = tl.load(v_off + offs_d2[None, :] + start_n * stride_vn)
        else:
            v2 = tl.load(
                v_off + offs_d2[None, :] + start_n * stride_vn,
                mask=(start_n + offs_n)[:, None] < seqlen_k,
                other=0.0,
            )
        acc_o2 += tl.dot(p, v2, out_dtype = tl.float16)
        
        # -- update statistics
        m_i = m_ij
        l_i_new = tl_exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl_log(l_i_new)

    o_scale = tl_exp(m_i - lse_i)
    acc_o0 = acc_o0 * o_scale[:, None]
    acc_o1 = acc_o1 * o_scale[:, None]
    acc_o2 = acc_o2 * o_scale[:, None]

    #
    # store
    # rematerialize offsets to save registers
    #
    start_m = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output
    
    out_off = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om)
    )
    if EVEN_M:
        tl.store(out_off + offs_d0[None, :], acc_o0)
        tl.store(out_off + offs_d1[None, :], acc_o1)
        tl.store(out_off + offs_d2[None, :], acc_o2)
    else:
        tl.store(out_off + offs_d0[None, :], acc_o0, mask=offs_m[:, None] < seqlen_q)
        tl.store(out_off + offs_d1[None, :], acc_o1, mask=offs_m[:, None] < seqlen_q)
        tl.store(out_off + offs_d2[None, :], acc_o2, mask=offs_m[:, None] < seqlen_q)
        

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
        for BM in [128] \
        for BN in [32] \
        for s in [3] \
        for w in [4] \
    ],
    key=['seqlen_q', 'seqlen_k', 'seqlen_q_rounded'],
)

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q, K, V, Bias,
    Out, Lse, TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_bb, stride_bh, stride_bm,
    stride_ob, stride_oh, stride_om,
    nheads,
    seqlen_q, seqlen_k, seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(1)
    off_hb = tl.program_id(0)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    # off_hb = off_b * nheads + off_h
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[None, :] * stride_kn + offs_d[:, None])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize pointer to m and l
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, cache_modifier=".cg")
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0, cache_modifier=".cg")
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0, cache_modifier=".cg")
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0, cache_modifier=".cg"
            )
    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn, cache_modifier=".cg")
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[:, None] < headdim, other=0.0, cache_modifier=".cg")
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[None, :] < seqlen_k,
                    other=0.0,
                    cache_modifier=".cg"
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[None, :] < seqlen_k) & (offs_d[:, None] < headdim),
                    other=0.0,
                    cache_modifier=".cg"
                )
        qk = tl.dot(q, k)

        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:
            # Need to mask out otherwise the softmax is wrong;
            # seems ok
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))

        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n, cache_modifier=".cg").to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0, cache_modifier=".cg"
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n, cache_modifier=".cg").to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                        cache_modifier=".cg"
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl_exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl_exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(lse_i, tl.max(qk, 1) * softmax_scale)
            qk = qk*softmax_scale - m_ij[:, None]
            p = tl_exp(qk)

        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl_exp(m_i - m_ij)
        # acc_o_scale = tl.math.exp2(m_i - m_ij)

        # # -- update output accumulator --
        acc_o = acc_o * acc_o_scale[:, None]

        # update acc_o
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn, cache_modifier=".cg")
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0, cache_modifier=".cg")
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                    cache_modifier=".cg"
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                    cache_modifier=".cg"
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl_exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl_log(l_i_new)

    o_scale = tl_exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]

    #
    # store
    # rematerialize offsets to save registers
    #
    start_m = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )
# from triton.runtime import driver
# import os
# LIBDEVICE_PATH = os.getenv("TRITON_LIBDEVICE_PATH", driver.libdevice_path)
# @core.extern
# def tl_exp(arg0, _builder=None):
#     return core.extern_elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
#                                    {(core.dtype("fp32"),): ("__nv_tl_exp", core.dtype("fp32")),
#                                     }, is_pure=True, _builder=_builder)
@triton.autotune(
    configs = [
        triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
        for BM in [128] \
        for BN in [32] \
        for s in [3] \
        for w in [4] \
    ],
    key=['seqlen_q', 'seqlen_k', 'seqlen_q_rounded'],
)

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel_hdim96(
    Q, K, V, Bias,
    Out, Lse, TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_bb, stride_bh, stride_bm,
    stride_ob, stride_oh, stride_om,
    nheads,
    seqlen_q, seqlen_k, seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(1)
    off_hb = tl.program_id(0)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    # off_hb = off_b * nheads + off_h
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_d0 = tl.arange(0, 32)
    offs_d1 = 32 + tl.arange(0, 32)
    offs_d2 = 64 + tl.arange(0, 32)
    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)

    q_off = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm)
    )

    k_off = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[None, :] * stride_kn)
    )
    v_off = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn)
    )

    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize pointer to m and l
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    acc_o0 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
    acc_o1 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
    acc_o2 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        q0 = tl.load(q_off + offs_d0[None, :])
        q1 = tl.load(q_off + offs_d1[None, :])
        q2 = tl.load(q_off + offs_d2[None, :])

    else:
        q0 = tl.load(q_off + offs_d0[None, :], mask=offs_m[:, None] < seqlen_q, other=0.0)
        q1 = tl.load(q_off + offs_d1[None, :], mask=offs_m[:, None] < seqlen_q, other=0.0)
        q2 = tl.load(q_off + offs_d2[None, :], mask=offs_m[:, None] < seqlen_q, other=0.0)
    
    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:
            k0 = tl.load(k_off + offs_d0[:, None] + start_n * stride_kn)
        else:
            k0 = tl.load(
                        k_off + offs_d0[:, None] + start_n * stride_kn,
                        mask=(start_n + offs_n)[None, :] < seqlen_k,
                        other=0.0
                    )
        qk = tl.dot(q0, k0)
        if EVEN_N & EVEN_M:
            k1 = tl.load(k_off + offs_d1[:, None] + start_n * stride_kn)
        else:
            k1 = tl.load(
                        k_off + offs_d1[:, None] + start_n * stride_kn,
                        mask=(start_n + offs_n)[None, :] < seqlen_k,
                        other=0.0
                    )
        qk += tl.dot(q1, k1)
        if EVEN_N & EVEN_M:
            k2 = tl.load(k_off + offs_d2[:, None] + start_n * stride_kn)
        else:
            k2 = tl.load(
                        k_off + offs_d2[:, None] + start_n * stride_kn,
                        mask=(start_n + offs_n)[None, :] < seqlen_k,
                        other=0.0
                    )
        qk += tl.dot(q2, k2)

        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:
            # Need to mask out otherwise the softmax is wrong;
            # seems ok
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))

        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl_exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl_exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(lse_i, tl.max(qk, 1) * softmax_scale)
            qk = qk*softmax_scale - m_ij[:, None]
            p = tl_exp(qk)

        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl_exp(m_i - m_ij)
        # acc_o_scale = tl.math.exp2(m_i - m_ij)

        # # -- update output accumulator --
        # acc_o = acc_o * acc_o_scale[:, None]
        acc_o0 = acc_o0 * acc_o_scale[:, None]
        acc_o1 = acc_o1 * acc_o_scale[:, None]
        acc_o2 = acc_o2 * acc_o_scale[:, None]
        
        # update acc_o
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            v0 = tl.load(v_off + offs_d0[None, :] + start_n * stride_vn)
        else:
            v0 = tl.load(
                v_off + offs_d0[None, :] + start_n * stride_vn,
                mask=(start_n + offs_n)[:, None] < seqlen_k,
                other=0.0,
            )
        p = p.to(v0.dtype)
        acc_o0 += tl.dot(p, v0)

        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            v1 = tl.load(v_off + offs_d1[None, :] + start_n * stride_vn)
        else:
            v1 = tl.load(
                v_off + offs_d1[None, :] + start_n * stride_vn,
                mask=(start_n + offs_n)[:, None] < seqlen_k,
                other=0.0,
            )
        acc_o1 += tl.dot(p, v1)

        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            v2 = tl.load(v_off + offs_d2[None, :] + start_n * stride_vn)
        else:
            v2 = tl.load(
                v_off + offs_d2[None, :] + start_n * stride_vn,
                mask=(start_n + offs_n)[:, None] < seqlen_k,
                other=0.0,
            )
        acc_o2 += tl.dot(p, v2)
        
        # -- update statistics
        m_i = m_ij
        l_i_new = tl_exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl_log(l_i_new)

    o_scale = tl_exp(m_i - lse_i)
    acc_o0 = acc_o0 * o_scale[:, None]
    acc_o1 = acc_o1 * o_scale[:, None]
    acc_o2 = acc_o2 * o_scale[:, None]

    #
    # store
    # rematerialize offsets to save registers
    #
    start_m = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output
    
    out_off = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om)
    )
    if EVEN_M:
        tl.store(out_off + offs_d0[None, :], acc_o0)
        tl.store(out_off + offs_d1[None, :], acc_o1)
        tl.store(out_off + offs_d2[None, :], acc_o2)
    else:
        tl.store(out_off + offs_d0[None, :], acc_o0, mask=offs_m[:, None] < seqlen_q)
        tl.store(out_off + offs_d1[None, :], acc_o1, mask=offs_m[:, None] < seqlen_q)
        tl.store(out_off + offs_d2[None, :], acc_o2, mask=offs_m[:, None] < seqlen_q)
        




def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def _flash_attn_forward(q, k, v, bias=None, causal=False, softmax_scale=None):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (batch * nheads, triton.cdiv(seqlen_q, META["BLOCK_M"]))

    if d == 96:
        _fwd_kernel = _fwd_kernel_hdim96
    if d == 96 and q.dtype == torch.float16:
        _fwd_kernel = _fwd_kernel_hdim96_fp16
        _fwd_kernel = _fwd_kernel_seqk77_hdim96
    _fwd_kernel[grid](
        q, k, v, bias,
        o, lse, tmp,
        softmax_scale,
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1),
        *bias_strides,
        o.stride(0), o.stride(2), o.stride(1),
        nheads,
        seqlen_q, seqlen_k, seqlen_q_rounded,
        d,  # headdim
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        # BLOCK_M=BLOCK,
        # BLOCK_N=BLOCK,
        # num_warps=num_warps,
        # num_stages=1,
    )
    # print(f"_fwd_kernel.best_config ", _fwd_kernel.best_config, flush = True)
    return o, lse, softmax_scale  # softmax_scale could have been updated



class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias=None, causal=False, softmax_scale=None):
        """
        q: (batch_size, seqlen_q, nheads, headdim)
        k, v: (batch_size, seqlen_k, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
            For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
            ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        # Make sure that the last dimension is contiguous
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
        o, lse, ctx.softmax_scale = _flash_attn_forward(
            q, k, v, bias=bias, causal=causal, softmax_scale=softmax_scale
        )
        ctx.save_for_backward(q, k, v, o, lse, bias)
        ctx.causal = causal
        return o


    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, bias = ctx.saved_tensors
        assert not ctx.needs_input_grad[3], "FlashAttention does not support bias gradient yet"
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            _flash_attn_backward(
                do,
                q,
                k,
                v,
                o,
                lse,
                dq,
                dk,
                dv,
                bias=bias,
                causal=ctx.causal,
                softmax_scale=ctx.softmax_scale,
            )
        return dq, dk, dv, None, None, None


flash_attn_func = FlashAttnFunc.apply

if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    import time

    args = parse_args()
    torch.manual_seed(args.seed)

    # Optionally use the context manager to ensure one of the fused kernels is run
    dtype=torch.float16
    device_ = torch.device("cuda:4")
    query = torch.rand([1, 32926, 32, 96], dtype=dtype, device=device_, requires_grad = True)
    key = torch.rand([1, 77, 32, 96], dtype=dtype, device=device_, requires_grad = True)
    value = torch.rand([1, 77, 32, 96], dtype=dtype, device=device_, requires_grad = True)
    torch.cuda.set_device(device_)

    ref_out = F.scaled_dot_product_attention(query.permute(0, 2, 1, 3),key.permute(0, 2, 1, 3),value.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

    tt_out = flash_attn_func(query,key,value)

    for i, j in zip(ref_out.shape, tt_out.shape):
        assert i == j

    print('TEST: ')
    # print(tt_out - ref_out)
    print('max abs diff: ', torch.max(abs(tt_out - ref_out)))
    print()
    assert torch.allclose(tt_out, ref_out, atol=1e-2, rtol=0)

    TORCH_HAS_FP8 = False
    HAS_FLASH = False

    BATCH, N_HEADS, HEAD_DIM = 4, 32, 64
    # vary seq length for fixed head and batch=4
    configs = []
    # for mode in ["fwd", "bwd"]:
    for mode in ["fwd"]:
        # for causal in [True, False]:
        for causal in [False]:
            if mode == "bwd" and not causal:
                continue
            configs.append(
                triton.testing.Benchmark(
                    x_names=["N_CTX"],
                    # x_vals=[2**i for i in range(10, 15)],
                    x_vals=[32926],
                    line_arg="provider",

                    line_vals=["triton"] + (["triton-fp8"] if TORCH_HAS_FP8 else []) +
                    (["flash"] if HAS_FLASH else [] )+ ['pytorch'],
                    line_names=["Triton"] + (["Triton [FP8]"] if TORCH_HAS_FP8 else []) +
                    (["Flash-2"] if HAS_FLASH else [] ) + ['pytorch'],

                    # line_vals=["triton-fp16"] + ['pytorch-fp16'],
                    # line_names=["Triton [FP16]"] + ['pytorch-fp16'],

                    styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("orange", "-")],
                    ylabel="ms",
                    plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                    args={
                        "H": N_HEADS,
                        "BATCH": BATCH,
                        "HEAD_DIM": HEAD_DIM,
                        "mode": mode,
                        "causal": causal,
                    },
                ))


    @triton.testing.perf_report(configs)
    def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device="cuda"):

        # setting
        assert mode in ["fwd", "bwd"]
    
        warmup = 100
        rep = 100

        if "triton" in provider:
            if mode == "fwd" and "fp8" in provider:
                raise
            sm_scale = 1.3
            fn = lambda: flash_attn_func(query, key, value)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

        if "pytorch" in provider:
            sm_scale = 1.3
            fn = lambda : F.scaled_dot_product_attention(query.permute(0, 2, 1, 3),key.permute(0, 2, 1, 3),value.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            # fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

        # flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
        # total_flops = 2 * flops_per_matmul
        # if causal:
        #     total_flops *= 0.5
        # if mode == "bwd":
        #     total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
        # return total_flops / ms * 1e-9
        return ms


    bench_flash_attention.run(show_plots=True, print_data=True)



