import math
import torch
import triton
import triton.language as tl
from dlblas.utils import register_dlblas_op, SymVar, Tensor, ChoiceSpace

if triton.__version__ >= "3.0.0":
    from triton.language.extra.cuda.libdevice import fast_expf as tl_exp
    from triton.language.extra.cuda.libdevice import fast_logf as tl_log
else:
    from triton.language.math import fast_expf as tl_exp
    from triton.language.math import fast_logf as tl_log


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
        for BM in [128, 256]
        for BN in [32, 64]
        for s in [2]
        for w in [4]
    ],
    key=["seqlen_q", "seqlen_k", "seqlen_q_rounded"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Bias,
    Out,
    Lse,
    TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    head_dim: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
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
    offs_d = tl.arange(0, head_dim)

    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)

    q_off = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm)
    k_off = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn)
    v_off = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn)

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
    acc_o = tl.zeros([BLOCK_M, head_dim], dtype=tl.float16)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        q = tl.load(q_off + offs_d[None, :])
    else:
        q = tl.load(q_off + offs_d[None, :], mask=offs_m[:, None] < seqlen_q)
    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:
            k = tl.load(k_off + offs_d[None, :] + start_n * stride_kn)
        else:
            k = tl.load(
                k_off + offs_d[None, :] + start_n * stride_kn,
                mask=(start_n + offs_n)[:, None] < seqlen_k,
                other=0.0,
            )

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float16)

        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:
            # Need to mask out otherwise the softmax is wrong;
            # seems ok
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

        if IS_CAUSAL:
            qk += tl.where(
                offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf")
            )

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
            qk = qk * softmax_scale - m_ij[:, None]
            p = tl_exp(qk)

        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl_exp(m_i - m_ij).to(tl.float16)
        # # -- update output accumulator --

        acc_o = acc_o * acc_o_scale[:, None]

        # update acc_o
        if (
            EVEN_N & EVEN_M
        ):  # If we just do "if EVEN_N", there seems to be some race condition
            v = tl.load(v_off + offs_d[None, :] + start_n * stride_vn)
        else:
            v = tl.load(
                v_off + offs_d[None, :] + start_n * stride_vn,
                mask=(start_n + offs_n)[:, None] < seqlen_k,
                other=0.0,
            )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v, out_dtype=tl.float16)
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

    out_off = (
        Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om)
    )
    if EVEN_M:
        tl.store(out_off + offs_d[None, :], acc_o)
    else:
        tl.store(out_off + offs_d[None, :], acc_o, mask=offs_m[:, None] < seqlen_q)


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
                "Last 2 dimensions of bias must be (1, seqlen_k)"
                " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (
        (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)
    )

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty(
        (batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32
    )
    tmp = torch.empty(
        (batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32
    )
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (batch * nheads, triton.cdiv(seqlen_q, META["BLOCK_M"]))

    _fwd_kernel[grid](
        q,
        k,
        v,
        bias,
        o,
        lse,
        tmp,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,  # headdim
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type,
        causal,
        # BLOCK_M=BLOCK,
        # BLOCK_N=BLOCK,
        # num_warps=num_warps,
        # num_stages=1,
    )
    # print(f"_fwd_kernel.best_config ", _fwd_kernel.best_config, flush = True)
    return o, lse, softmax_scale  # softmax_scale could have been updated


class FlashAttentionV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.Any, q, k, v):
        o, lse, softmax_scale = _flash_attn_forward(q, k, v)
        return o


def call(q, k, v):
    return FlashAttentionV2.apply(q, k, v)


def bench_fn(q, k, v):
    fn = lambda: call(q, k, v)
    ms = triton.testing.do_bench(fn, warmup=100, rep=100)
    return ms


# register
name = "flash_attention_v2"
for dtype in [torch.bfloat16, torch.float16, torch.float32]:
    for device_ in ["cuda"]:
        b, s, h, d = SymVar("b"), SymVar("s"), SymVar("h"), SymVar("d")
        # we dont' actually allocate tensor
        q = Tensor((b, s, h, d), dtype=dtype, device=device_)
        k = Tensor((b, s, h, d), dtype=dtype, device=device_)
        v = Tensor((b, s, h, d), dtype=dtype, device=device_)
        # space = ChoiceSpace([])
        register_dlblas_op(name, None, (q, k, v), call, bench_fn, call)
