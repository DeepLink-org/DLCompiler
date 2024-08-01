import math
import torch
import triton
import triton.language as tl
# register
from dlblas import register_dlblas_op, SymVar, Tensor, ChoiceSpace


def get_fa_autotune_config():
    return [
        triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
        for BM in [64, 128] \
        for BN in [32, 64] \
        for s in [2, 3, 4] \
        for w in [2, 4, 8] \
    ]

@triton.heuristics(
    {
        "DIVISIBLE_M": lambda args: args["M"] % args["BLOCK_M"] == 0,
        "DIVISIBLE_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _fa_fwd_kernel(
    Q, K, V, sm_scale,
    dropout_p,
    seed,
    offset,
    L, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N, P_SEQ,
    num_groups,
    head_dim,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, IS_DROPOUT: tl.constexpr, LARGER_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    O += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * M # l's shape is (B, H, M)

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    if IS_DROPOUT:
        rowblock_base = off_z * H * M * N + off_h * M * N + start_m * BLOCK_M * N
        offs_rng_base = offset + rowblock_base
        offs_rng_base += tl.arange(0, BLOCK_M)[:, None] * N
        offs_rng_base += tl.arange(0, BLOCK_N)[None, :]

    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok) # (BLOCK_M, BLOCK_DMODEL)
    l_ptrs = L + offs_m

    # initialize pointer to m and l, fp32 for accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, mask=offs_k[None, :] < head_dim, cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < head_dim), cache_modifier=".cg")

    if IS_CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N

    # loop over k, v and update accumulators
    offs_n_init = offs_n_base
    k_ptrs = K + (offs_k[:, None] * stride_vk + offs_n_init[None, :] * stride_vn)
    v_ptrs = V + (offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        # -- load k, v --
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, mask=offs_k[:, None] < head_dim, cache_modifier=".cg")
            v = tl.load(v_ptrs, mask=offs_k[None, :] < head_dim, cache_modifier=".cg")
        else:
            mask_n = offs_n < N
            k = tl.load(k_ptrs, mask=mask_n[None, :] & (offs_k[:, None] < head_dim), cache_modifier=".cg")
            v = tl.load(v_ptrs, mask=mask_n[:, None] & (offs_k[None, :] < head_dim), cache_modifier=".cg")

        # -- compute qk ---
        s = tl.dot(q, k)

        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float("-inf"))
        if IS_CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
        p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)

        # -- compute partial sumexpn before applying dropout
        p_sum = tl.sum(p, 1)

        # -- apply dropout --
        if IS_DROPOUT:
            offs_rng = start_n + offs_rng_base
            pmask = tl.rand(seed, offs_rng, n_rounds=6) > dropout_p
            p *= pmask.to(tl.float32)

        # -- scale and update acc: acc *= alpha[:, None]--
        acc *= alpha[:, None]
        acc += tl.dot(p.to(input_dtype), v)

        # -- update m_i and l_i --
        l_i = l_i * alpha + p_sum
        m_i = m_i_new
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    # write back l & o
    if IS_CAUSAL and LARGER_M:
        is_empty_line = (offs_m + P_SEQ) < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l = tl.where(is_empty_line, float("-inf"), m_i * sm_scale + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l = m_i * sm_scale + tl.log(l_i) # log(normalizer)

    # -- scale o due to dropout
    if IS_DROPOUT:
        scale = 1.0 / (1.0 - dropout_p)
        acc *= scale

    if DIVISIBLE_M:
        tl.store(l_ptrs, l, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), mask=offs_k[None, :] < head_dim, cache_modifier=".cg")
    else:
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None] & (offs_k[None, :] < head_dim), cache_modifier=".cg")


def call(q, k, v, causal, sm_scale, dropout_p):
    Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Dq == Dk == Dv, "feature size of q, k, v should be equal"
    assert Dk in {16, 32, 64, 128}

    B, H, M, D = q.shape
    N = k.shape[2]
    Hk, Hv = k.shape[1], v.shape[1]
    assert Hk == Hv, "num of heads in k and v should be equal"
    assert H % Hk == 0, "number of heads in q must be a multiple of that in k & v"
    num_groups = H // Hk

    P_SEQ = N - M
    larger_m = M > N

    if sm_scale is None:
        sm_scale = 1. / math.sqrt(D)

    is_dropout = dropout_p > 0
    if is_dropout:
        offset_increment = B * H * M * N
        # seed, offset = philox_cuda_seed_offset(offset_increment)
    else:
        seed, offset = 0, 0

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), H, B)
    o = torch.empty_like(q)
    L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)

    _fa_fwd_kernel[grid](
                    q, k, v, sm_scale,
                    dropout_p, seed, offset,
                    L, o,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                    B, H, M, N, P_SEQ, num_groups, D,
                    BLOCK_DMODEL=triton.next_power_of_2(D),
                    IS_CAUSAL=causal, IS_DROPOUT=is_dropout, LARGER_M=larger_m,
                )
    return o


def bench_fn(q, k, v, causal=False, sm_scale=0.0, dropout_p=0.0):
    fn = lambda: call(q, k, v, causal, sm_scale, dropout_p)
    ms = triton.testing.do_bench(fn, warmup=100, rep=100)
    return ms


# register
name = 'flash_attention'
for dtype in [torch.float16, torch.float32]:
    # for now, epilogue is not added to op name
    for device in ['cuda']:
        batch, seqLenQ = SymVar('batch'), SymVar('seqLenQ')
        headN, headD = SymVar('headN'), SymVar('headD')
        seqLenKV = SymVar('seqLenKV')
        # we dont' actually allocate tensor
        q = Tensor((batch, headN, seqLenQ, headD), dtype=dtype, device=device)
        k = Tensor((batch, headN, seqLenKV, headD), dtype=dtype, device=device)
        v = Tensor((batch, headN, seqLenKV, headD), dtype=dtype, device=device)

        # NOTE: the underlying kernel is the same jit'ed function, but Triton
        # will dispatch to different kernels based on the input params
        #
        # why do we still need another dispatch layer in op_registry?
        # because e.g. matmul may have different Triton implemetation...
        #
        space = ChoiceSpace(get_fa_autotune_config())
        register_dlblas_op(name, space, (q, k, v), call, bench_fn, _fa_fwd_kernel)

