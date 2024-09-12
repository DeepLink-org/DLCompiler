import torch
import triton
import triton.language as tl
from dlblas.utils.libentry import libentry
from dlblas.utils import register_dlblas_op, SymVar, Tensor, ChoiceSpace


@triton.jit
def _get_cos_sin(
    batch_idx,
    offs_seq,
    rope_dim_range0,
    rope_dim_range1,
    seq_len,
    rope_head_dim,
    cos,
    sin,
    stride_cos_bsz,
    stride_cos_seq,
    stride_cos_dim,
):
    offs_cs0 = (
        batch_idx * stride_cos_bsz
        + offs_seq[:, None] * stride_cos_seq
        + rope_dim_range0[None, :] * stride_cos_dim
    )
    offs_cs1 = (
        batch_idx * stride_cos_bsz
        + offs_seq[:, None] * stride_cos_seq
        + rope_dim_range1[None, :] * stride_cos_dim
    )
    cos0_data = tl.load(
        cos + offs_cs0,
        mask=(offs_seq[:, None] < seq_len) & (rope_dim_range0[None, :] < rope_head_dim),
    )
    cos1_data = tl.load(
        cos + offs_cs1,
        mask=(offs_seq[:, None] < seq_len) & (rope_dim_range1[None, :] < rope_head_dim),
    )
    sin0_data = tl.load(
        sin + offs_cs0,
        mask=(offs_seq[:, None] < seq_len) & (rope_dim_range0[None, :] < rope_head_dim),
    )
    sin1_data = tl.load(
        sin + offs_cs1,
        mask=(offs_seq[:, None] < seq_len) & (rope_dim_range1[None, :] < rope_head_dim),
    )
    return cos0_data, cos1_data, sin0_data, sin1_data


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SEQ": BS}, num_stages=s, num_warps=w)
        for BS in [1, 2, 4, 8]
        for s in [1, 2]
        for w in [1, 2]
    ],
    key=["seq_len", "q_head_dim", "v_head_dim"],
)
@triton.jit
def _fwd_kernel(
    q,
    k_pe,
    kv,
    cos,
    sin,
    out_kv,
    stride_q_bsz,
    stride_q_seq,
    stride_q_head,
    stride_q_dim,
    stride_kpe_bsz,
    stride_kpe_seq,
    stride_kpe_head,
    stride_kpe_dim,
    stride_cos_bsz,
    stride_cos_seq,
    stride_cos_dim,
    stride_okv_bsz,
    stride_okv_seq,
    stride_okv_kv,
    stride_okv_head,
    stride_okv_dim,
    stride_kv_bsz,
    stride_kv_seq,
    stride_kv_head,
    stride_kv_dim,
    seq_len,
    q_head_dim,
    rope_head_dim,
    nope_head_dim,
    v_head_dim,
    v_pad_dim,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_ROPE_DIM: tl.constexpr,
    BLOCK_NOPE_DIM: tl.constexpr,
    BLOCK_V_DIM: tl.constexpr,
    BLOCK_V_PAD_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    offs_seq = seq_idx * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    v_dims = tl.arange(0, BLOCK_V_DIM)
    v_pad_dims = tl.arange(0, BLOCK_V_PAD_DIM)
    nope_dims = tl.arange(0, BLOCK_NOPE_DIM)
    rope_dim_range0 = tl.arange(0, BLOCK_ROPE_DIM // 2)
    rope_dim_range1 = tl.arange(BLOCK_ROPE_DIM // 2, BLOCK_ROPE_DIM)
    rope_dim0_interleaved = 2 * rope_dim_range0
    rope_dim1_interleaved = 1 + 2 * rope_dim_range0
    offs_q0 = (
        batch_idx * stride_q_bsz
        + offs_seq[:, None] * stride_q_seq
        + head_idx * stride_q_head
    )
    offs_q1 = (
        batch_idx * stride_q_bsz
        + offs_seq[:, None] * stride_q_seq
        + head_idx * stride_q_head
    )
    q0_data = tl.load(
        q + offs_q0 + (nope_head_dim + rope_dim0_interleaved[None, :]) * stride_q_dim,
        mask=(offs_seq[:, None] < seq_len)
        & (rope_dim0_interleaved[None, :] < q_head_dim),
    )
    q1_data = tl.load(
        q + offs_q1 + (nope_head_dim + rope_dim1_interleaved[None, :]) * stride_q_dim,
        mask=(offs_seq[:, None] < seq_len)
        & (rope_dim1_interleaved[None, :] < q_head_dim),
    )

    # offs_cs0 = (
    #     batch_idx * stride_cos_bsz
    #     + offs_seq[:, None] * stride_cos_seq
    #     + rope_dim_range0[None, :] * stride_cos_dim
    # )
    # offs_cs1 = (
    #     batch_idx * stride_cos_bsz
    #     + offs_seq[:, None] * stride_cos_seq
    #     + rope_dim_range1[None, :] * stride_cos_dim
    # )
    # cos0_data = tl.load(
    #     cos + offs_cs0,
    #     mask=(offs_seq[:, None] < seq_len) & (rope_dim_range0[None, :] < rope_head_dim),
    # )
    # cos1_data = tl.load(
    #     cos + offs_cs1,
    #     mask=(offs_seq[:, None] < seq_len) & (rope_dim_range1[None, :] < rope_head_dim),
    # )
    # sin0_data = tl.load(
    #     sin + offs_cs0,
    #     mask=(offs_seq[:, None] < seq_len) & (rope_dim_range0[None, :] < rope_head_dim),
    # )
    # sin1_data = tl.load(
    #     sin + offs_cs1,
    #     mask=(offs_seq[:, None] < seq_len) & (rope_dim_range1[None, :] < rope_head_dim),
    # )
    cos0_data, cos1_data, sin0_data, sin1_data = _get_cos_sin(
        batch_idx,
        offs_seq,
        rope_dim_range0,
        rope_dim_range1,
        seq_len,
        rope_head_dim,
        cos,
        sin,
        stride_cos_bsz,
        stride_cos_seq,
        stride_cos_dim,
    )
    # compute q
    out_q0_data = q0_data * cos0_data - q1_data * sin0_data
    out_q1_data = q1_data * cos1_data + q0_data * sin1_data
    tl.store(
        q + offs_q0 + (nope_head_dim + rope_dim_range0[None, :]) * stride_q_dim,
        out_q0_data,
        mask=(offs_seq[:, None] < seq_len) & (rope_dim_range0[None, :] < q_head_dim),
    )
    tl.store(
        q + offs_q1 + (nope_head_dim + rope_dim_range1[None, :]) * stride_q_dim,
        out_q1_data,
        mask=(offs_seq[:, None] < seq_len) & (rope_dim_range1[None, :] < q_head_dim),
    )
    # read k_nope from kv, then write to out_kv
    offs_kv = (
        batch_idx * stride_kv_bsz
        + offs_seq[:, None] * stride_kv_seq
        + head_idx * stride_kv_head
    )
    k_nope_data = tl.load(
        kv + offs_kv + nope_dims * stride_kv_dim,
        mask=(offs_seq[:, None] < seq_len) & (nope_dims[None, :] < nope_head_dim),
    )
    offs_okv_k_nope = (
        batch_idx * stride_okv_bsz
        + offs_seq[:, None] * stride_okv_seq
        + head_idx * stride_okv_head
        + nope_dims * stride_okv_dim
    )
    tl.store(
        out_kv + offs_okv_k_nope,
        k_nope_data,
        mask=(offs_seq[:, None] < seq_len) & (nope_dims[None, :] < nope_head_dim),
    )
    # load k_pe, k_pe num_heads is 1
    offs_kpe0 = (
        batch_idx * stride_kpe_bsz
        + offs_seq[:, None] * stride_kpe_seq
        + rope_dim0_interleaved[None, :] * stride_kpe_dim
    )
    offs_kpe1 = (
        batch_idx * stride_kpe_bsz
        + offs_seq[:, None] * stride_kpe_seq
        + rope_dim1_interleaved[None, :] * stride_kpe_dim
    )
    kpe0_data = tl.load(
        k_pe + offs_kpe0,
        mask=(offs_seq[:, None] < seq_len)
        & (rope_dim0_interleaved[None, :] < rope_head_dim),
    )
    kpe1_data = tl.load(
        k_pe + offs_kpe1,
        mask=(offs_seq[:, None] < seq_len)
        & (rope_dim1_interleaved[None, :] < rope_head_dim),
    )
    out_kpe0_data = kpe0_data * cos0_data - kpe1_data * sin0_data
    out_kpe1_data = kpe1_data * cos1_data + kpe0_data * sin1_data
    # write out_kpe to out_kv
    offs_okv_kpe = (
        batch_idx * stride_okv_bsz
        + offs_seq[:, None] * stride_okv_seq
        + head_idx * stride_okv_head
    )
    tl.store(
        out_kv + offs_okv_kpe + (nope_head_dim + rope_dim_range0) * stride_okv_dim,
        out_kpe0_data,
        mask=(offs_seq[:, None] < seq_len) & (rope_dim_range0[None, :] < rope_head_dim),
    )
    tl.store(
        out_kv + offs_okv_kpe + (nope_head_dim + rope_dim_range1) * stride_okv_dim,
        out_kpe1_data,
        mask=(offs_seq[:, None] < seq_len) & (rope_dim_range1[None, :] < rope_head_dim),
    )
    # write v to out_kv
    v_data = tl.load(
        kv + offs_kv + (nope_head_dim + v_dims) * stride_kv_dim,
        mask=(offs_seq[:, None] < seq_len) & (v_dims[None, :] < v_head_dim),
    )
    tl.store(
        out_kv
        + batch_idx * stride_okv_bsz
        + offs_seq[:, None] * stride_okv_seq
        + stride_okv_kv
        + head_idx * stride_okv_head
        + v_dims * stride_okv_dim,
        v_data,
        mask=(offs_seq[:, None] < seq_len) & (v_dims[None, :] < v_head_dim),
    )
    tl.store(
        out_kv
        + batch_idx * stride_okv_bsz
        + offs_seq[:, None] * stride_okv_seq
        + stride_okv_kv
        + head_idx * stride_okv_head
        + (v_head_dim + v_pad_dims) * stride_okv_dim,
        tl.zeros((BLOCK_SEQ, BLOCK_V_PAD_DIM), kv.dtype.element_ty),
        mask=(offs_seq[:, None] < seq_len) & (v_pad_dims[None, :] < v_pad_dim),
    )


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SEQ": BS}, num_stages=s, num_warps=w)
        for BS in [1, 2, 4, 8]
        for s in [1, 2]
        for w in [1, 2]
    ],
    key=["seq_len", "q_head_dim", "rope_head_dim"],
)
@triton.jit
def _bwd_kernel(
    d_q,
    d_kv,
    cos,
    sin,
    do_q,
    do_k_pe,
    do_kv,
    stride_dq_bsz,
    stride_dq_seq,
    stride_dq_head,
    stride_dq_dim,
    stride_dkv_bsz,
    stride_dkv_seq,
    stride_dkv_kv,
    stride_dkv_head,
    stride_dkv_dim,
    stride_cos_bsz,
    stride_cos_seq,
    stride_cos_dim,
    stride_dokpe_bsz,
    stride_dokpe_seq,
    stride_dokpe_head,
    stride_dokpe_dim,
    stride_kv_bsz,
    stride_kv_seq,
    stride_kv_head,
    stride_kv_dim,
    seq_len,
    q_head_dim,
    rope_head_dim,
    nope_head_dim,
    v_head_dim,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_ROPE_DIM: tl.constexpr,
    BLOCK_NOPE_DIM: tl.constexpr,
    BLOCK_V_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    offs_seq = seq_idx * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    v_dims = tl.arange(0, BLOCK_V_DIM)
    nope_dim_range = tl.arange(0, BLOCK_NOPE_DIM)
    rope_dim_range0 = tl.arange(0, BLOCK_ROPE_DIM // 2)
    rope_dim_range1 = tl.arange(BLOCK_ROPE_DIM // 2, BLOCK_ROPE_DIM)
    rope_dim0_interleaved = 2 * rope_dim_range0
    rope_dim1_interleaved = 1 + 2 * rope_dim_range0
    offs_dq = (
        batch_idx * stride_dq_bsz
        + offs_seq[:, None] * stride_dq_seq
        + head_idx * stride_dq_head
    )
    dq0_data = tl.load(
        d_q + offs_dq + (nope_head_dim + rope_dim_range0[None, :]) * stride_dq_dim,
        mask=(offs_seq[:, None] < seq_len) & (rope_dim_range0[None, :] < q_head_dim),
    )
    dq1_data = tl.load(
        d_q + offs_dq + (nope_head_dim + rope_dim_range1[None, :]) * stride_dq_dim,
        mask=(offs_seq[:, None] < seq_len) & (rope_dim_range1[None, :] < q_head_dim),
    )
    cos0_data, cos1_data, sin0_data, sin1_data = _get_cos_sin(
        batch_idx,
        offs_seq,
        rope_dim_range0,
        rope_dim_range1,
        seq_len,
        rope_head_dim,
        cos,
        sin,
        stride_cos_bsz,
        stride_cos_seq,
        stride_cos_dim,
    )
    do_q0_data = dq1_data * sin1_data + dq0_data * cos0_data
    do_q1_data = dq1_data * cos1_data - dq0_data * sin0_data
    tl.store(
        do_q
        + offs_dq
        + (nope_head_dim + rope_dim0_interleaved[None, :]) * stride_dq_dim,
        do_q0_data,
        mask=(offs_seq[:, None] < seq_len)
        & (rope_dim0_interleaved[None, :] < q_head_dim),
    )
    tl.store(
        do_q
        + offs_dq
        + (nope_head_dim + rope_dim1_interleaved[None, :]) * stride_dq_dim,
        do_q1_data,
        mask=(offs_seq[:, None] < seq_len)
        & (rope_dim1_interleaved[None, :] < q_head_dim),
    )
    dq_nope_data = tl.load(
        d_q + offs_dq + nope_dim_range[None, :] * stride_dq_dim,
        mask=(offs_seq[:, None] < seq_len) & (nope_dim_range[None, :] < nope_head_dim),
    )
    tl.store(
        do_q + offs_dq + nope_dim_range * stride_dq_dim,
        dq_nope_data,
        mask=(offs_seq[:, None] < seq_len) & (nope_dim_range[None, :] < nope_head_dim),
    )
    # for do_k_pe
    offs_d_kv = (
        batch_idx * stride_dkv_bsz
        + offs_seq[:, None] * stride_dkv_seq
        + head_idx * stride_dkv_head
    )
    d_k_pe0_data = tl.load(
        d_kv + offs_d_kv + (nope_head_dim + rope_dim_range0[None, :]) * stride_dkv_dim,
        mask=(offs_seq[:, None] < seq_len) & (rope_dim_range0[None, :] < q_head_dim),
    )
    d_k_pe1_data = tl.load(
        d_kv + offs_d_kv + (nope_head_dim + rope_dim_range1[None, :]) * stride_dkv_dim,
        mask=(offs_seq[:, None] < seq_len) & (rope_dim_range1[None, :] < q_head_dim),
    )
    do_kpe0_data = d_k_pe1_data * sin1_data + d_k_pe0_data * cos0_data
    do_kpe1_data = d_k_pe1_data * cos1_data - d_k_pe0_data * sin0_data
    offs_do_kpe = (
        batch_idx * stride_dokpe_bsz
        + offs_seq[:, None] * stride_dokpe_seq
        + head_idx * stride_dokpe_head
    )
    tl.store(
        do_k_pe + offs_do_kpe + rope_dim0_interleaved[None, :],
        do_kpe0_data,
        mask=(offs_seq[:, None] < seq_len)
        & (rope_dim0_interleaved[None, :] < rope_head_dim),
    )
    tl.store(
        do_k_pe + offs_do_kpe + rope_dim1_interleaved[None, :],
        do_kpe1_data,
        mask=(offs_seq[:, None] < seq_len)
        & (rope_dim1_interleaved[None, :] < rope_head_dim),
    )
    # for do_kv
    d_k_nope_data = tl.load(
        d_kv + offs_d_kv + nope_dim_range[None, :] * stride_dkv_dim,
        mask=(offs_seq[:, None] < seq_len) & (nope_dim_range[None, :] < nope_head_dim),
    )
    d_v_data = tl.load(
        d_kv + offs_d_kv + stride_dkv_kv + v_dims[None, :] * stride_dkv_dim,
        mask=(offs_seq[:, None] < seq_len) & (v_dims[None, :] < v_head_dim),
    )
    offs_do_kv = (
        batch_idx * stride_kv_bsz
        + offs_seq[:, None] * stride_kv_seq
        + head_idx * stride_kv_head
    )
    tl.store(
        do_kv + offs_do_kv + nope_dim_range[None, :],
        d_k_nope_data,
        mask=(offs_seq[:, None] < seq_len) & (nope_dim_range[None, :] < nope_head_dim),
    )
    tl.store(
        do_kv + offs_do_kv + nope_head_dim + v_dims[None, :],
        d_v_data,
        mask=(offs_seq[:, None] < seq_len) & (v_dims[None, :] < v_head_dim),
    )


class PartialRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.Any, q, k_pe, kv, cos, sin):
        bsz, seq_len, num_heads, q_head_dim = q.shape
        assert bsz == k_pe.shape[0] and seq_len == k_pe.shape[1] and 1 == k_pe.shape[2]
        qk_rope_head_dim = k_pe.shape[3]
        assert qk_rope_head_dim == triton.next_power_of_2(qk_rope_head_dim)
        qk_nope_head_dim = q_head_dim - qk_rope_head_dim

        assert (
            bsz == kv.shape[0] and seq_len == kv.shape[1] and num_heads == kv.shape[2]
        )
        v_head_dim = kv.shape[3] - qk_nope_head_dim
        assert (
            bsz == cos.shape[0]
            and seq_len == cos.shape[1]
            and qk_rope_head_dim == cos.shape[2]
        )
        assert cos.shape == sin.shape
        stride_q_bsz, stride_q_seq, stride_q_head, stride_q_dim = q.stride()
        stride_kpe_bsz, stride_kpe_seq, stride_kpe_head, stride_kpe_dim = k_pe.stride()
        stride_cos_bsz, stride_cos_seq, stride_cos_dim = cos.stride()
        out_kv = kv.new_empty(bsz, seq_len, 2, num_heads, q_head_dim)
        with torch.cuda.device(q.device):
            grid = lambda META: (
                bsz,
                triton.cdiv(seq_len, META["BLOCK_SEQ"]),
                num_heads,
            )
            _fwd_kernel[grid](
                q,
                k_pe,
                kv,
                cos,
                sin,
                out_kv,
                stride_q_bsz,
                stride_q_seq,
                stride_q_head,
                stride_q_dim,
                stride_kpe_bsz,
                stride_kpe_seq,
                stride_kpe_head,
                stride_kpe_dim,
                stride_cos_bsz,
                stride_cos_seq,
                stride_cos_dim,
                stride_okv_bsz=out_kv.stride(0),
                stride_okv_seq=out_kv.stride(1),
                stride_okv_kv=out_kv.stride(2),
                stride_okv_head=out_kv.stride(3),
                stride_okv_dim=out_kv.stride(4),
                stride_kv_bsz=kv.stride(0),
                stride_kv_seq=kv.stride(1),
                stride_kv_head=kv.stride(2),
                stride_kv_dim=kv.stride(3),
                seq_len=seq_len,
                q_head_dim=q_head_dim,
                rope_head_dim=qk_rope_head_dim,
                nope_head_dim=qk_nope_head_dim,
                v_head_dim=v_head_dim,
                v_pad_dim=q_head_dim - v_head_dim,
                BLOCK_ROPE_DIM=triton.next_power_of_2(qk_rope_head_dim),
                BLOCK_NOPE_DIM=triton.next_power_of_2(qk_nope_head_dim),
                BLOCK_V_DIM=triton.next_power_of_2(v_head_dim),
                BLOCK_V_PAD_DIM=triton.next_power_of_2(q_head_dim - v_head_dim),
            )
            # print(
            #     f"_partial_rotary_emb_kernel.best_config ",
            #     _partial_rotary_emb_kernel.best_config,
            # )
        ctx.save_for_backward(kv, cos, sin)
        return q, out_kv

    @staticmethod
    def backward(ctx, d_q, d_kv):
        kv, cos, sin = ctx.saved_tensors
        bsz, seq_len, num_heads, q_head_dim = d_q.shape
        rope_head_dim = cos.shape[-1]
        nope_head_dim = q_head_dim - rope_head_dim
        v_head_dim = kv.shape[3] - nope_head_dim
        do_q = d_q.new_empty(bsz, seq_len, num_heads, q_head_dim)
        do_k_pe_tmp = d_q.new_empty(bsz, seq_len, num_heads, rope_head_dim)
        do_kv = torch.empty_like(kv)
        with torch.cuda.device(d_q.device):
            grid = lambda META: (
                bsz,
                triton.cdiv(seq_len, META["BLOCK_SEQ"]),
                num_heads,
            )
            _bwd_kernel[grid](
                d_q,
                d_kv,
                cos,
                sin,
                do_q,
                do_k_pe_tmp,
                do_kv,
                stride_dq_bsz=d_q.stride(0),
                stride_dq_seq=d_q.stride(1),
                stride_dq_head=d_q.stride(2),
                stride_dq_dim=d_q.stride(3),
                stride_dkv_bsz=d_kv.stride(0),
                stride_dkv_seq=d_kv.stride(1),
                stride_dkv_kv=d_kv.stride(2),
                stride_dkv_head=d_kv.stride(3),
                stride_dkv_dim=d_kv.stride(4),
                stride_cos_bsz=cos.stride(0),
                stride_cos_seq=cos.stride(1),
                stride_cos_dim=cos.stride(2),
                stride_dokpe_bsz=do_k_pe_tmp.stride(0),
                stride_dokpe_seq=do_k_pe_tmp.stride(1),
                stride_dokpe_head=do_k_pe_tmp.stride(2),
                stride_dokpe_dim=do_k_pe_tmp.stride(3),
                stride_kv_bsz=kv.stride(0),
                stride_kv_seq=kv.stride(1),
                stride_kv_head=kv.stride(2),
                stride_kv_dim=kv.stride(3),
                seq_len=seq_len,
                q_head_dim=q_head_dim,
                rope_head_dim=rope_head_dim,
                nope_head_dim=nope_head_dim,
                v_head_dim=v_head_dim,
                BLOCK_ROPE_DIM=triton.next_power_of_2(rope_head_dim),
                BLOCK_NOPE_DIM=triton.next_power_of_2(nope_head_dim),
                BLOCK_V_DIM=triton.next_power_of_2(v_head_dim),
            )
        do_k_pe = torch.sum(do_k_pe_tmp, dim=2, keepdim=True)
        return do_q, do_k_pe, do_kv, None, None


def call(q, k_pe, kv, cos, sin):
    return PartialRotaryEmb.apply(q, k_pe, kv, cos, sin)


def bench_fn(q, k_pe, kv, cos, sin):
    fn = lambda: call(q, k_pe, kv, cos, sin)
    ms = triton.testing.do_bench(fn, warmup=20, rep=20)
    return ms


# register
name = "partial_rotary_emb"
for dtype in [torch.bfloat16, torch.float16, torch.float32]:
    for device_ in ["cuda"]:
        num_heads = SymVar("num_heads")
        qk_nope_head_dim = SymVar("qk_nope_head_dim")
        qk_rope_head_dim = SymVar("qk_rope_head_dim")
        v_head_dim = SymVar("v_head_dim")
        q_head_dim = SymVar("q_head_dim")
        bsz, q_len = SymVar("bsz"), SymVar("q_len")
        # we dont' actually allocate tensor
        q = Tensor((bsz, q_len, num_heads, q_head_dim), dtype=dtype, device=device_)
        k_pe = Tensor(
            (bsz, q_len, SymVar("one"), qk_rope_head_dim),
            dtype=dtype,
            device=device_,
        )
        kv = Tensor(
            (bsz, q_len, num_heads, SymVar("qk_nope_head_dim + v_head_dim")),
            dtype=dtype,
            device=device_,
        )
        cos = Tensor((bsz, q_len, qk_rope_head_dim), dtype=dtype, device=device_)
        sin = Tensor((bsz, q_len, qk_rope_head_dim), dtype=dtype, device=device_)
        # space = ChoiceSpace([])
        register_dlblas_op(name, None, (q, k_pe, kv, cos, sin), call, bench_fn, call)
