# modify from: https://github.com/InternLM/lmdeploy
import torch
import triton
import triton.language as tl
from dlblas.utils import register_dlblas_op, SymVar, Tensor


@triton.jit
def apply_rotary_pos_emb_qk_kernel(
    Q,
    K,
    COS,
    SIN,
    POS,
    Q_EMB,
    K_EMB,
    seq_len,
    stride_qs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_qes: tl.constexpr,
    stride_qeh: tl.constexpr,
    stride_qed: tl.constexpr,
    stride_kes: tl.constexpr,
    stride_keh: tl.constexpr,
    stride_ked: tl.constexpr,
    half_size: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_QH: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """apply rotary on key AND query kernel."""
    seq_block_id = tl.program_id(0)

    pos_offset = seq_block_id * BLOCK + tl.arange(0, BLOCK)
    pos_ids = tl.load(POS + pos_offset, pos_offset < seq_len, other=-1)

    feat_size = half_size * 2
    feat_offset_l = tl.arange(0, BLOCK_N)
    feat_offset_h = half_size + feat_offset_l
    seq_mask = (pos_offset < seq_len)[:, None] & (feat_offset_l < half_size)[None, :]
    cs_offset_l = pos_ids[:, None] * feat_size + feat_offset_l[None, :]
    cs_offset_h = pos_ids[:, None] * feat_size + feat_offset_h[None, :]
    pos_ids_mask = pos_ids[:, None] >= 0
    cos_l = tl.load(COS + cs_offset_l, mask=pos_ids_mask)
    cos_h = tl.load(COS + cs_offset_h, mask=pos_ids_mask)
    sin_l = tl.load(SIN + cs_offset_l, mask=pos_ids_mask)
    sin_h = tl.load(SIN + cs_offset_h, mask=pos_ids_mask)

    q_ptr = Q + pos_offset * stride_qs
    qe_ptr = Q_EMB + pos_offset * stride_qes
    for hidx in range(BLOCK_QH):
        qh_ptr = q_ptr[:, None] + hidx * stride_qh
        q_l = tl.load(qh_ptr + feat_offset_l[None, :] * stride_qd, mask=seq_mask)
        q_h = tl.load(qh_ptr + feat_offset_h[None, :] * stride_qd, mask=seq_mask)
        qe_l = q_l * cos_l - q_h * sin_l
        qe_h = q_h * cos_h + q_l * sin_h

        qeh_ptr = qe_ptr[:, None] + hidx * stride_qeh
        tl.store(qeh_ptr + feat_offset_l[None, :] * stride_qed, qe_l, mask=seq_mask)
        tl.store(qeh_ptr + feat_offset_h[None, :] * stride_qed, qe_h, mask=seq_mask)

    k_ptr = K + pos_offset * stride_ks
    ke_ptr = K_EMB + pos_offset * stride_kes
    for hidx in range(BLOCK_KH):
        kh_ptr = k_ptr[:, None] + hidx * stride_kh
        k_l = tl.load(kh_ptr + feat_offset_l[None, :] * stride_kd, mask=seq_mask)
        k_h = tl.load(kh_ptr + feat_offset_h[None, :] * stride_kd, mask=seq_mask)
        ke_l = k_l * cos_l - k_h * sin_l
        ke_h = k_h * cos_h + k_l * sin_h

        keh_ptr = ke_ptr[:, None] + hidx * stride_keh
        tl.store(keh_ptr + feat_offset_l[None, :] * stride_ked, ke_l, mask=seq_mask)
        tl.store(keh_ptr + feat_offset_h[None, :] * stride_ked, ke_h, mask=seq_mask)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor = None,
    position_ids_1d: torch.Tensor = None,
    q_embed: torch.Tensor = None,
    k_embed: torch.Tensor = None,
):
    """Apply rotary positional embedding on query and key.

    Args:
        q (Tensor): Query state.
        k (Tensor): Key state.
        cos (Tensor): cosine matrix (seq_len, dim).
        sin (Tensor): sine matrix (seq_len, dim).
        position_ids (Tensor): Position ids of q and k.
        position_ids_1d (Tensor): 1d Position ids.
        q_embed (Tensor): output q, can be same as q
        k_embed (Tensor): output k, can be same as k

    Returns:
        Tuple[Tensor, Tensor]: Embedded query and key.
    """
    if cos.device != q.device or cos.dtype != q.dtype:
        cos = cos.to(device=q.device, dtype=q.dtype)
    if sin.device != q.device or sin.dtype != q.dtype:
        sin = sin.to(device=q.device, dtype=q.dtype)
    if position_ids_1d is None:
        seq_length = position_ids[..., -1] + 1
        position_ids_1d = [ids[:l] for ids, l in zip(position_ids, seq_length)]
        position_ids_1d = torch.cat(position_ids_1d)

    if q_embed is None:
        q_embed = torch.empty_like(q)
    if k_embed is None:
        k_embed = torch.empty_like(k)

    seq_len = position_ids_1d.size(-1)
    BLOCK = 32
    half_size = q.size(-1) // 2
    BLOCK_N = triton.next_power_of_2(half_size)
    num_heads_q = q.size(-2)
    num_heads_k = k.size(-2)
    num_warps = 4
    num_stages = 2

    grid = [triton.cdiv(seq_len, BLOCK)]
    apply_rotary_pos_emb_qk_kernel[grid](
        q,
        k,
        cos,
        sin,
        position_ids_1d,
        q_embed,
        k_embed,
        seq_len=seq_len,
        stride_qs=q.stride(-3),
        stride_qh=q.stride(-2),
        stride_qd=q.stride(-1),
        stride_ks=k.stride(-3),
        stride_kh=k.stride(-2),
        stride_kd=k.stride(-1),
        stride_qes=q_embed.stride(-3),
        stride_qeh=q_embed.stride(-2),
        stride_qed=q_embed.stride(-1),
        stride_kes=k_embed.stride(-3),
        stride_keh=k_embed.stride(-2),
        stride_ked=k_embed.stride(-1),
        half_size=half_size,
        BLOCK=BLOCK,
        BLOCK_QH=num_heads_q,
        BLOCK_KH=num_heads_k,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return q_embed, k_embed


class RotaryPosEmb(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.Any, q, k, cos, sin, position_ids_1d):
        q_embed, k_embed = apply_rotary_pos_emb(
            q, k, cos, sin, None, position_ids_1d, None, None
        )
        return q_embed, k_embed


def call(q, k, cos, sin, position_ids_1d):
    return RotaryPosEmb.apply(q, k, cos, sin, position_ids_1d)


def bench_fn(q, k, cos, sin, position_ids_1d):
    fn = lambda: call(q, k, cos, sin, position_ids_1d)
    ms = triton.testing.do_bench(fn, warmup=20, rep=20)
    return ms


# register
name = "apply_rotary_pos_emb"
for dtype in [torch.bfloat16, torch.float16, torch.float32]:
    for device_ in ["cuda"]:
        s, h, d = SymVar("s"), SymVar("h"), SymVar("d")
        # we dont' actually allocate tensor
        q = Tensor((s, h, d), dtype=dtype, device=device_)
        k = Tensor((s, h, d), dtype=dtype, device=device_)
        cos = Tensor((s, d), dtype=dtype, device=device_)
        sin = Tensor((s, d), dtype=dtype, device=device_)
        position_ids_1d = Tensor((s,), dtype=torch.int64, device=device_)
        # space = ChoiceSpace([])
        register_dlblas_op(
            name, None, (q, k, cos, sin, position_ids_1d), call, bench_fn, call
        )
