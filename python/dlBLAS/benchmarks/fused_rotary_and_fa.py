import triton
import dlblas
from python.dlBLAS.dlblas.utils.device_utils import get_idle_device
import torch
import torch.nn.functional as F


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    # out1.copy_(x1 * cos - x2 * sin)
    # out2.copy_(x2 * cos + x1 * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def none_fused_rotary_and_fa(
    seq_len, heads, dim, query, key, value, cos, sin, position_ids_1d
):
    query_emb, key_emb = dlblas.apply_rotary_pos_emb(
        query.view(seq_len, heads, dim),
        key.view(seq_len, heads, dim),
        cos.view(seq_len, dim),
        sin.view(seq_len, dim),
        position_ids_1d,
    )
    return dlblas.flash_attention_v2(
        query_emb.view(1, seq_len, heads, dim),
        key_emb.view(1, seq_len, heads, dim),
        value,
    )


def test():
    device_ = torch.device(get_idle_device())
    torch.cuda.set_device(device_)
    dtype = torch.float16

    seq_len, heads, dim = 25600, 32, 64
    query = torch.rand([1, seq_len, heads, dim], dtype=dtype, device=device_)
    key = torch.rand([1, seq_len, heads, dim], dtype=dtype, device=device_)
    value = torch.rand([1, seq_len, heads, dim], dtype=dtype, device=device_)
    cos = torch.rand([1, seq_len, dim], dtype=dtype, device=device_)
    sin = torch.rand([1, seq_len, dim], dtype=dtype, device=device_)
    position_ids_1d = torch.arange(0, seq_len, device=device_)
    # query_emb, key_emb = apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=2)
    ref_out = none_fused_rotary_and_fa(
        seq_len, heads, dim, query, key, value, cos, sin, position_ids_1d
    )

    tt_out = dlblas.fused_rotary_and_fa(query, key, value, cos, sin)

    for i, j in zip(ref_out.shape, tt_out.shape):
        assert i == j

    print("TEST: ")
    # print(tt_out)
    print("max abs diff: ", torch.max(abs(tt_out - ref_out)))
    assert torch.allclose(tt_out, ref_out, atol=1e-2, rtol=0)

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["op"],
            x_vals=["fwd"],
            line_arg="provider",
            line_vals=["fused", "none-fused", "rotary"],
            line_names=["fused", "none-fused", "rotary"],
            ylabel="ms",
            plot_name=f"fused_rotary_emb_and_fa(batchSize={1}, seqlen:{seq_len}, num_heads:{heads}, dim:{dim})",
            args={"SeqLen": seq_len},
        )
    )

    @triton.testing.perf_report(configs)
    def bench_fn(SeqLen, op, provider, device=device_):
        warmup = 200
        rep = 200

        if "fused" in provider:
            fn = lambda: dlblas.fused_rotary_and_fa(query, key, value, cos, sin)

        if "none-fused" in provider:
            fn = lambda: none_fused_rotary_and_fa(
                seq_len, heads, dim, query, key, value, cos, sin, position_ids_1d
            )
        if "rotary" in provider:
            fn = lambda: dlblas.apply_rotary_pos_emb(
                query.view(seq_len, heads, dim),
                key.view(seq_len, heads, dim),
                cos.view(seq_len, dim),
                sin.view(seq_len, dim),
                position_ids_1d,
            )
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    test()
    print("sucessfully!")
