import dlblas
from dlblas.utils.gpu_helper import get_idle_device


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


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    import time

    # torch.manual_seed(1000)

    # Optionally use the context manager to ensure one of the fused kernels is run
    dtype = torch.float16
    device_ = torch.device("cuda:4")
    torch.cuda.set_device(device_)

    seq_len, heads, dim = 25600, 32, 128
    query = torch.rand([1, seq_len, heads, dim], dtype=dtype, device=device_)
    key = torch.rand([1, seq_len, heads, dim], dtype=dtype, device=device_)
    value = torch.rand([1, seq_len, heads, dim], dtype=dtype, device=device_)
    cos = torch.rand([1, seq_len, dim], dtype=dtype, device=device_)
    sin = torch.rand([1, seq_len, dim], dtype=dtype, device=device_)
    query_emb, key_emb = apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=2)
    ref_out = F.scaled_dot_product_attention(
        query_emb.permute(0, 2, 1, 3),
        key_emb.permute(0, 2, 1, 3),
        value.permute(0, 2, 1, 3),
    ).permute(0, 2, 1, 3)

    tt_out = dlblas.fused_rotary_and_fa(query, key, value, cos, sin)

    for i, j in zip(ref_out.shape, tt_out.shape):
        assert i == j

    print("TEST: ")
    # print(tt_out)
    print("max abs diff: ", torch.max(abs(tt_out - ref_out)))
    assert torch.allclose(tt_out, ref_out, atol=1e-2, rtol=0)
