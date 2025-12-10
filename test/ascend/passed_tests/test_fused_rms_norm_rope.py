import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.deeplink as dl


DEVICE = "npu"


def _rms_norm_kernel(input: torch.Tensor, weight: torch.Tensor, epsilon: float) -> None:
    # TODO: Remove this contiguous call when the kernel is updated to support non-contiguous input
    # If removed, also need to remove contiguous in MatcherRMSNorm
    input_contiguous = input.contiguous()
    return torch.nn.functional.rms_norm(input_contiguous, weight.shape, weight, epsilon)


def _compute_inv_freq(base: float, head_size) -> torch.Tensor:
    """Compute the inverse frequency."""
    # NOTE(woosuk): To exactly match the HF implementation, we need to
    # use CPU to compute the cache and then move it to GPU. However, we
    # create the cache on GPU for faster initialization. This may cause
    # a slight numerical difference between the HF implementation and ours.
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, head_size, 2, dtype=torch.float, device="npu") / head_size)
    )
    return inv_freq


def _compute_cos_sin_cache(head_size) -> torch.Tensor:
    """Compute the cos and sin cache."""
    inv_freq = _compute_inv_freq(10000.0, head_size)
    t = torch.arange(4096, dtype=torch.float32, device="npu")

    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)
    return cache


def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def _rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_shape = query.shape
    query = query.view(num_tokens, -1, head_size)
    query_rot = query[..., :head_size]
    query_pass = query[..., head_size:]
    query_rot = apply_rotary_emb_torch(query_rot, cos, sin, is_neox)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, head_size)
    key_rot = key[..., :head_size]
    key_pass = key[..., head_size:]
    key_rot = apply_rotary_emb_torch(key_rot, cos, sin, is_neox)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


@triton.jit
def _compute_rotary_emb(
    x1,
    x2,
    cos,
    sin,
):
    cos = tl.expand_dims(cos, -2)
    sin = tl.expand_dims(sin, -2)
    cos = cos.to(tl.float32)
    sin = sin.to(tl.float32)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return o1, o2


@triton.jit
def rms_norm_rope_kernel(
    q1,
    q2,
    k1,
    k2,
    v,
    weight,
    cos,
    sin,
    q1_out,
    q2_out,
    k1_out,
    k2_out,
    v_out,
    q_size: tl.constexpr,
    kv_size: tl.constexpr,
    head_dim: tl.constexpr,
    num_tokens: tl.constexpr,
):
    half_q_offsets = tl.arange(0, q_size // 2)
    half_kv_offsets = tl.arange(0, kv_size // 2)
    num_q_offsets = tl.arange(0, q_size // head_dim)
    num_kv_offsets = tl.arange(0, kv_size // head_dim)
    head_offsets = tl.arange(0, head_dim)
    half_head_offsets = tl.arange(0, head_dim // 2)
    num_token_offsets = tl.arange(0, num_tokens)
    half_q_size = q_size // 2
    half_kv_size = kv_size // 2
    half_dim = head_dim // 2

    with dl.async_task(scope=dl.async_task.vector):
        q1_data = tl.load(
            q1 + num_token_offsets[:, None] * half_q_size + half_q_offsets[None, :]
        )
        q2_data = tl.load(
            q2 + num_token_offsets[:, None] * half_q_size + half_q_offsets[None, :]
        )
        k1_data = tl.load(
            k1 + num_token_offsets[:, None] * half_kv_size + half_kv_offsets[None, :]
        )
        k2_data = tl.load(
            k2 + num_token_offsets[:, None] * half_kv_size + half_kv_offsets[None, :]
        )
        v_data = tl.load(
            v
            + num_token_offsets[:, None, None] * kv_size
            + num_kv_offsets[None, :, None] * head_dim
            + head_offsets[None, None, :]
        )
        cos_data = tl.load(
            cos
            + num_token_offsets[:, None] * head_dim // 2
            + half_head_offsets[None, :]
        )
        sin_data = tl.load(
            sin
            + num_token_offsets[:, None] * head_dim // 2
            + half_head_offsets[None, :]
        )
        weight_data = tl.load(weight + half_head_offsets)

    with dl.async_task(scope=dl.async_task.vector):
        q1_data = tl.view(q1_data, num_tokens, q_size // head_dim, head_dim // 2)
        q2_data = tl.view(q2_data, num_tokens, q_size // head_dim, head_dim // 2)
        var_q1 = tl.sum(q1_data * q1_data, -1) / head_dim
        var_q2 = tl.sum(q2_data * q2_data, -1) / head_dim
        var_q = var_q1 + var_q2
        var_q = tl.expand_dims(var_q, -1)
        q1_data = q1_data * tl.math.rsqrt(var_q + 1e-5)
        q2_data = q2_data * tl.math.rsqrt(var_q + 1e-5)
        q1_data = weight_data * q1_data
        q2_data = weight_data * q2_data
        k1_data = tl.view(k1_data, num_tokens, kv_size // head_dim, head_dim // 2)
        k2_data = tl.view(k2_data, num_tokens, kv_size // head_dim, head_dim // 2)
        var_k1 = tl.sum(k1_data * k1_data, axis=-1) / head_dim
        var_k2 = tl.sum(k2_data * k2_data, axis=-1) / head_dim
        var_k = var_k1 + var_k2
        var_k = tl.expand_dims(var_k, -1)
        k1_data = k1_data * tl.math.rsqrt(var_k + 1e-5)
        k2_data = k2_data * tl.math.rsqrt(var_k + 1e-5)
        k1_data = weight_data * k1_data
        k2_data = weight_data * k2_data
        q1, q2 = _compute_rotary_emb(q1_data, q2_data, cos_data, sin_data)
        k1, k2 = _compute_rotary_emb(k1_data, k2_data, cos_data, sin_data)
        tl.store(
            q1_out
            + num_token_offsets[:, None, None] * half_q_size
            + num_q_offsets[None, :, None] * half_dim
            + half_head_offsets[None, None, :],
            q1,
        )
        tl.store(
            q2_out
            + num_token_offsets[:, None, None] * half_q_size
            + num_q_offsets[None, :, None] * half_dim
            + half_head_offsets[None, None, :],
            q2,
        )
        tl.store(
            k1_out
            + num_token_offsets[:, None, None] * half_kv_size
            + num_kv_offsets[None, :, None] * half_dim
            + half_head_offsets[None, None, :],
            k1,
        )
        tl.store(
            k2_out
            + num_token_offsets[:, None, None] * half_kv_size
            + num_kv_offsets[None, :, None] * half_dim
            + half_head_offsets[None, None, :],
            k2,
        )
        tl.store(
            v_out
            + num_token_offsets[:, None, None] * kv_size
            + num_kv_offsets[None, :, None] * head_dim
            + head_offsets[None, None, :],
            v_data,
        )


def rms_norm_rope(
    qkv,
    positions,
    num_heads_q,
    num_heads_kv,
    head_dim,
    num_tokens,
):
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q1, q2 = q.view(num_tokens, num_heads_q, head_dim).chunk(2, dim=-1)
    k1, k2 = k.view(num_tokens, num_heads_kv, head_dim).chunk(2, dim=-1)
    weight = torch.ones(head_dim // 2, dtype=torch.float32, device="npu")
    cache = _compute_cos_sin_cache(head_dim)

    positions = positions.flatten()
    cos_sin = cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    q1_out = torch.empty(
        (num_tokens, num_heads_q, head_dim // 2), dtype=torch.float32, device="npu"
    )
    q2_out = torch.empty(
        (num_tokens, num_heads_q, head_dim // 2), dtype=torch.float32, device="npu"
    )
    k1_out = torch.empty(
        (num_tokens, num_heads_kv, head_dim // 2), dtype=torch.float32, device="npu"
    )
    k2_out = torch.empty(
        (num_tokens, num_heads_kv, head_dim // 2), dtype=torch.float32, device="npu"
    )
    v_out = torch.empty(
        (num_tokens, num_heads_kv, head_dim), dtype=torch.float32, device="npu"
    )
    grid = (1,)

    rms_norm_rope_kernel[grid](
        q1.contiguous(),
        q2.contiguous(),
        k1.contiguous(),
        k2.contiguous(),
        v.contiguous(),
        weight.contiguous(),
        cos.contiguous(),
        sin.contiguous(),
        q1_out,
        q2_out,
        k1_out,
        k2_out,
        v_out,
        q_size,
        kv_size,
        head_dim,
        num_tokens,
    )
    q_out = torch.cat([q1_out, q2_out], dim=-1)
    k_out = torch.cat([k1_out, k2_out], dim=-1)
    q_out = q_out.view(num_tokens, q_size)
    k_out = k_out.view(num_tokens, kv_size)
    v_out = v_out.view(num_tokens, kv_size)
    return torch.cat([q_out, k_out, v_out], dim=-1)


# TODO! reference rms_norm implementation in torch
# def torch_rms_norm(hidden_states, weight, variance_epsilon=1e-6):
#     """pytorch forward."""
#     input_dtype = hidden_states.dtype
#     hidden_states = hidden_states.to(torch.float32)
#     variance = hidden_states.pow(2).mean(-1, keepdim=True)
#     hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
#     return weight * hidden_states.to(input_dtype)


def _apply_qk_norm_rope(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
) -> torch.Tensor:
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q_weight = torch.ones(head_dim, dtype=torch.float32, device="npu")

    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
    q_by_head = _rms_norm_kernel(q_by_head, q_weight, 1e-5)
    q = q_by_head.view(q.shape)

    k_weight = torch.ones(head_dim, dtype=torch.float32, device="npu")
    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_dim, head_dim)
    k_by_head = _rms_norm_kernel(k_by_head, k_weight, 1e-5)
    k = k_by_head.view(k.shape)

    cache = _compute_cos_sin_cache(head_dim)
    q, k = _rotary_embedding(positions, q, k, head_dim, cache, True)
    return torch.cat([q, k, v], dim=-1)


def test_rms_norm_rope():
    """test rms norm rope."""
    num_heads, num_kv_heads, head_dim = 4, 4, 16
    num_tokens = 4

    total_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv_base = torch.randn(num_tokens, total_dim, dtype=torch.float32, device="npu")
    qkv_base1 = qkv_base.clone()
    positions = torch.arange(num_tokens, dtype=torch.long, device="npu")
    positions1 = positions.clone()

    torch_output = _apply_qk_norm_rope(
        qkv=qkv_base,
        positions=positions,
        num_heads_q=num_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
    )

    triton_output = rms_norm_rope(
        qkv=qkv_base1,
        positions=positions1,
        num_heads_q=num_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
        num_tokens=num_tokens,
    )
    assert torch.allclose(torch_output, triton_output, atol=1e-2, rtol=0)
    print("test rms_norm_rope passed!")


if __name__ == "__main__":
    test_rms_norm_rope()
