import torch
import triton
import triton.language as tl
import triton.language.extra.deeplink as dl
import pytest


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


@triton.jit
def _compute_rotary_emb(
    x1,
    x2,
    cos,
    sin,
):
    cos = tl.expand_dims(cos, -2)
    sin = tl.expand_dims(sin, -2)
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
    BLOCK_SIZE: tl.constexpr,
    SUB_BLK: tl.constexpr,
):
    id = tl.program_id(0)
    num_q_offsets = tl.arange(0, q_size // head_dim)
    num_kv_offsets = tl.arange(0, kv_size // head_dim)
    head_offsets = tl.arange(0, head_dim)
    half_head_offsets = tl.arange(0, head_dim // 2)
    num_token_offsets = tl.arange(0, BLOCK_SIZE) + id * BLOCK_SIZE
    half_q_size = q_size // 2
    half_kv_size = kv_size // 2
    half_dim = head_dim // 2

    q1_data = tl.load(
        q1
        + num_token_offsets[:, None, None] * half_q_size
        + num_q_offsets[None, :, None] * half_dim
        + half_head_offsets[None, None, :],
    )
    q2_data = tl.load(
        q2
        + num_token_offsets[:, None, None] * half_q_size
        + num_q_offsets[None, :, None] * half_dim
        + half_head_offsets[None, None, :],
    )
    k1_data = tl.load(
        k1
        + num_token_offsets[:, None, None] * half_kv_size
        + num_kv_offsets[None, :, None] * half_dim
        + half_head_offsets[None, None, :],
    )
    k2_data = tl.load(
        k2
        + num_token_offsets[:, None, None] * half_kv_size
        + num_kv_offsets[None, :, None] * half_dim
        + half_head_offsets[None, None, :],
    )
    v_data = tl.load(
        v
        + num_token_offsets[:, None, None] * kv_size
        + num_kv_offsets[None, :, None] * head_dim
        + head_offsets[None, None, :]
    )
    cos_data = tl.load(
        cos + num_token_offsets[:, None] * head_dim // 2 + half_head_offsets[None, :]
    )
    sin_data = tl.load(
        sin + num_token_offsets[:, None] * head_dim // 2 + half_head_offsets[None, :]
    )
    weight_data = tl.load(weight + half_head_offsets)

    for s in dl.parallel(0, 2, bind_sub_block=True):
        q1_sub_data = dl.extract_slice(
            q1_data,
            (s * SUB_BLK, 0, 0),
            (SUB_BLK, q_size // head_dim, head_dim // 2),
            (1, 1, 1),
        )
        q2_sub_data = dl.extract_slice(
            q2_data,
            (s * SUB_BLK, 0, 0),
            (SUB_BLK, q_size // head_dim, head_dim // 2),
            (1, 1, 1),
        )
        k1_sub_data = dl.extract_slice(
            k1_data,
            (s * SUB_BLK, 0, 0),
            (SUB_BLK, kv_size // head_dim, head_dim // 2),
            (1, 1, 1),
        )
        k2_sub_data = dl.extract_slice(
            k2_data,
            (s * SUB_BLK, 0, 0),
            (SUB_BLK, kv_size // head_dim, head_dim // 2),
            (1, 1, 1),
        )
        cos_sub_data = dl.extract_slice(
            cos_data, (s * SUB_BLK, 0), (SUB_BLK, head_dim // 2), (1, 1)
        )
        sin_sub_data = dl.extract_slice(
            sin_data, (s * SUB_BLK, 0), (SUB_BLK, head_dim // 2), (1, 1)
        )

        # rms norm
        var_q1 = tl.sum(q1_sub_data * q1_sub_data, -1) / head_dim
        var_q2 = tl.sum(q2_sub_data * q2_sub_data, -1) / head_dim
        var_q = var_q1 + var_q2
        var_q = tl.expand_dims(var_q, -1)
        q1_sub_data = q1_sub_data * tl.math.rsqrt(var_q + 1e-5)
        q2_sub_data = q2_sub_data * tl.math.rsqrt(var_q + 1e-5)
        q1_sub_data = weight_data * q1_sub_data
        q2_sub_data = weight_data * q2_sub_data
        var_k1 = tl.sum(k1_sub_data * k1_sub_data, axis=-1) / head_dim
        var_k2 = tl.sum(k2_sub_data * k2_sub_data, axis=-1) / head_dim
        var_k = var_k1 + var_k2
        var_k = tl.expand_dims(var_k, -1)
        k1_sub_data = k1_sub_data * tl.math.rsqrt(var_k + 1e-5)
        k2_sub_data = k2_sub_data * tl.math.rsqrt(var_k + 1e-5)
        k1_sub_data = weight_data * k1_sub_data
        k2_sub_data = weight_data * k2_sub_data

        # rotary embedding
        q1_rope, q2_rope = _compute_rotary_emb(
            q1_sub_data, q2_sub_data, cos_sub_data, sin_sub_data
        )
        k1_rope, k2_rope = _compute_rotary_emb(
            k1_sub_data, k2_sub_data, cos_sub_data, sin_sub_data
        )
        num_token_sub_offsets = tl.arange(0, SUB_BLK) + id * BLOCK_SIZE + s * SUB_BLK

        tl.store(
            q1_out
            + num_token_sub_offsets[:, None, None] * half_q_size
            + num_q_offsets[None, :, None] * half_dim
            + half_head_offsets[None, None, :],
            q1_rope,
        )
        tl.store(
            q2_out
            + num_token_sub_offsets[:, None, None] * half_q_size
            + num_q_offsets[None, :, None] * half_dim
            + half_head_offsets[None, None, :],
            q2_rope,
        )
        tl.store(
            k1_out
            + num_token_sub_offsets[:, None, None] * half_kv_size
            + num_kv_offsets[None, :, None] * half_dim
            + half_head_offsets[None, None, :],
            k1_rope,
        )
        tl.store(
            k2_out
            + num_token_sub_offsets[:, None, None] * half_kv_size
            + num_kv_offsets[None, :, None] * half_dim
            + half_head_offsets[None, None, :],
            k2_rope,
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
    BLOCK_SIZE,
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
    grid = (num_tokens // BLOCK_SIZE,)

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
        BLOCK_SIZE,
        BLOCK_SIZE // 2,
    )
    q_out = torch.cat([q1_out, q2_out], dim=-1)
    k_out = torch.cat([k1_out, k2_out], dim=-1)
    q_out = q_out.view(num_tokens, q_size)
    k_out = k_out.view(num_tokens, kv_size)
    v_out = v_out.view(num_tokens, kv_size)
    return torch.cat([q_out, k_out, v_out], dim=-1)


# ------------------ PyTorch 参考实现（用于验证精度） ------------------
def _apply_qk_norm_rope(qkv, positions, num_heads_q, num_heads_kv, head_dim):
    """Pure PyTorch implementation mirroring rms_norm_rope for correctness checks.

    Args:
        qkv: (num_tokens, total_dim)
        positions: (num_tokens,)
        num_heads_q, num_heads_kv, head_dim: ints

    Returns:
        concatenated tensor of shape (num_tokens, q_size + kv_size + kv_size)
    """
    device = qkv.device
    num_tokens = qkv.shape[0]
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    # reshape
    q = q.view(num_tokens, num_heads_q, head_dim)
    k = k.view(num_tokens, num_heads_kv, head_dim)

    # split halves
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)

    # RMS norm scale (same as kernel: sum of squares over full head_dim / head_dim)
    var_q = (q1.pow(2).sum(-1) + q2.pow(2).sum(-1)) / head_dim
    scale_q = torch.rsqrt(var_q + 1e-5).unsqueeze(-1)
    q1 = q1 * scale_q
    q2 = q2 * scale_q

    var_k = (k1.pow(2).sum(-1) + k2.pow(2).sum(-1)) / head_dim
    scale_k = torch.rsqrt(var_k + 1e-5).unsqueeze(-1)
    k1 = k1 * scale_k
    k2 = k2 * scale_k

    # weight (the kernel uses a per-dim weight of ones)
    weight = torch.ones(head_dim // 2, dtype=torch.float32, device=device)
    q1 = q1 * weight
    q2 = q2 * weight
    k1 = k1 * weight
    k2 = k2 * weight

    # rotary
    cache = _compute_cos_sin_cache(head_dim)
    cos_sin = cache.index_select(0, positions.flatten())
    cos, sin = cos_sin.chunk(2, dim=-1)
    # reshape cos/sin to broadcast: (num_tokens, 1, head_dim//2)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q1_rope = q1 * cos - q2 * sin
    q2_rope = q2 * cos + q1 * sin
    k1_rope = k1 * cos - k2 * sin
    k2_rope = k2 * cos + k1 * sin

    # reconstruct
    q_out = torch.cat([q1_rope, q2_rope], dim=-1).view(num_tokens, q_size)
    k_out = torch.cat([k1_rope, k2_rope], dim=-1).view(num_tokens, kv_size)
    v_out = v  # v is unchanged

    return torch.cat([q_out, k_out, v_out], dim=-1)


# ------------------ 测试与基准 ------------------


def test_rms_norm_rope():
    """test rms norm rope and benchmark compared to the Triton kernel."""
    num_heads, num_kv_heads, head_dim = 16, 4, 128
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
        BLOCK_SIZE=2,
    )
    assert torch.allclose(torch_output, triton_output, atol=1e-2, rtol=0)
    print("test rms_norm_rope passed!")

    # 性能测试部分
    try:
        import triton.testing as tt

        def benchmark_fn(fn, *args):
            return tt.do_bench(lambda: fn(*args), warmup=10, rep=20)

        # Triton 版本性能
        tri_time = benchmark_fn(
            rms_norm_rope,
            qkv_base1,
            positions1,
            num_heads,
            num_kv_heads,
            head_dim,
            num_tokens,
            2,
        )

        # PyTorch 版本性能
        torch_time = benchmark_fn(
            _apply_qk_norm_rope, qkv_base, positions, num_heads, num_kv_heads, head_dim
        )

        # 打印性能对比结果
        print(f"\n=== 性能对比 ===")
        print(
            f"Triton: {tri_time:.4f} ms | PyTorch: {torch_time:.4f} ms | 加速比: {torch_time/tri_time:.2f}x"
        )
    except Exception:
        print("triton.testing.do_bench unavailable: 跳过基准测试。")


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim,num_tokens,BLOCK_SIZE",
    [
        (16, 4, 128, 4, 2),
        (8, 2, 64, 8, 4),
        (32, 16, 64, 256, 4),
        (48, 16, 64, 192, 4),
    ],
)
def test_rms_norm_rope_correctness(
    num_heads,
    num_kv_heads,
    head_dim,
    num_tokens,
    BLOCK_SIZE,
):
    total_dim = (num_heads + 2 * num_kv_heads) * head_dim

    qkv = torch.randn(num_tokens, total_dim, dtype=torch.float32, device="npu")
    positions = torch.arange(num_tokens, dtype=torch.long, device="npu")

    torch_out = _apply_qk_norm_rope(
        qkv=qkv,
        positions=positions,
        num_heads_q=num_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
    )

    triton_out = rms_norm_rope(
        qkv=qkv.clone(),
        positions=positions.clone(),
        num_heads_q=num_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
        num_tokens=num_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    assert torch.allclose(torch_out, triton_out, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_rms_norm_rope()
