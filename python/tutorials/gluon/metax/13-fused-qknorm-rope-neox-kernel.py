"""Performance and accuracy results for fused Q/K RMSNorm + NeoX RoPE.

Measured on 2026-08-12: MetaX C500, PyTorch 2.8.0+metax3.7.2.0, BF16,
20 warmups and 100 CUDA-event samples. Each sample restores QKV before the
start event, so the in-place input copy is not timed. The values below are p50
milliseconds. ``T`` is an integer total scheduled-token count, with values
``(1, 1024, 2048, 4096, 8192, 10240, 20480, 40960)``.

C kernel p50 (ms):

| model | packed QKV [T, W] | T=1 | T=1024 | T=2048 | T=4096 | T=8192 | T=10240 | T=20480 | T=40960 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | [T, 4096] | 0.0141 | 0.0884 | 0.1528 | 0.2852 | 0.5501 | 0.6826 | 1.3884 | 2.7028 |
| Qwen3-8B | [T, 6144] | 0.0146 | 0.1505 | 0.2808 | 0.5435 | 1.0650 | 1.3284 | 2.6769 | 5.2700 |
| Qwen3-30B-A3B-Instruct-2507 | [T, 5120] | 0.0146 | 0.1485 | 0.2783 | 0.5389 | 1.0566 | 1.3158 | 2.6619 | 5.2334 |
| Qwen3-32B | [T, 10240] | 0.0154 | 0.2783 | 0.5396 | 1.0565 | 2.1297 | 2.6606 | 5.2342 | 10.3940 |
| Qwen3-235B-A22B | [T, 9216] | 0.0151 | 0.2764 | 0.5356 | 1.0524 | 2.1056 | 2.6260 | 5.2216 | 10.3648 |
| Qwen3-VL-8B-Instruct text tower | [T, 6144] | 0.0143 | 0.1505 | 0.2813 | 0.5435 | 1.0652 | 1.3285 | 2.6771 | 5.2682 |

Gluon kernel p50 (ms):

| model | packed QKV [T, W] | T=1 | T=1024 | T=2048 | T=4096 | T=8192 | T=10240 | T=20480 | T=40960 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | [T, 4096] | 0.0461 | 0.0384 | 0.0607 | 0.1019 | 0.1846 | 0.2258 | 0.4819 | 0.8860 |
| Qwen3-8B | [T, 6144] | 0.0451 | 0.0591 | 0.0964 | 0.1774 | 0.3333 | 0.4116 | 0.8517 | 1.6091 |
| Qwen3-30B-A3B-Instruct-2507 | [T, 5120] | 0.0328 | 0.0591 | 0.0973 | 0.1738 | 0.3269 | 0.4027 | 0.8306 | 1.5704 |
| Qwen3-32B | [T, 10240] | 0.0328 | 0.0952 | 0.1741 | 0.3261 | 0.6807 | 0.8334 | 1.5786 | 3.1046 |
| Qwen3-235B-A22B | [T, 9216] | 0.0317 | 0.0945 | 0.1718 | 0.3233 | 0.6449 | 0.8005 | 1.5593 | 3.0546 |
| Qwen3-VL-8B-Instruct text tower | [T, 6144] | 0.0324 | 0.0604 | 0.0996 | 0.1779 | 0.3346 | 0.4133 | 0.8471 | 1.6072 |

For T>=1024, the C/Gluon speedup is 2.30x to 3.39x.  T=1 remains launch-bound.

Before timing, all six model geometries were checked against the benchmark's
FP32 RMSNorm + NeoX RoPE reference with their checkpoint Q/K norm weights.
The largest observed absolute error was 0.00390625, below ``atol=0.03125``;
V was bitwise unchanged for every case.
"""

import pytest
import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


HEAD_DIM = 128
C_LOGICAL_WARPS_PER_BLOCK = 16
C500_WARPS_PER_BLOCK = 4
# RoPE theta is the frequency base used to construct the rotary cos/sin table.
# The suffix names the magnitude (1M = one million), not a token count.
ROPE_THETA_1M = 1_000_000.0
ROPE_THETA_5M = 5_000_000.0
ROPE_THETA_10M = 10_000_000.0
MODEL_CASES = (
    ("Qwen3-0.6B", 16, 8, 1e-6, ROPE_THETA_1M),
    ("Qwen3-8B", 32, 8, 1e-6, ROPE_THETA_1M),
    ("Qwen3-30B-A3B-Instruct-2507", 32, 4, 1e-6, ROPE_THETA_10M),
    ("Qwen3-32B", 64, 8, 1e-6, ROPE_THETA_1M),
    ("Qwen3-235B-A22B", 64, 4, 1e-6, ROPE_THETA_1M),
    ("Qwen3-VL-8B-Instruct text tower", 32, 8, 1e-6, ROPE_THETA_5M),
)


def make_cos_sin_cache(max_position: int, head_dim: int, theta: float) -> torch.Tensor:
    inv_freq = 1.0 / theta ** (
        torch.arange(0, head_dim, 2, device="cuda", dtype=torch.float32) / head_dim
    )
    positions = torch.arange(max_position, device="cuda", dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)


def is_maca():
    try:
        target = triton.runtime.driver.active.get_current_target()
    except RuntimeError:
        return False
    return target.backend == "maca"


@gluon.jit
def fused_qknorm_rope_neox_kernel(
    qkv_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    position_ids_ptr,
    num_tokens,
    qkv_stride_token,
    eps,
    Q_HEADS: gl.constexpr,
    KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
):
    # One program contains sixteen Q/K-head rows.
    program = gl.program_id(axis=0)
    logical_warps_per_block: gl.constexpr = 16
    c500_warps_per_block: gl.constexpr = 4
    qk_heads = Q_HEADS + KV_HEADS

    # Four C500 warps partition the head axis.  The D axis maps eight adjacent
    # values to each thread: 1 * 8 * 4 * 16 * 4 * 1 = [16 heads, 128 D values].
    head_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, HEAD_DIM // 16],
        threads_per_warp=[4, 16],
        warps_per_cta=[c500_warps_per_block, 1],
        order=[1, 0],
    )
    head_in_block = gl.arange(
        0,
        logical_warps_per_block,
        layout=gl.SliceLayout(1, head_layout),
    )
    half_dim: gl.constexpr = HEAD_DIM // 2
    dim = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, head_layout))
    paired_dim = (dim + half_dim) % HEAD_DIM
    rope_dim = dim % half_dim

    global_head = program * logical_warps_per_block + head_in_block
    token = global_head // qk_heads
    qk_head = global_head % qk_heads
    valid_token = token < num_tokens

    # Q and K are the first Q_HEADS + KV_HEADS head tiles in each packed row.
    # Do not compute an address for V.
    qk_base = token * qkv_stride_token + qk_head * HEAD_DIM
    qk_ptrs = qkv_ptr + qk_base[:, None] + dim[None, :]
    valid = valid_token[:, None]
    x = gl.load(qk_ptrs, mask=valid, other=0.0).to(gl.float32)
    sum_of_squares = gl.sum(x * x, axis=1)
    rms_rcp = gl.rsqrt(sum_of_squares / HEAD_DIM + eps)[:, None]

    is_query = qk_head[:, None] < Q_HEADS
    paired_x = gl.load(qkv_ptr + qk_base[:, None] + paired_dim[None, :], mask=valid, other=0.0).to(gl.float32)
    # C selects the Q/K weight base before loading.  The condition is uniform
    # within each head row, so only that row's two weight vectors are loaded.
    weight_ptr = gl.where(is_query, q_weight_ptr, k_weight_ptr)
    scale = gl.load(weight_ptr + dim[None, :]).to(gl.float32)
    paired_scale = gl.load(weight_ptr + paired_dim[None, :]).to(gl.float32)
    x = x * rms_rcp * scale
    paired_x = paired_x * rms_rcp * paired_scale

    position = gl.load(position_ids_ptr + token, mask=valid_token, other=0)
    cache_base = position[:, None] * HEAD_DIM
    cos = gl.load(
        cos_sin_cache_ptr + cache_base + rope_dim[None, :],
        mask=valid,
        other=0.0,
    ).to(gl.float32)
    sin = gl.load(
        cos_sin_cache_ptr + cache_base + half_dim + rope_dim[None, :],
        mask=valid,
        other=0.0,
    ).to(gl.float32)
    is_first_half = dim[None, :] < half_dim
    rope = gl.where(is_first_half, x * cos - paired_x * sin, x * cos + paired_x * sin)
    gl.store(qk_ptrs, rope.to(qkv_ptr.dtype.element_ty), mask=valid)


def fused_qknorm_rope_neox_packed(
    qkv, q_heads, kv_heads, q_weight, k_weight, cos_sin_cache, position_ids, eps
):
    """Run the C-operator equivalent on packed qkv[T, (Q + K + V) * 128]."""
    assert qkv.is_cuda and qkv.is_contiguous() and qkv.dtype == torch.bfloat16
    assert qkv.ndim == 2
    assert qkv.shape[1] == (q_heads + 2 * kv_heads) * HEAD_DIM
    assert q_weight.shape == (HEAD_DIM,) and q_weight.dtype == qkv.dtype
    assert k_weight.shape == (HEAD_DIM,) and k_weight.dtype == qkv.dtype
    assert cos_sin_cache.is_cuda and cos_sin_cache.is_contiguous()
    assert cos_sin_cache.shape[1] == HEAD_DIM
    assert position_ids.shape == (qkv.shape[0],) and position_ids.dtype == torch.int64

    # One program handles sixteen Q/K-head rows; the final program masks
    # unused rows.
    qk_programs = qkv.shape[0] * (q_heads + kv_heads)
    grid = (triton.cdiv(qk_programs, C_LOGICAL_WARPS_PER_BLOCK),)
    fused_qknorm_rope_neox_kernel[grid](
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        position_ids,
        qkv.shape[0],
        qkv.stride(0),
        eps,
        Q_HEADS=q_heads,
        KV_HEADS=kv_heads,
        HEAD_DIM=HEAD_DIM,
        num_warps=C500_WARPS_PER_BLOCK,
        num_ctas=1,
    )


def torch_fused_qknorm_rope_neox(
    qkv, q_heads, kv_heads, q_weight, k_weight, cos_sin_cache, position_ids, eps
):
    """PyTorch FP32 reference for accuracy checks; leaves packed V unchanged."""
    assert qkv.ndim == 2 and qkv.dtype == torch.bfloat16
    assert qkv.shape[1] == (q_heads + 2 * kv_heads) * HEAD_DIM
    assert q_weight.shape == (HEAD_DIM,) and k_weight.shape == (HEAD_DIM,)
    assert cos_sin_cache.shape[1] == HEAD_DIM
    assert position_ids.shape == (qkv.shape[0],)

    qk_heads = q_heads + kv_heads
    qk_width = qk_heads * HEAD_DIM
    # The Gluon kernel processes this packed Q/K region in place.  Keep the
    # reference operations in FP32 until the final BF16 conversion.
    qk = qkv[:, :qk_width].view(-1, qk_heads, HEAD_DIM).float()
    weights = torch.cat((
        q_weight.float().expand(q_heads, -1),
        k_weight.float().expand(kv_heads, -1),
    ))
    rms_rcp = torch.rsqrt((qk * qk).mean(dim=-1, keepdim=True) + eps)
    qk = qk * rms_rcp * weights.unsqueeze(0)
    cos_sin = cos_sin_cache.index_select(0, position_ids).float()
    cos, sin = cos_sin.chunk(2, dim=-1)
    first, second = qk.chunk(2, dim=-1)
    qk = torch.cat((
        first * cos[:, None] - second * sin[:, None],
        second * cos[:, None] + first * sin[:, None],
    ), dim=-1).to(qkv.dtype)
    return torch.cat((qk.flatten(1), qkv[:, qk_width:]), dim=1)


@pytest.mark.skipif(not is_maca(), reason="Requires MetaX/MACA target")
@pytest.mark.parametrize(("name", "q_heads", "kv_heads", "eps", "theta"), MODEL_CASES)
def test_maca_fused_qknorm_rope_neox_models(name, q_heads, kv_heads, eps, theta):
    torch.manual_seed(7)
    tokens, cache_rows = 37, 128
    qkv = torch.randn(
        (tokens, (q_heads + 2 * kv_heads) * HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )
    original = qkv.clone()
    q_weight = torch.randn((HEAD_DIM,), device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn((HEAD_DIM,), device="cuda", dtype=torch.bfloat16)
    cos_sin_cache = make_cos_sin_cache(cache_rows, HEAD_DIM, theta)
    position_ids = torch.randint(
        cache_rows, (tokens,), device="cuda", dtype=torch.int64
    )
    expected = torch_fused_qknorm_rope_neox(
        original, q_heads, kv_heads, q_weight, k_weight, cos_sin_cache, position_ids, eps
    )
    fused_qknorm_rope_neox_packed(
        qkv, q_heads, kv_heads, q_weight, k_weight, cos_sin_cache, position_ids, eps
    )
    torch.cuda.synchronize()

    qk_width = (q_heads + kv_heads) * HEAD_DIM
    assert torch.equal(qkv[:, qk_width:], original[:, qk_width:]), f"{name}: V was modified"
    torch.testing.assert_close(qkv[:, :qk_width], expected[:, :qk_width], atol=3.125e-2, rtol=1e-2)
