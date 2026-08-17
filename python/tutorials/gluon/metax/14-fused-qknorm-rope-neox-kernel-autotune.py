"""Layout-autotuned fused Q/K RMSNorm + NeoX RoPE for MetaX C500.

This clone preserves the original in-place algorithm and launch geometry but
leaves its register ownership to the native post-TTGIR layout autotuner. It
contains no source-level layout declaration or conversion.

Measured on 2026-08-12: MetaX C500, PyTorch 2.8.0+metax3.7.2.0, BF16,
20 warmups and 100 CUDA-event samples. Each sample restores QKV before the
start event, so the in-place copy is excluded. Values are kernel-only p50
milliseconds. ``T`` has the same meaning and type as in the original test: an
integer total scheduled-token count, with values
``(1, 1024, 2048, 4096, 8192, 10240, 20480, 40960)``.

The tables below are historical measurements from the former per-``T`` winner
key. The current implementation selects once per 128x ``T`` dispatch range, so
these numbers do not measure its winner-reuse policy.

C kernel p50 (ms):

| model | packed QKV [T, W] | T=1 | T=1024 | T=2048 | T=4096 | T=8192 | T=10240 | T=20480 | T=40960 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | [T, 4096] | 0.0143 | 0.0888 | 0.1528 | 0.2851 | 0.5512 | 0.6824 | 1.3893 | 2.7026 |
| Qwen3-8B | [T, 6144] | 0.0146 | 0.1505 | 0.2812 | 0.5437 | 1.0652 | 1.3275 | 2.6772 | 5.2710 |
| Qwen3-30B-A3B-Instruct-2507 | [T, 5120] | 0.0148 | 0.1487 | 0.2781 | 0.5389 | 1.0575 | 1.3162 | 2.6604 | 5.2346 |
| Qwen3-32B | [T, 10240] | 0.0154 | 0.2783 | 0.5399 | 1.0568 | 2.1366 | 2.6607 | 5.2315 | 10.3935 |
| Qwen3-235B-A22B | [T, 9216] | 0.0154 | 0.2762 | 0.5353 | 1.0519 | 2.1064 | 2.6291 | 5.2227 | 10.3634 |
| Qwen3-VL-8B-Instruct text tower | [T, 6144] | 0.0146 | 0.1503 | 0.2813 | 0.5435 | 1.0650 | 1.3271 | 2.6757 | 5.2700 |

Gluon layout-autotune p50 (ms):

| model | packed QKV [T, W] | T=1 | T=1024 | T=2048 | T=4096 | T=8192 | T=10240 | T=20480 | T=40960 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | [T, 4096] | 0.0481 | 0.0343 | 0.0471 | 0.0748 | 0.1318 | 0.1603 | 0.3109 | 0.6394 |
| Qwen3-8B | [T, 6144] | 0.0509 | 0.0428 | 0.0773 | 0.1316 | 0.2176 | 0.2678 | 0.5202 | 1.0547 |
| Qwen3-30B-A3B-Instruct-2507 | [T, 5120] | 0.0479 | 0.0445 | 0.0701 | 0.1196 | 0.2243 | 0.2757 | 0.5400 | 1.0865 |
| Qwen3-32B | [T, 10240] | 0.0479 | 0.0668 | 0.1116 | 0.2086 | 0.4050 | 0.4978 | 1.0075 | 1.9300 |
| Qwen3-235B-A22B | [T, 9216] | 0.0476 | 0.0673 | 0.1139 | 0.2117 | 0.4357 | 0.5390 | 1.0363 | 1.9886 |
| Qwen3-VL-8B-Instruct text tower | [T, 6144] | 0.0476 | 0.0430 | 0.0684 | 0.1183 | 0.2176 | 0.2680 | 0.5199 | 1.0542 |

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


@gluon.jit(
    do_not_specialize=["num_tokens"],
    enable_gluon_layout_autotune=True,
)
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

    head_in_block = gl.arange(0, logical_warps_per_block)
    half_dim: gl.constexpr = HEAD_DIM // 2
    dim = gl.arange(0, HEAD_DIM)
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

    # Native layout autotuning benchmarks multiple candidates. Since the
    # operator updates Q/K in place, tune on disposable storage before running
    # the selected winner once on the tensor checked below.
    tune_qkv = original.clone()
    fused_qknorm_rope_neox_packed(
        tune_qkv,
        q_heads,
        kv_heads,
        q_weight,
        k_weight,
        cos_sin_cache,
        position_ids,
        eps,
    )
    fused_qknorm_rope_neox_packed(
        qkv, q_heads, kv_heads, q_weight, k_weight, cos_sin_cache, position_ids, eps 
    )
    torch.cuda.synchronize()

    qk_width = (q_heads + kv_heads) * HEAD_DIM
    assert torch.equal(qkv[:, qk_width:], original[:, qk_width:]), f"{name}: V was modified"
    torch.testing.assert_close(qkv[:, :qk_width], expected[:, :qk_width], atol=3.125e-2, rtol=1e-2)
