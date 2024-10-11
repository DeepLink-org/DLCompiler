import torch
import triton
import triton.language as tl
from torch import Tensor
from dlblas.utils import register_dlblas_op, SymVar, Tensor, ChoiceSpace
from dlblas.utils.libentry import libentry


@triton.jit
def _fill_kv_cache_kernel(
    key,
    value,
    key_cache,
    value_cache,
    kv_indices,
    stride_k_n,
    stride_k_h,
    stride_k_d,
    stride_v_n,
    stride_v_h,
    stride_v_d,
    stride_kcache_b,
    stride_kcache_h,
    stride_kcache_d,
    stride_vcache_b,
    stride_vcache_h,
    stride_vcache_d,
    num_tokens,
    num_heads,
    head_dim_k,
    head_dim_v,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_DIM_K: tl.constexpr,
    BLOCK_DIM_V: tl.constexpr,
):
    block_id = tl.program_id(0)
    offs_block = tl.arange(0, BLOCK_SIZE)
    offs_heads = tl.arange(0, BLOCK_HEADS)[None, :, None]
    offs_dim_k = tl.arange(0, BLOCK_DIM_K)[None, None, :]
    offs_dim_v = tl.arange(0, BLOCK_DIM_V)[None, None, :]
    kv_indices_ptrs = kv_indices + block_id * BLOCK_SIZE + offs_block
    kv_indices_data = tl.load(
        kv_indices_ptrs, mask=(block_id * BLOCK_SIZE + offs_block < num_tokens)
    )
    offs_k = (
        (block_id * BLOCK_SIZE + offs_block)[:, None, None] * stride_k_n
        + offs_heads * stride_k_h
        + offs_dim_k * stride_k_d
    )

    masks_k = (
        (block_id * BLOCK_SIZE + offs_block[:, None, None] < num_tokens)
        & (offs_heads < num_heads)
        & (offs_dim_k < head_dim_k)
    )
    key_data = tl.load(key + offs_k, mask=masks_k)
    offs_kcache = (
        kv_indices_data[:, None, None] * stride_kcache_b
        + offs_heads * stride_kcache_h
        + offs_dim_k * stride_kcache_d
    )
    tl.store(key_cache + offs_kcache, key_data, mask=masks_k)
    offs_v = (
        (block_id * BLOCK_SIZE + offs_block)[:, None, None] * stride_v_n
        + offs_heads * stride_v_h
        + offs_dim_v * stride_v_d
    )
    masks_v = (
        (block_id * BLOCK_SIZE + offs_block[:, None, None] < num_tokens)
        & (offs_heads < num_heads)
        & (offs_dim_v < head_dim_v)
    )
    value_data = tl.load(value + offs_v, mask=masks_v)
    offs_vcache = (
        kv_indices_data[:, None, None] * stride_vcache_b
        + offs_heads * stride_vcache_h
        + offs_dim_v * stride_vcache_d
    )
    tl.store(value_cache + offs_vcache, value_data, mask=masks_v)


def call(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_indices: torch.Tensor,
):
    num_blocks, block_size, num_heads, head_dim_k = key_cache.shape
    _, _, _, head_dim_v = value_cache.shape
    num_tokens, _, _ = key.shape
    assert key.shape[:2] == value.shape[:2]
    assert key_cache.shape[:3] == value_cache.shape[:3]
    assert block_size == triton.next_power_of_2(block_size)
    _, stride_kcache_b, stride_kcache_h, stride_kcache_d = key_cache.stride()
    _, stride_vcache_b, stride_vcache_h, stride_vcache_d = value_cache.stride()
    stride_k_n, stride_k_h, stride_k_d = key.stride()
    stride_v_n, stride_v_h, stride_v_d = value.stride()
    total_blocks = (num_tokens + block_size - 1) // block_size
    _fill_kv_cache_kernel[(total_blocks,)](
        key,
        value,
        key_cache,
        value_cache,
        kv_indices,
        stride_k_n,
        stride_k_h,
        stride_k_d,
        stride_v_n,
        stride_v_h,
        stride_v_d,
        stride_kcache_b,
        stride_kcache_h,
        stride_kcache_d,
        stride_vcache_b,
        stride_vcache_h,
        stride_vcache_d,
        num_tokens,
        num_heads,
        head_dim_k,
        head_dim_v,
        block_size,
        triton.next_power_of_2(num_heads),
        triton.next_power_of_2(head_dim_k),
        triton.next_power_of_2(head_dim_v),
    )
    return key_cache, value_cache


def bench_fn(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_indices: torch.Tensor,
):
    fn = lambda: call(key, value, key_cache, value_cache, kv_indices)
    ms = triton.testing.do_bench(fn, warmup=10, rep=10)
    return ms


# register
name = "fill_kv_cache"
for dtype in [torch.float16, torch.float32]:
    for device in ["cuda"]:
        num_blocks, block_size = SymVar("num_blocks"), SymVar("block_size")
        head_dim_k, head_dim_v = SymVar("head_dim_k"), SymVar("head_dim_v")
        num_heads, num_tokens = SymVar("num_heads"), SymVar("num_tokens")

        # we dont' actually allocate tensor
        key = Tensor((num_tokens, num_heads, head_dim_k), dtype=dtype, device=device)
        value = Tensor((num_tokens, num_heads, head_dim_v), dtype=dtype, device=device)
        key_cache = Tensor(
            (num_blocks, block_size, num_heads, head_dim_k), dtype=dtype, device=device
        )
        value_cache = Tensor(
            (num_blocks, block_size, num_heads, head_dim_v), dtype=dtype, device=device
        )
        kv_indices = Tensor((num_tokens,), dtype=torch.int32, device=device)
        # space = ChoiceSpace([])
        register_dlblas_op(
            name,
            None,
            (key, value, key_cache, value_cache, kv_indices),
            call,
            bench_fn,
            call,
        )
