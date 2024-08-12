import torch
import pytest
import triton
import dlblas 


def torch_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> None:
    output = torch.empty_like(query)

    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    block_size = value_cache.shape[2]
    head_size = value_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables = block_tables.cpu().tolist()
    context_lens = context_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size
            k = key_cache[block_number, :, block_offset, :]
            keys.append(k)
            v = value_cache[block_number, :, block_offset, :]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        S = torch.bmm(q.transpose(0, 1).float(), keys.permute(1, 2, 0).float()) * scale
        P = torch.softmax(S, dim=-1)
        out = torch.bmm(P, values.transpose(0, 1).float()).transpose(0, 1)
        out = out.to(values.dtype)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)

    return output


NUM_BLOCKS = 1000


def base_paged_attention(
    num_seqs,
    num_query_heads,
    query_group_size,
    head_size,
    block_size,
    max_seq_len,
    dtype=torch.float16,
    device="cuda",
):
    torch.set_default_dtype(dtype)
    torch.set_default_device(device=device)

    num_kv_heads = num_query_heads // query_group_size

    context_lens = torch.randint(1, max_seq_len, [num_seqs], dtype=torch.int32)
    context_lens[0] = max_seq_len
    max_context_len = context_lens.max().item()
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size

    attn_scale = head_size**-0.5
    q = torch.empty(num_seqs, num_query_heads, head_size)
    q.uniform_(-attn_scale, attn_scale)

    k_cache = torch.empty(NUM_BLOCKS, num_kv_heads, block_size, head_size)
    k_cache.uniform_(-attn_scale, attn_scale)
    v_cache = torch.empty_like(k_cache)
    v_cache.uniform_(-attn_scale, attn_scale)

    # (NUM_SEQS, MAX_NUM_BLOCKS_PER_SEQ)
    block_tables = torch.randint(0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq))

    out = dlblas.paged_attention(
        q,
        k_cache,
        v_cache,
        context_lens,
        block_tables,
        attn_scale,
        max_context_len,
    )

    ref_out = torch_paged_attention(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        attn_scale,
    )
    print(torch.abs(out - ref_out).max())
    assert torch.allclose(out, ref_out, atol=2e-3, rtol=1e-5)


@pytest.mark.parametrize("num_seqs", [1, 32])
@pytest.mark.parametrize("num_query_heads", [64])
@pytest.mark.parametrize("query_group_size", [1, 8])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("block_size", [16, 128, 256])
@pytest.mark.parametrize("max_seq_len", [512, 4096])
def test_paged_attention_default(
    num_seqs,
    num_query_heads,
    query_group_size,
    head_size,
    block_size,
    max_seq_len,
    dtype=torch.float16,
    device="cuda",
):
    base_paged_attention(
        num_seqs,
        num_query_heads,
        query_group_size,
        head_size,
        block_size,
        max_seq_len,
    )

@pytest.mark.parametrize("num_seqs", [1, 16])
@pytest.mark.parametrize("num_query_heads", [64])
@pytest.mark.parametrize("query_group_size", [1, 8])
@pytest.mark.parametrize("head_size", [32, 64])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("max_seq_len", [2048])
def test_paged_attention_by_num_splits(
    num_seqs,
    num_query_heads,
    query_group_size,
    head_size,
    block_size,
    max_seq_len,
    dtype=torch.float16,
    device="cuda",
):
    base_paged_attention(
        num_seqs,
        num_query_heads,
        query_group_size,
        head_size,
        block_size,
        max_seq_len,
    )

@pytest.mark.parametrize("num_seqs, num_query_heads, query_group_size, head_size, block_size, max_seq_len", [
    (1, 12, 1, 64, 16, 2),
    (16, 64, 8, 32, 16, 2048),
    (16, 64, 1, 64, 16, 2048),
])
def test_paged_attention_by_case(
    num_seqs,
    num_query_heads,
    query_group_size,
    head_size,
    block_size,
    max_seq_len,
    dtype=torch.float16,
    device="cuda",
):
    base_paged_attention(
        num_seqs,
        num_query_heads,
        query_group_size,
        head_size,
        block_size,
        max_seq_len,
    )


def bench(
    num_seqs=32,
    num_query_heads=64,
    query_group_size = 8,
    head_size = 64,
    block_size = 256,
    max_seq_len = 81920,
    dtype=torch.float16,
    device="cuda",
):
    torch.set_default_dtype(dtype)
    torch.set_default_device(device=device)

    
    num_kv_heads = num_query_heads // query_group_size

    context_lens = torch.randint(1, max_seq_len, [num_seqs], dtype=torch.int32)
    context_lens[0] = max_seq_len
    max_context_len = context_lens.max().item()
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size

    attn_scale = head_size**-0.5
    q = torch.empty(num_seqs, num_query_heads, head_size)
    q.uniform_(-attn_scale, attn_scale)

    k_cache = torch.empty(NUM_BLOCKS, num_kv_heads, block_size, head_size)
    k_cache.uniform_(-attn_scale, attn_scale)
    v_cache = torch.empty_like(k_cache)
    v_cache.uniform_(-attn_scale, attn_scale)

    # (NUM_SEQS, MAX_NUM_BLOCKS_PER_SEQ)
    block_tables = torch.randint(0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq))

    

    ref_out = torch_paged_attention(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        attn_scale,
    )
    
    for i in range(8):
        out = dlblas.paged_attention(
        q,
        k_cache,
        v_cache,
        context_lens,
        block_tables,
        attn_scale,
        max_context_len
        )
        # print(torch.abs(out - ref_out).max())
        assert torch.allclose(out, ref_out, atol=2e-3, rtol=1e-5)
        fn = lambda: dlblas.paged_attention(q, k_cache, v_cache, context_lens, block_tables, attn_scale, max_context_len)
        ms = triton.testing.do_bench(fn, warmup=20, rep=20)
        print(f"num_splits:{i}, ms:{ms}")

bench()