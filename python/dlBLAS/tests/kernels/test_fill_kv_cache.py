import torch
import dlblas
import triton


class ReshapePagedCache(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        num_tokens = k.shape[0]
        block_size = k_cache.shape[1]
        for i in range(num_tokens):
            if slot_mapping[i] >= 0:
                block_id = torch.div(slot_mapping[i], block_size, rounding_mode="floor")
                block_offset = slot_mapping[i] % block_size
                k_cache[block_id, block_offset, :, :] = k[i]
                v_cache[block_id, block_offset, :, :] = v[i]


def test_fill_kv_cache_0():
    test_cases = 10

    num_tokens_list = torch.randint(
        low=1, high=1024, size=(test_cases,), dtype=torch.int32
    )
    num_heads_list = torch.randint(
        low=1, high=64, size=(test_cases,), dtype=torch.int32
    )
    head_size_list = torch.randint(
        low=1, high=16, size=(test_cases,), dtype=torch.int32
    )
    block_size_list = torch.randint(
        low=1, high=4, size=(test_cases,), dtype=torch.int32
    )
    block_size_list *= 16

    for i in range(test_cases):
        num_tokens = num_tokens_list[i]
        num_heads = num_heads_list[i]
        head_dim_k = head_size_list[i]
        head_dim_v = head_size_list[test_cases - i - 1]
        block_size = block_size_list[i]
        if block_size != triton.next_power_of_2(block_size):
            continue

        num_blocks = (int)((num_tokens + block_size - 1) / block_size)
        print(
            f"num_tokens: {num_tokens}, num_heads: {num_heads}, head_dim_k: {head_dim_k}, head_dim_v:{head_dim_v}, num_blocks: {num_blocks}, block_size: {block_size}, testing..."
        )
        key = torch.randn(num_tokens, num_heads, head_dim_k, dtype=torch.half).cuda()
        value = torch.randn(num_tokens, num_heads, head_dim_v, dtype=torch.half).cuda()
        key_cache = torch.randn(
            num_blocks, block_size, num_heads, head_dim_k, dtype=torch.half
        ).cuda()
        value_cache = torch.randn(
            num_blocks, block_size, num_heads, head_dim_v, dtype=torch.half
        ).cuda()

        # num_slots = num_blocks * block_size
        # slot_mapping = random.sample(range(num_tokens), num_tokens)
        # slot_mapping = torch.tensor(slot_mapping, dtype=torch.int).cuda()
        slot_mapping = torch.tensor(range(num_tokens), dtype=torch.int).cuda()
        if num_blocks > 2:
            tmp = slot_mapping[0:block_size].clone()
            slot_mapping[0:block_size] = slot_mapping[block_size : 2 * block_size]
            slot_mapping[block_size : 2 * block_size] = tmp

        ref_key_cache, ref_value_cache = key_cache.clone(), value_cache.clone()
        reshape_paged_cache = ReshapePagedCache()

        reshape_paged_cache(key, value, ref_key_cache, ref_value_cache, slot_mapping)
        key_cache, value_cache = dlblas.fill_kv_cache(
            key, value, key_cache, value_cache, slot_mapping
        )
        key_cache = key_cache.cpu().float()
        ref_key_cache = ref_key_cache.cpu().float()
        value_cache = value_cache.cpu().float()
        ref_value_cache = ref_value_cache.cpu().float()
        assert torch.allclose(key_cache, ref_key_cache) and torch.allclose(
            value_cache, ref_value_cache
        )
        max_diff = max(
            torch.max(torch.abs(key_cache - ref_key_cache)),
            torch.max(torch.abs(value_cache - ref_value_cache)),
        )
        assert max_diff < 0.0001


if __name__ == "__main__":
    test_fill_kv_cache_0()
    print("success!")
