################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
import flashinfer
import torch
from typing import List, Optional


def prepare_cos_sin_cache(head_dim, max_position_embeddings, rope_theta):
    device = torch.cuda.current_device()
    inv_freq = 1.0 / (rope_theta**(torch.arange(0, head_dim, 2).float().to(device) / head_dim))

    t = torch.arange(max_position_embeddings, device=device, dtype=inv_freq.dtype)

    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rmsnorm_ref(x, w, eps=1e-6):
    return flashinfer.norm.rmsnorm(x.view(-1, x.size(-1)), w, eps).view_as(x)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor, cos_sin_cache: torch.Tensor):
    """Applies Rotary Position Embedding inplace."""
    q = q.clone()
    k = k.clone()
    bsz, seq, _, head_dim = q.shape
    flashinfer.apply_rope_with_cos_sin_cache_inplace(position_ids, q.view(bsz * seq, -1), k.view(bsz * seq, -1),
                                                     head_dim, cos_sin_cache, True),
    q = q.view(bsz, seq, -1, head_dim)
    k = k.view(bsz, seq, -1, head_dim)
    return q, k


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx:start_idx + query_len]
        q = q * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len).cuda()
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool().cuda()
        if sliding_window is not None:
            sliding_window_mask = torch.triu(empty_mask,
                                             diagonal=kv_len - (query_len + sliding_window) + 1).bool().logical_not()
            mask |= sliding_window_mask
        if soft_cap > 0.0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len
    return torch.cat(outputs, dim=0)


def torch_gate_silu_mul_up(x):
    intermediate_size = x.shape[1] // 2
    out = torch.empty((x.shape[0], intermediate_size), device=x.device, dtype=x.dtype)
    gate_state, up_state = torch.chunk(x, 2, dim=-1)
    out = torch.nn.functional.silu(gate_state) * up_state
    return out


def torch_all_reduce(local_input: torch.Tensor, pg: torch.distributed.ProcessGroup):
    output = torch.clone(local_input)
    torch.distributed.all_reduce(output, group=pg)
    return output
