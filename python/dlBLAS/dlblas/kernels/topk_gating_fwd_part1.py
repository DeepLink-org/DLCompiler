import torch
import triton
import triton.language as tl
import triton.language.core as tlc
from dlblas.utils import register_dlblas_op, SymVar, Tensor, ChoiceSpace

def get_autotune_config():
    return [
        triton.Config({'BLOCK_S': BS}, num_stages=s, num_warps=w) \
        for BS in [2, 4] \
        for s in [1] \
        for w in [1, 2] \
    ]

@triton.jit
def _topk_gating_kernel_part1(
    logits_ptr,
    masks_ptr, # output
    gates_ptr, 
    fill_value,
    stride_s,
    seq_len,
    K: tl.constexpr,
    BLOCK_S: tl.constexpr,
    EXPERTS: tl.constexpr,
):
    # the softmax computation for each row is independent
    # each block process each row
    pid = tl.program_id(axis=0)
    
    offs_s = pid * BLOCK_S  + tl.arange(0, BLOCK_S)[:,None]
    offs_e = tl.arange(0, EXPERTS)[None,:]

    logits_ptrs = logits_ptr + offs_s * stride_s + offs_e
    gates_ptrs = gates_ptr + offs_s * stride_s + offs_e

    # load data
    logits_data = tl.load(logits_ptrs, mask=offs_s < seq_len)
    logits_exp = tl.exp(logits_data)
    denom1 = tl.sum(logits_exp, axis=1)
    gates_data = logits_exp / denom1[:,None]
    tl.store(gates_ptrs, gates_data, mask=offs_s < seq_len)

    for idx in tlc.static_range(K):
        gates_max = tl.max(gates_data, axis = 1)
        gates_max_b = tl.broadcast_to(tl.reshape(gates_max, (BLOCK_S, 1)), (BLOCK_S, EXPERTS))
        mask1_data = tl.zeros((BLOCK_S, EXPERTS), tl.int64)
        mask1_data = tl.where(gates_data == gates_max_b, 1, mask1_data)
        masks_ptrs = masks_ptr + idx * seq_len * EXPERTS + offs_s * stride_s + offs_e
        tl.store(masks_ptrs, mask1_data, mask=offs_s < seq_len)
        gates_data = tl.where(mask1_data > 0, fill_value, gates_data)


def call(logits: torch.Tensor, k: int):
    s, e = logits.shape
    stride_se_s, _ = logits.stride()
    gates = torch.empty_like(logits)
    masks = torch.empty((k, s, e), dtype=torch.int64, device=logits.device)
    fill_value = torch.finfo(logits.dtype).min

    grid = lambda META: (triton.cdiv(s, META["BLOCK_S"]), )
    _topk_gating_kernel_part1[grid](
        logits,
        masks,
        gates,
        fill_value,
        stride_se_s,
        seq_len = s,
        K = k,
        EXPERTS = e,
    )
    return gates, masks


def bench_fn(logits: torch.Tensor, k: int):
    fn = lambda: call(logits, k)
    ms = triton.testing.do_bench(fn, warmup=100, rep=100)
    return ms


# register
name = '_topk_gating_fwd_part1'
for dtype in [torch.float16, torch.float32]:
    for device in ['cuda']:
        seqLen, experts = SymVar('seqLen'), SymVar('experts')
        k = SymVar('k')
        # we dont' actually allocate tensor
        logits = Tensor((seqLen, experts), dtype=dtype, device=device)
        space = ChoiceSpace(get_autotune_config())
        register_dlblas_op(name, space, (logits, torch.SymInt), call, bench_fn, _topk_gating_kernel_part1)
