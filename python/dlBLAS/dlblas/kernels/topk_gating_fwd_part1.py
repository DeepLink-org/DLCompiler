import torch
import triton
import triton.language as tl
import triton.language.core as tlc
from dlblas.utils import register_dlblas_op, SymVar, Tensor, ChoiceSpace
if triton.__version__ >= "3.0.0":
    from triton.language.extra.cuda.libdevice import fast_expf as tl_exp
else:
    from triton.language.math import fast_expf as tl_exp


@triton.autotune(
    configs = [
        triton.Config({'BLOCK_S': BS}, num_stages=s, num_warps=w) \
        for BS in [2, 4] \
        for s in [1] \
        for w in [1, 2] \
    ],
    key=['seq_len'],
)
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
    logits_exp = tl_exp(logits_data)
    denom1 = tl.sum(logits_exp, axis=1)
    gates_data = logits_exp / denom1[:,None]
    tl.store(gates_ptrs, gates_data, mask=offs_s < seq_len)

    for idx in tlc.static_range(K):
        max_idx = tl.argmax(gates_data, axis = 1, tie_break_left=False)
        mask_data = tl.zeros((BLOCK_S, EXPERTS), tl.int64)
        all_ids = tl.broadcast_to(tl.arange(0, EXPERTS)[None,:], (BLOCK_S, EXPERTS))
        max_idx = tl.broadcast_to(tl.reshape(max_idx, (BLOCK_S, 1)), (BLOCK_S, EXPERTS))
        mask_data = tl.where(all_ids == max_idx, 1, mask_data)
        masks_ptrs = masks_ptr + idx * seq_len * EXPERTS + offs_s * stride_s + offs_e
        tl.store(masks_ptrs, mask_data, mask=offs_s < seq_len)
        gates_data = tl.where(mask_data > 0, fill_value, gates_data)


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
    ms = triton.testing.do_bench(fn, warmup=20, rep=20)
    return ms


# register
name = '_topk_gating_fwd_part1'
for dtype in [torch.float16, torch.float32]:
    for device in ['cuda']:
        seqLen, experts = SymVar('seqLen'), SymVar('experts')
        k = SymVar('k')
        # we dont' actually allocate tensor
        logits = Tensor((seqLen, experts), dtype=dtype, device=device)
        register_dlblas_op(name, None, (logits, torch.SymInt), call, bench_fn, _topk_gating_kernel_part1)
