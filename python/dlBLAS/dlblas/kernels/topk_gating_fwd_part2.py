import torch
import triton
import triton.language as tl
import triton.language.core as tlc
from dlblas.utils import register_dlblas_op, SymVar, Tensor, ChoiceSpace
# from tutel import moe as tutel_moe


@triton.jit
def _topk_gating_fwd_part2(
    gates,
    masks,
    locations,
    res,
    ce,
    stride_s,
    SEQ_LEN: tl.constexpr, 
    BLOCK_S: tl.constexpr, 
    K: tl.constexpr,
    EXPERTS: tl.constexpr,
    KS: tl.constexpr,
    BLOCK_KS: tl.constexpr,
):
    pid_e = tl.program_id(axis=0)
    offs_ks = tl.arange(0, BLOCK_KS)
    offs_g = tl.arange(0, BLOCK_S)
    masks_ptrs = masks + offs_ks * stride_s + pid_e
    mask0_data = tl.load(masks_ptrs, mask=offs_ks < SEQ_LEN)
    masks_data = tl.load(masks_ptrs, mask=offs_ks < KS)
    loctions_data = tl.cumsum(masks_data, axis=0) - 1
    gates_ptrs = gates + offs_g * stride_s + pid_e
    gates_data = tl.load(gates_ptrs, mask=offs_g < SEQ_LEN)
    me = tl.sum(gates_data, axis=0) / SEQ_LEN
    ce_data = tl.sum(mask0_data, axis=0) / SEQ_LEN
    mul = me * ce_data * EXPERTS * EXPERTS
    res_ptrs = res + pid_e
    ce_ptrs = ce + pid_e
    locations_ptrs = locations + offs_ks * stride_s + pid_e
    tl.store(locations_ptrs, loctions_data, mask=offs_ks < KS)
    tl.store(res_ptrs, mul, mask=pid_e < EXPERTS)
    tl.store(ce_ptrs, ce_data, mask=pid_e < EXPERTS)


def call(gates: torch.Tensor, masks: torch.Tensor, k: int):
    s, e = gates.shape
    stride_se_s, _ = gates.stride()
    locations = torch.empty((k, s, e), dtype=torch.int64, device=gates.device)
    res = torch.empty((e,), dtype=gates.dtype, device=logits.device)
    ce = torch.empty_like(res)
    _topk_gating_fwd_part2[(e,)](
        gates,
        masks,
        locations,
        res,
        ce,
        stride_se_s,
        SEQ_LEN = s,
        BLOCK_S= triton.next_power_of_2(s),
        K = k,
        EXPERTS = e,
        KS = k * s,
        BLOCK_KS = triton.next_power_of_2(k * s),
    )
    return locations, res, ce
    # locations = tutel_moe.fast_cumsum_sub_one(masks.view(-1, e))
    # return locations.reshape(k,s,e), exp_counts, res, ce


def bench_fn(gates: torch.Tensor, masks: torch.Tensor, k: int):
    fn = lambda: call(gates, masks, k)
    ms = triton.testing.do_bench(fn, warmup=20, rep=20)
    return ms


# register
name = '_topk_gating_fwd_part2'
for dtype in [torch.float16, torch.float32]:
    for device in ['cuda']:
        seqLen, experts = SymVar('seqLen'), SymVar('experts')
        k = SymVar('k')
        # we dont' actually allocate tensor
        logits = Tensor((seqLen, experts), dtype=dtype, device=device)
        masks = Tensor((k, seqLen, experts), dtype=torch.int64, device=device)
        register_dlblas_op(name, None, (logits, masks, torch.SymInt), call, bench_fn, _topk_gating_fwd_part2)
