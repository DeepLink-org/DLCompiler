import torch
import triton
import triton.language as tl
import triton.language.core as tlc
from dlblas.utils import register_dlblas_op, SymVar, Tensor

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_S': BS}, num_stages=s, num_warps=w) \
        for BS in [1, 2, 4] \
        for s in [1] \
        for w in [1] \
    ],
    key=['s'],
)
@triton.jit
def _topk_gating_bwd_kernel(
    locations_kse,
    ce,
    masks,
    gates,
    diag_mask,
    grad_logits,
    stride_kse_k, stride_kse_s,
    grad_l_aux: tl.constexpr,
    min_value: tl.constexpr,
    k:tl.constexpr, s: tl.constexpr, e: tl.constexpr, c: tl.constexpr, 
    BLOCK_K: tl.constexpr, BLOCK_S: tl.constexpr
):
    offs_k = tl.arange(0, BLOCK_K)[:,None]
    # pid_s = tl.program_id(axis=0)
    pid = tl.program_id(axis=0)
    offs_s = pid * BLOCK_S  + tl.arange(0, BLOCK_S)[:,None]

    e_offset = tl.arange(0, e)
    gates_ptrs = gates + offs_s * stride_kse_s + e_offset
    ce_ptrs = ce + e_offset
    diag_mask_ptrs = diag_mask + e_offset[:, None] * e + e_offset[None, :]
    gates = tl.load(gates_ptrs)
    ce = tl.load(ce_ptrs)
    diag_mask = tl.load(diag_mask_ptrs)
    #
    diag_mask = tl.broadcast_to(tl.expand_dims(diag_mask, axis=0), (BLOCK_S, e, e))
    
    grad_me = grad_l_aux * ce * e * e / e
    grad_gates_t = (tl.zeros((e, ), dtype=tl.float32) + 1) * grad_me / s
    grad_gates = grad_gates_t 
    
    grad_gates_expand = tl.expand_dims(grad_gates, axis=0)
    grad_gates_expand = tl.broadcast_to(grad_gates_expand, (e, e))
    grad_gates_expand = tl.expand_dims(grad_gates_expand, axis=0)
    gates_expand = tl.expand_dims(gates, axis=2)
    gates_in1 = tl.broadcast_to(gates_expand, (BLOCK_S, e, e))
    gates_in2 = tl.broadcast_to(tl.trans(gates_expand, 0,2,1), (BLOCK_S, e, e))
    ger = gates_in1 * gates_in2
    softmax_grad = diag_mask * gates_in1 - ger
    grad_logits_data = tl.sum(softmax_grad * tl.broadcast_to(grad_gates_expand, (BLOCK_S, e, e)), axis=2)
    grad_logits_ptrs = grad_logits + offs_s * stride_kse_s + e_offset
    tl.store(grad_logits_ptrs, grad_logits_data)


def call(grad_l_aux, locations, masks, gates, ce):

    k = locations.shape[0]
    k, s, e = masks.shape
    stride_kse_k, stride_kse_s, _ = masks.stride()
    
    grad_logits = torch.empty((s,e), dtype=gates.dtype, device=ce.device)
    diag_mask = torch.diag(torch.ones(e, device=ce.device))
    assert e == triton.next_power_of_2(e)
    
    grid = lambda META: (triton.cdiv(s, META["BLOCK_S"]), )
    _topk_gating_bwd_kernel[grid](
        locations,
        ce,
        masks,
        gates,
        diag_mask,
        grad_logits,
        stride_kse_k, stride_kse_s,
        grad_l_aux,
        torch.finfo(gates.dtype).eps,
        k, s, e, c,
        BLOCK_K=triton.next_power_of_2(k)
    )
    # print(f"_bwd_kernel.best_config ", _topk_gating_bwd_kernel.best_config, flush = True)
    return grad_logits


def bench_fn(grad_l_aux, locations, masks, gates, ce):
    fn = lambda: call(grad_l_aux, locations, masks, gates, ce)
    ms = triton.testing.do_bench(fn, warmup=20, rep=20)
    return ms


# register
name = '_topk_gating_bwd'
for dtype in [torch.float16, torch.float32]:
    for device in ['cuda']:
        seqLen, experts = SymVar('seqLen'), SymVar('experts')
        k, c= SymVar('k'), SymVar('c')
        # we dont' actually allocate tensor
        # grad_combine = Tensor((seqLen, experts, c), dtype=dtype, device=device)
        locations = Tensor((k, seqLen, experts), dtype=torch.int64, device=device)
        masks = Tensor((k, seqLen, experts), dtype=torch.int64, device=device)
        gates = Tensor((seqLen, experts), dtype=dtype, device=device)
        ce = Tensor((experts,), dtype=dtype, device=device)
        register_dlblas_op(name, None, (torch.SymFloat, locations, masks, gates, ce),
                           call, bench_fn, _topk_gating_bwd_kernel)
