import torch
import triton
import triton.language as tl
import triton.language.core as tlc
from dlblas.utils import register_dlblas_op, SymVar, Tensor

@triton.jit
def _topk_gating_bwd_kernel(
    grad_combine,
    locations_kse,
    ce,
    masks,
    gates,
    diag_mask,
    grad_logits,
    stride_kse_k, stride_kse_s,
    stride_sec_s, stride_sec_e,
    grad_l_aux: tl.constexpr,
    min_value: tl.constexpr,
    k:tl.constexpr, s: tl.constexpr, e: tl.constexpr, c: tl.constexpr, 
    BLOCK_K: tl.constexpr
):
    offs_k = tl.arange(0, BLOCK_K)[:,None]
    pid_s = tl.program_id(axis=0)
    e_offset = tl.arange(0, e)
    gates_ptrs = gates + pid_s * stride_kse_s + e_offset
    ce_ptrs = ce + e_offset
    diag_mask_ptrs = diag_mask + e_offset[:, None] * e + e_offset[None, :]
    gates = tl.load(gates_ptrs)
    ce = tl.load(ce_ptrs)
    diag_mask = tl.load(diag_mask_ptrs)

    locations_se_ptrs = locations_kse + offs_k * stride_kse_k + pid_s * stride_kse_s + e_offset
    locations_se = tl.load(locations_se_ptrs, mask=offs_k < k)
    mask_ptrs = masks + offs_k * stride_kse_k + pid_s * stride_kse_s + e_offset
    mask_data = tl.load(mask_ptrs, mask=offs_k < k)
    locations_s = tl.sum(locations_se * mask_data, axis=1)
    grad_gates_ptrs = grad_combine + pid_s * stride_sec_s + e_offset[:,None] * stride_sec_e + locations_s[None,:]
    grad_gates = tl.load(grad_gates_ptrs)
    grad_gates_s = tl.sum(tl.trans(grad_gates) * mask_data, axis=1)
    gates_s = tl.sum(gates * mask_data, axis=1)

    denom_s = tl.sum(gates_s, axis=0)
    denom_s_output = tl.where(denom_s < min_value, min_value, denom_s)

    grad_denom_s = tl.sum(grad_gates_s * gates_s, axis=0)
    
    grad_denom_s = -(grad_denom_s) / (denom_s * denom_s)
    grad_denom_s = tl.where(denom_s < min_value, 0, grad_denom_s)
    grad_gates_s_ = grad_gates_s / denom_s_output
  
    grad_gates_s_ = tl.broadcast_to(tl.reshape(grad_gates_s_, (BLOCK_K, 1)), (BLOCK_K, e))
    # grad_denom_s = tl.broadcast_to(tl.reshape(grad_denom_s, (1, e)), (k, e))
    grad_gates = (grad_gates_s_ + grad_denom_s) * mask_data
    
    grad_me = grad_l_aux * ce * e * e / e
    grad_gates_t = (tl.zeros((e, ), dtype=tl.float32) + 1) * grad_me / s
    grad_gates = grad_gates_t + tl.sum(grad_gates, axis=0)
    
    grad_gates_expand = tl.expand_dims(grad_gates, axis=0)
    gates_expand = tl.expand_dims(gates, axis=0)
    gates_in1 = tl.broadcast_to(gates_expand, (e, e))
    gates_in2 = tl.broadcast_to(tl.trans(gates_expand), (e, e))
    ger = gates_in1 * gates_in2
    softmax_grad = diag_mask * gates_in1 - ger
    grad_logits_data = tl.sum(softmax_grad * tl.broadcast_to(grad_gates_expand, (e, e)), axis=1)
    grad_logits_ptrs = grad_logits + pid_s * stride_kse_s + e_offset
    tl.store(grad_logits_ptrs, grad_logits_data)


def call(grad_l_aux, grad_combine, locations, masks, gates, ce):
    assert grad_combine.shape[0] == locations.shape[1]
    assert grad_combine.shape[1] == locations.shape[2]
    s, e, c = grad_combine.shape
    k = locations.shape[0]
    stride_sec_s, stride_sec_e, _ = grad_combine.stride()
    stride_kse_k, stride_kse_s, _ = masks.stride()
    
    grad_logits = torch.empty((s,e), dtype=gates.dtype, device=grad_combine.device)
    diag_mask = torch.diag(torch.ones(e, device=grad_combine.device))
    assert e == triton.next_power_of_2(e)
    
    with torch.cuda.device(grad_combine.device.index):
        _topk_gating_bwd_kernel[(s, )](
            grad_combine,
            locations,
            ce,
            masks,
            gates,
            diag_mask,
            grad_logits,
            stride_kse_k, stride_kse_s,
            stride_sec_s, stride_sec_e,
            grad_l_aux,
            torch.finfo(gates.dtype).eps,
            k, s, e, c,
            BLOCK_K=triton.next_power_of_2(k)
        )
    return grad_logits


def bench_fn(grad_l_aux, grad_combine, locations, masks, gates, ce):
    fn = lambda: call(grad_l_aux, grad_combine, locations, masks, gates, ce)
    ms = triton.testing.do_bench(fn, warmup=20, rep=20)
    return ms


# register
name = '_topk_gating_bwd'
for dtype in [torch.float16, torch.float32]:
    for device in ['cuda']:
        seqLen, experts = SymVar('seqLen'), SymVar('experts')
        k, c= SymVar('k'), SymVar('c')
        # we dont' actually allocate tensor
        grad_combine = Tensor((seqLen, experts, c), dtype=dtype, device=device)
        locations = Tensor((k, seqLen, experts), dtype=torch.int64, device=device)
        masks = Tensor((k, seqLen, experts), dtype=torch.int64, device=device)
        gates = Tensor((seqLen, experts), dtype=dtype, device=device)
        ce = Tensor((experts,), dtype=dtype, device=device)
        register_dlblas_op(name, None, (torch.SymFloat, grad_combine, locations, masks, gates, ce),
                           call, bench_fn, _topk_gating_bwd_kernel)
