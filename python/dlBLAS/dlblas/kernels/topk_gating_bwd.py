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
    BLOCK_SIZE_e: tl.constexpr,
):
    # offs_k = tl.arange(0, BLOCK_K)[:,None]
    pid_s = tl.program_id(axis=0)
    e_offset = tl.arange(0, BLOCK_SIZE_e)
    gates_ptrs = gates + pid_s * stride_kse_s + e_offset
    ce_ptrs = ce + e_offset
    diag_mask_ptrs = diag_mask + e_offset[:, None] * e + e_offset[None, :]
    gates = tl.load(gates_ptrs, mask=e_offset < e)
    ce = tl.load(ce_ptrs, mask=e_offset < e)
    diag_mask = tl.load(diag_mask_ptrs, mask=e_offset[:,None] < e and e_offset[None,:] < e)
    
    locations0_se_ptrs = locations_kse + 0 * stride_kse_k + pid_s * stride_kse_s + e_offset
    locations1_se_ptrs = locations_kse + 1 * stride_kse_k + pid_s * stride_kse_s + e_offset
    locations2_se_ptrs = locations_kse + 2 * stride_kse_k + pid_s * stride_kse_s + e_offset
    locations3_se_ptrs = locations_kse + 3 * stride_kse_k + pid_s * stride_kse_s + e_offset
    locations4_se_ptrs = locations_kse + 4 * stride_kse_k + pid_s * stride_kse_s + e_offset
    locations5_se_ptrs = locations_kse + 5 * stride_kse_k + pid_s * stride_kse_s + e_offset
   
    mask0_ptrs = masks + 0 * stride_kse_k + pid_s * stride_kse_s + e_offset
    mask1_ptrs = masks + 1 * stride_kse_k + pid_s * stride_kse_s + e_offset
    mask2_ptrs = masks + 2 * stride_kse_k + pid_s * stride_kse_s + e_offset
    mask3_ptrs = masks + 3 * stride_kse_k + pid_s * stride_kse_s + e_offset
    mask4_ptrs = masks + 4 * stride_kse_k + pid_s * stride_kse_s + e_offset
    mask5_ptrs = masks + 5 * stride_kse_k + pid_s * stride_kse_s + e_offset
    
    
    locations0_se = tl.load(locations0_se_ptrs, mask=e_offset < e)
    locations1_se = tl.load(locations1_se_ptrs, mask=e_offset < e)
    locations2_se = tl.load(locations2_se_ptrs, mask=e_offset < e)
    locations3_se = tl.load(locations3_se_ptrs, mask=e_offset < e)
    locations4_se = tl.load(locations4_se_ptrs, mask=e_offset < e)
    locations5_se = tl.load(locations5_se_ptrs, mask=e_offset < e)

    mask0_data = tl.load(mask0_ptrs, mask=e_offset < e)
    mask1_data = tl.load(mask1_ptrs, mask=e_offset < e)
    mask2_data = tl.load(mask2_ptrs, mask=e_offset < e)
    mask3_data = tl.load(mask3_ptrs, mask=e_offset < e)
    mask4_data = tl.load(mask4_ptrs, mask=e_offset < e)
    mask5_data = tl.load(mask5_ptrs, mask=e_offset < e)
    
    locations0_s = tl.sum(locations0_se * mask0_data, axis=0).to(tl.int32)
    locations1_s = tl.sum(locations1_se * mask1_data, axis=0).to(tl.int32)
    locations2_s = tl.sum(locations2_se * mask2_data, axis=0).to(tl.int32)
    locations3_s = tl.sum(locations3_se * mask3_data, axis=0).to(tl.int32)
    locations4_s = tl.sum(locations4_se * mask4_data, axis=0).to(tl.int32)
    locations5_s = tl.sum(locations5_se * mask5_data, axis=0).to(tl.int32)

    grad_gates0_ptrs = grad_combine + pid_s * stride_sec_s + e_offset * stride_sec_e + locations0_s
    grad_gates0 = tl.load(grad_gates0_ptrs, locations0_s < c)
    grad_gates1_ptrs = grad_combine + pid_s * stride_sec_s + e_offset * stride_sec_e + locations1_s
    grad_gates1 = tl.load(grad_gates1_ptrs, locations1_s < c)
    grad_gates2_ptrs = grad_combine + pid_s * stride_sec_s + e_offset * stride_sec_e + locations2_s
    grad_gates2 = tl.load(grad_gates2_ptrs, locations2_s < c)
    grad_gates3_ptrs = grad_combine + pid_s * stride_sec_s + e_offset * stride_sec_e + locations3_s
    grad_gates3 = tl.load(grad_gates3_ptrs, locations3_s < c)
    grad_gates4_ptrs = grad_combine + pid_s * stride_sec_s + e_offset * stride_sec_e + locations4_s
    grad_gates4 = tl.load(grad_gates4_ptrs, locations4_s < c)
    grad_gates5_ptrs = grad_combine + pid_s * stride_sec_s + e_offset * stride_sec_e + locations5_s
    grad_gates5 = tl.load(grad_gates5_ptrs, locations5_s < c)
    
    grad_gates0_s = tl.sum(grad_gates0 * mask0_data, axis=0)
    grad_gates1_s = tl.sum(grad_gates1 * mask1_data, axis=0)
    grad_gates2_s = tl.sum(grad_gates2 * mask2_data, axis=0)
    grad_gates3_s = tl.sum(grad_gates3 * mask3_data, axis=0)
    grad_gates4_s = tl.sum(grad_gates4 * mask4_data, axis=0)
    grad_gates5_s = tl.sum(grad_gates5 * mask5_data, axis=0)
    
    # compute the gates1_s and gates2_s to re-compute the denom_s in forward
    gates0_s = tl.sum(gates * mask0_data, axis=0)
    gates1_s = tl.sum(gates * mask1_data, axis=0)
    gates2_s = tl.sum(gates * mask2_data, axis=0)
    gates3_s = tl.sum(gates * mask3_data, axis=0)
    gates4_s = tl.sum(gates * mask4_data, axis=0)
    gates5_s = tl.sum(gates * mask5_data, axis=0)
    
    denom_s = gates0_s + gates1_s + gates2_s + gates3_s + gates4_s + gates5_s
    denom_s_output = tl.where(denom_s < min_value, min_value, denom_s)

    grad_denom_s = grad_gates0_s * gates0_s + grad_gates1_s * gates1_s + grad_gates2_s * gates2_s + grad_gates3_s * gates3_s + grad_gates4_s * gates4_s + grad_gates5_s * gates5_s
    grad_denom_s = -(grad_denom_s) / (denom_s * denom_s)
    grad_denom_s = tl.where(denom_s < min_value, 0, grad_denom_s)
    
    grad_gates0_s_ = grad_gates0_s / denom_s_output
    grad_gates1_s_ = grad_gates1_s / denom_s_output
    grad_gates2_s_ = grad_gates2_s / denom_s_output
    grad_gates3_s_ = grad_gates3_s / denom_s_output
    grad_gates4_s_ = grad_gates4_s / denom_s_output
    grad_gates5_s_ = grad_gates5_s / denom_s_output
    
    grad_gates0 = (grad_gates0_s_ + grad_denom_s) * mask0_data
    grad_gates1 = (grad_gates1_s_ + grad_denom_s) * mask1_data
    grad_gates2 = (grad_gates2_s_ + grad_denom_s) * mask2_data
    grad_gates3 = (grad_gates3_s_ + grad_denom_s) * mask3_data
    grad_gates4 = (grad_gates4_s_ + grad_denom_s) * mask4_data
    grad_gates5 = (grad_gates5_s_ + grad_denom_s) * mask5_data
    
    grad_me = grad_l_aux * ce * e * e / e
    grad_gates_t = (tl.zeros((e, ), dtype=tl.float32) + 1) * grad_me / s
    
    grad_gates = grad_gates0 + grad_gates1 + grad_gates2 + grad_gates3 + grad_gates4 + grad_gates5 + grad_gates_t
    
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
    
    grad_logits = torch.empty((s,e), device=grad_combine.device)
    diag_mask = torch.diag(torch.ones(e, device=grad_combine.device))
    block_size_e = triton.next_power_of_2(e)
    
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
            block_size_e,
        )
        
    return grad_logits


def bench_fn(grad_l_aux, grad_combine, locations, masks, gates, ce):
    return float('inf')


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
