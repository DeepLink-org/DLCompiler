import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def _fused_bwd_kernel(
    grad_combine,
    locations1_se,
    locations2_se,
    ce,
    mask1,
    mask2,
    gates,
    diag_mask,
    grad_logits,
    stride_sec_s, stride_sec_e,
    stride_se_s,
    grad_l_aux: tl.constexpr,
    min_value: tl.constexpr,
    s: tl.constexpr, e: tl.constexpr, c: tl.constexpr,
    BLOCK_SIZE_e: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    e_offset = tl.arange(0, BLOCK_SIZE_e)
    
    locations1_se_ptrs = locations1_se + pid * stride_se_s + e_offset
    locations2_se_ptrs = locations2_se + pid * stride_se_s + e_offset
    mask1_ptrs = mask1 + pid * stride_se_s + e_offset
    mask2_ptrs = mask2 + pid * stride_se_s + e_offset
    gates_ptrs = gates + pid * stride_se_s + e_offset
    ce_ptrs = ce + e_offset
    diag_mask_ptrs = diag_mask + e_offset[:, None] * e + e_offset[None, :]
    
    locations1_se = tl.load(locations1_se_ptrs, mask=e_offset < e)
    locations2_se = tl.load(locations2_se_ptrs, mask=e_offset < e)
    mask1_data = tl.load(mask1_ptrs, mask=e_offset < e)
    mask2_data = tl.load(mask2_ptrs, mask=e_offset < e)
    gates = tl.load(gates_ptrs, mask=e_offset < e)
    ce = tl.load(ce_ptrs, mask=e_offset < e)
    diag_mask = tl.load(diag_mask_ptrs, mask=e_offset[:,None] < e and e_offset[None,:] < e)
    
    locations1_s = tl.sum(locations1_se, axis=0).to(tl.int32)
    locations2_s = tl.sum(locations2_se, axis=0).to(tl.int32)

    grad_gates1_ptrs = grad_combine + pid * stride_sec_s + e_offset * stride_sec_e + locations1_s
    grad_gates1 = tl.load(grad_gates1_ptrs, locations1_s < c)
    grad_gates2_ptrs = grad_combine + pid * stride_sec_s + e_offset * stride_sec_e + locations2_s
    grad_gates2 = tl.load(grad_gates2_ptrs, locations1_s < c)
    
    grad_gates1_s = tl.sum(grad_gates1 * mask1_data, axis=0)
    grad_gates2_s = tl.sum(grad_gates2 * mask2_data, axis=0)
    
    # compute the gates1_s and gates2_s to re-compute the denom_s in forward
    gates1_s = tl.sum(gates * mask1_data, axis=0)
    gates2_s = tl.sum(gates * mask2_data, axis=0)
    
    denom_s = gates1_s + gates2_s
    denom_s_output = tl.where(denom_s < min_value, min_value, denom_s)
    grad_denom_s = -(grad_gates1_s * gates1_s + grad_gates2_s * gates2_s) / (denom_s * denom_s)
    grad_denom_s = tl.where(denom_s < min_value, 0, grad_denom_s)
    
    grad_gates1_s_ = grad_gates1_s / denom_s_output
    grad_gates2_s_ = grad_gates2_s / denom_s_output
 
    
    grad_gates1 = (grad_gates1_s_ + grad_denom_s) * mask1_data
    grad_gates2 = (grad_gates2_s_ + grad_denom_s) * mask2_data
    
    grad_me = grad_l_aux * ce * e * e / e
    grad_gates_3 = (tl.zeros((e, ), dtype=tl.float32) + 1) * grad_me / s
    
    grad_gates = grad_gates1 + grad_gates2 + grad_gates_3
    
    grad_gates_expand = tl.expand_dims(grad_gates, axis=0)
    gates_expand = tl.expand_dims(gates, axis=0)
    gates_in1 = tl.broadcast_to(gates_expand, (e, e))
    gates_in2 = tl.broadcast_to(tl.trans(gates_expand), (e, e))
    ger = gates_in1 * gates_in2
    softmax_grad = diag_mask * gates_in1 - ger
    grad_logits_data = tl.sum(softmax_grad * tl.broadcast_to(grad_gates_expand, (e, e)), axis=1)

    grad_logits_ptrs = grad_logits + pid * stride_se_s + e_offset

    tl.store(grad_logits_ptrs, grad_logits_data)

def fused_bwd(grad_l_aux, grad_combine, loca1, loca2, mask1, mask2, gates, ce):
    
    s, e, c = grad_combine.shape
    stride_sec_s, stride_sec_e, _ = grad_combine.stride()
    stride_se_s, _ = mask1.stride()
    
    grad_logits = torch.zeros((s,e), device=grad_combine.device)
    diag_mask = torch.diag(torch.ones(e, device=grad_combine.device))
    block_size_e = triton.next_power_of_2(e)
    
    with torch.cuda.device(grad_combine.device.index):
        
        _fused_bwd_kernel[(s, )](
            grad_combine,
            loca1,
            loca2,
            ce,
            mask1,
            mask2,
            gates,
            diag_mask,
            grad_logits,
            stride_sec_s, stride_sec_e,
            stride_se_s,
            grad_l_aux,
            torch.finfo(gates.dtype).eps,
            s, e, c,
            block_size_e,
        )
        
    return grad_logits

def BackwardTorch(grad_output1, grad_output2, locations1_se, locations2_se, mask1_float, mask2_float, gates, ce):
    gates1_s = torch.einsum("se,se->s", gates, mask1_float)
    gates2_s = torch.einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s_output = torch.clamp(denom_s, min=torch.finfo(gates.dtype).eps)
    
    s, e, c = grad_output1.shape
    
    locations1_s = torch.sum(locations1_se, dim=1).to(torch.int64)
    locations2_s = torch.sum(locations2_se, dim=1).to(torch.int64)
    
    locations1_sc = F.one_hot(locations1_s, num_classes=c).type_as(gates)
    locations2_sc = F.one_hot(locations2_s, num_classes=c).type_as(gates)
    
    grad_gates1 = torch.einsum("sec,sc->se", grad_output1, locations1_sc)
    grad_gates2 = torch.einsum("sec,sc->se", grad_output1, locations2_sc)
    
    grad_gates1_s = torch.einsum("se,se->s", grad_gates1, mask1_float)
    grad_gates2_s = torch.einsum("se,se->s", grad_gates2, mask2_float)
    
    grad_denom_s = -(grad_gates1_s * gates1_s + grad_gates2_s * gates2_s) / (denom_s ** 2)
    grad_denom_s[denom_s < torch.finfo(gates.dtype).eps] = 0
    
    grad_gates1_s_ = grad_gates1_s / denom_s_output
    grad_gates2_s_ = grad_gates2_s / denom_s_output
    
    grad_gates1_s_final = grad_gates1_s_ + grad_denom_s
    grad_gates2_s_final = grad_gates2_s_ + grad_denom_s
    
    grad_gates1 = torch.einsum("s,se->se", grad_gates1_s_final, mask1_float)
    grad_gates2 = torch.einsum("s,se->se", grad_gates2_s_final, mask2_float)
    
    s, e, c = grad_output1.shape
    grad_me = grad_output2 * e * e * ce /  ce.shape[0]
    grad_gates3 = grad_me / s * torch.ones((s, e), device=ce.device)
    
    grad_gates = grad_gates1 + grad_gates2 + grad_gates3
    
    grad_logits = []
    for i in range(grad_gates.shape[0]):
        softmax_grad = torch.diag(gates[i]) - torch.ger(gates[i], gates[i])
        grad_logits.append(torch.matmul(softmax_grad, grad_gates[i].t()))
    grad_logits = torch.stack(grad_logits)
    
    return grad_logits, None, None