import torch
import triton
import triton.language as tl
import triton.language.core as tlc
from stable_argsort import argsort

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_S': BS}, num_stages=s, num_warps=w) \
        for BS in [2, 4] \
        for s in [1] \
        for w in [2] \
    ],
    key=['seq_len', 'K'],
)
@triton.jit
def _fused_kernel_1(
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
    logits_data = tl.load(logits_ptrs)
    logits_exp = tl.exp(logits_data)
    denom1 = tl.sum(logits_exp, axis=1)
    gates_data = logits_exp / denom1[:,None]
    tl.store(gates_ptrs, gates_data)

    for idx in tlc.static_range(K):
        gates_max = tl.max(gates_data, axis = 1)
        gates_max_b = tl.broadcast_to(tl.reshape(gates_max, (BLOCK_S, 1)), (BLOCK_S, EXPERTS))
        mask1_data = tl.zeros((BLOCK_S, EXPERTS), tl.int1)
        mask1_data = tl.where(gates_data == gates_max_b, 1, mask1_data)
        masks_ptrs = masks_ptr + idx * seq_len * EXPERTS + offs_s * stride_s + offs_e
        tl.store(masks_ptrs, mask1_data)
        gates_data = tl.where(mask1_data > 0, fill_value, gates_data)
    



@triton.jit
def _fused_kernel_2(
    gates,
    masks,
    locations,
    exp_counts,
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

    exp_counts_data = tl.sum(mask0_data, axis=0)
    tl.store(exp_counts + pid_e, exp_counts_data)

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



@triton.jit
def _fused_kernel_3(
    gates,
    gates_all,
    locations,
    masks,
    combine_weights,
    dispatch_mask,
    stride_se_s,
    stride_kse_k,
    stride_kse_s,
    stride_sec_s, 
    stride_sec_e,
    CAPACITY: tl.constexpr,
    BLOCK_C: tl.constexpr,
    min_value: tl.constexpr,
    K: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_k = tl.arange(0, BLOCK_K)[:,None]
    s_pid = tl.program_id(axis=0)
    offs_e = tl.arange(0, EXPERTS)[None,:]
    

    locations_ptrs = locations + offs_k * stride_kse_k + s_pid * stride_kse_s + offs_e
    locations_data = tl.load(locations_ptrs, mask=offs_k < K)
    masks_ptrs = masks + offs_k * stride_kse_k + s_pid * stride_kse_s + offs_e
    masks_data = tl.load(masks_ptrs, mask=offs_k < K)
    masks_data *= tl.where(locations_data < CAPACITY, 1, 0)

    # test
    # tl.store(masks_ptrs, masks_data, mask=offs_k < K)

    locations_data *= masks_data
    locations_s = tl.sum(locations_data, axis=1)
    
    gates_ptrs = gates + s_pid * stride_se_s + offs_e
   
    
    gates_data = tl.load(gates_ptrs)
    #gate_s = torch.einsum("se,kse->ks", gates, mask_float)
    multi = tl.broadcast_to(gates_data, (BLOCK_K, EXPERTS)) * masks_data
    gates_s = tl.sum(multi, axis=1)

    denom_s = tl.sum(gates_s, axis=0)
    # torch.clamp
    denom_s = tl.where(denom_s < min_value, min_value, denom_s)

    gates_s /= denom_s
    # tl.device_print("shape:", (gates_s.shape[1]))
    gates_s = tl.reshape(gates_s, (BLOCK_K, 1))
    gates_all_data = tl.broadcast_to(gates_s, (BLOCK_K, EXPERTS)) * masks_data.to(tl.float16)

    # test
    gates_all_ptrs = gates_all + offs_k * stride_kse_k + s_pid * stride_kse_s + offs_e
    tl.store(gates_all_ptrs, gates_all_data, mask=offs_k < K)

    locations_s = tl.broadcast_to(tl.reshape(locations_s,(BLOCK_K, 1)), (BLOCK_K, BLOCK_C))
    one_hot_help = tl.broadcast_to(tl.reshape(tl.arange(0, BLOCK_C), (1,BLOCK_C)), (BLOCK_K, BLOCK_C))
    loc_sc = tl.where(locations_s == one_hot_help, 1, 0)
    loc_sc = tl.broadcast_to(tl.reshape(loc_sc,(BLOCK_K, 1, BLOCK_C)), (BLOCK_K, EXPERTS, BLOCK_C))
    gates_all_data = tl.broadcast_to(tl.reshape(gates_all_data,(BLOCK_K, EXPERTS,1)), (BLOCK_K, EXPERTS, BLOCK_C))

    combine_weights_data = tl.reshape(tl.sum(gates_all_data * loc_sc, axis=0), (1, EXPERTS, BLOCK_C))
    
    offs_ksc = s_pid * stride_sec_s + tl.arange(0, EXPERTS)[None,:,None] * stride_sec_e + tl.arange(0, BLOCK_C)[None, None, :] 
   
    mask_ = tl.arange(0, BLOCK_C)[None, None, :]  < CAPACITY
    tl.store(combine_weights + offs_ksc, combine_weights_data, mask=mask_)
    tl.store(dispatch_mask + offs_ksc, tl.where(combine_weights_data > 0, 1, 0), mask=mask_)
   


def _fused_topk_gating(logits, k, capacity):
    s, e = logits.shape
    stride_s, _ = logits.stride()
    gates = torch.empty_like(logits)
    masks = torch.empty((k, s, e), dtype=torch.bool, device=logits.device)
    gates_all = torch.empty((k, s, e), dtype=logits.dtype, device=logits.device)
    locations = torch.empty((k, s, e), dtype=torch.int64, device=logits.device)
    exp_counts = torch.empty((e,), dtype=torch.int64, device=logits.device)
    fill_value = torch.finfo(logits.dtype).min
    res = torch.empty((e,), device=logits.device)
    ce = torch.empty_like(res)

    combine_weights = torch.empty((s, e, capacity), device=gates.device)
    dispatch_mask = torch.empty((s, e, capacity), device=gates.device, dtype=torch.bool)
    
    stride_sec_s, stride_sec_e, _ = combine_weights.stride()
    stride_kse_k, stride_kse_s, _ = masks.stride()
    min_value = torch.finfo(gates.dtype).eps
    with torch.cuda.device(logits.device.index):
        grid1 = lambda META: (triton.cdiv(s, META["BLOCK_S"]), )
        _fused_kernel_1[grid1](
            logits,
            masks,
            gates,
            fill_value,
            stride_s,
            seq_len = s,
            K = k,
            EXPERTS = e,
        )

        _fused_kernel_2[(e,)](
            gates,
            masks,
            locations,
            exp_counts,
            res,
            ce,
            stride_s,
            SEQ_LEN = s,
            BLOCK_S= triton.next_power_of_2(s),
            K = k,
            EXPERTS = e,
            KS = k * s,
            BLOCK_KS = triton.next_power_of_2(k * s),
        )
        # print(f"triton gates:{gates}")
        # print(f"triton res:{torch.mean(res)}")
        # print(f"triton ce:{ce}")
        # print(f"triton capacity:{capacity}")

        # print(f"mask:{masks}")

        _fused_kernel_3[(s,)](
            gates,
            gates_all,
            locations,
            masks,
            combine_weights,
            dispatch_mask,
            stride_s,
            stride_kse_k,
            stride_kse_s,
            stride_sec_s, 
            stride_sec_e,
            CAPACITY = capacity,
            BLOCK_C = triton.next_power_of_2(capacity),
            min_value = min_value,
            K = k,
            EXPERTS = e,
            BLOCK_K = triton.next_power_of_2(k),
        )

        #test
        # locations_s = torch.randn((k, s), dtype=torch.float16, device=logits.device)
       
        # locations_sc = torch.nn.functional.one_hot(locations_s, num_classes=capacity).type_as(logits)
        # combine_sec = torch.einsum("kse,ksc->ksec", gates_all, locations_sc)
        # combine_weights = torch.sum(combine_sec, dim=0)
        # dispatch_mask = combine_weights.bool()
        
        # print(f"triton combine_weights:{combine_weights}")
    
    return res, combine_weights, dispatch_mask, exp_counts.to("cpu"), (locations, masks, gates, ce)


