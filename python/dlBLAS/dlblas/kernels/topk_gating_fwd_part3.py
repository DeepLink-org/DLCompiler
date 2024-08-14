import torch
import triton
import triton.language as tl
import triton.language.core as tlc
from dlblas.utils import register_dlblas_op, SymVar, Tensor, ChoiceSpace


@triton.jit
def _topk_gating_fwd_part3(
    gates,
    gates_all,
    locations,
    locations_s,
    masks,
    combine_weights,
    dispatch_mask,
    stride_ks_k,
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

    # for bwd
    tl.store(masks_ptrs, masks_data, mask=offs_k < K)

    locations_data *= masks_data
    locations_s_data = tl.reshape(tl.sum(locations_data, axis=1), (BLOCK_K, 1))
    
    gates_ptrs = gates + s_pid * stride_se_s + offs_e
   
    
    gates_data = tl.load(gates_ptrs)
    #gate_s = torch.einsum("se,kse->ks", gates, mask_float)
    multi = tl.broadcast_to(gates_data, (BLOCK_K, EXPERTS)) * masks_data
    gates_s = tl.sum(multi, axis=1)

    denom_s = tl.sum(gates_s, axis=0)
    # torch.clamp
    denom_s = tl.where(denom_s < min_value, min_value, denom_s)

    gates_s /= denom_s
    gates_s = tl.reshape(gates_s, (BLOCK_K, 1))
    gates_all_data = tl.broadcast_to(gates_s, (BLOCK_K, EXPERTS)) * masks_data.to(gates_s.dtype)

    # test
    gates_all_ptrs = gates_all + offs_k * stride_kse_k + s_pid * stride_kse_s + offs_e
    tl.store(gates_all_ptrs, gates_all_data, mask=offs_k < K)
    locas_s_ptrs = locations_s + offs_k * stride_ks_k + s_pid
    tl.store(locas_s_ptrs, locations_s_data, mask=offs_k < K)

    # locations_s = tl.broadcast_to(tl.reshape(locations_s,(BLOCK_K, 1)), (BLOCK_K, BLOCK_C))
    # one_hot_help = tl.broadcast_to(tl.reshape(tl.arange(0, BLOCK_C), (1,BLOCK_C)), (BLOCK_K, BLOCK_C))
    # loc_sc = tl.where(locations_s == one_hot_help, 1, 0)
    # loc_sc = tl.broadcast_to(tl.reshape(loc_sc,(BLOCK_K, 1, BLOCK_C)), (BLOCK_K, EXPERTS, BLOCK_C))
    # gates_all_data = tl.broadcast_to(tl.reshape(gates_all_data,(BLOCK_K, EXPERTS,1)), (BLOCK_K, EXPERTS, BLOCK_C))

    # combine_weights_data = tl.reshape(tl.sum(gates_all_data * loc_sc, axis=0), (1, EXPERTS, BLOCK_C))
    
    # offs_ksc = s_pid * stride_sec_s + tl.arange(0, EXPERTS)[None,:,None] * stride_sec_e + tl.arange(0, BLOCK_C)[None, None, :] 
   
    # mask_ = tl.arange(0, BLOCK_C)[None, None, :]  < CAPACITY
    # tl.store(combine_weights + offs_ksc, combine_weights_data, mask=mask_)
    # tl.store(dispatch_mask + offs_ksc, tl.where(combine_weights_data > 0, 1, 0), mask=mask_)
   


def call(gates, masks, locations, k, capacity):
    s, e = gates.shape
    stride_se_s, _ = gates.stride()
    gates_all = torch.empty((k, s, e), dtype=gates.dtype, device=gates.device)
    locations_s = torch.empty((k, s), dtype=torch.int64, device=gates.device)
    combine_weights = torch.empty((s, e, capacity), device=gates.device)
    dispatch_mask = torch.empty((s, e, capacity), device=gates.device, dtype=torch.bool)
    stride_sec_s, stride_sec_e, _ = combine_weights.stride()
    stride_kse_k, stride_kse_s, _ = masks.stride()
    min_value = torch.finfo(gates.dtype).eps
    
    _topk_gating_fwd_part3[(s,)](
        gates,
        gates_all,
        locations,
        locations_s,
        masks,
        combine_weights,
        dispatch_mask,
        stride_ks_k=s,
        stride_se_s=stride_se_s,
        stride_kse_k=stride_kse_k,
        stride_kse_s=stride_kse_s,
        stride_sec_s=stride_sec_s, 
        stride_sec_e=stride_sec_e,
        CAPACITY = capacity,
        BLOCK_C = triton.next_power_of_2(capacity),
        min_value = min_value,
        K = k,
        EXPERTS = e,
        BLOCK_K = triton.next_power_of_2(k),
    )

    locations_sc = torch.nn.functional.one_hot(locations_s, num_classes=capacity)
    combine_sec = torch.einsum("kse,ksc->ksec", gates_all, locations_sc)
    combine_weights = torch.sum(combine_sec, dim=0)
    dispatch_mask = combine_weights.bool()

    return combine_weights, dispatch_mask


def bench_fn(gates, masks, locations, k, capacity):
    fn = lambda: call(gates, masks, locations, k, capacity)
    ms = triton.testing.do_bench(fn, warmup=100, rep=100)
    return ms


# register
name = '_topk_gating_fwd_part3'
for dtype in [torch.float16, torch.float32]:
    for device in ['cuda']:
        seqLen, experts = SymVar('seqLen'), SymVar('experts')
        k, capacity = SymVar('k'), SymVar('capacity')
        # we dont' actually allocate tensor
        gates = Tensor((seqLen, experts), dtype=dtype, device=device)
        masks = Tensor((k, seqLen, experts), dtype=torch.int64, device=device)
        locations = Tensor((k, seqLen, experts), dtype=torch.int64, device=device)
        register_dlblas_op(name, None, (gates, masks, locations, torch.SymInt, torch.SymInt), call, bench_fn, _topk_gating_fwd_part3)
