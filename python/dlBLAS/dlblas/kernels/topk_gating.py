from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
import triton
import triton.language as tl

from dlblas.utils import register_dlblas_op, SymVar, Tensor, ChoiceSpace
from dlblas.op_registry import op_registry
import dlblas
# from topk_gating_fwd_part1 import _topk_gating_kernel_part1
# from topk_gating_bwd import fused_bwd

# from gshard_moe import _capacity, gumbel_rsample
gumbel_map: Dict[torch.device, Callable] = {}

def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity


@triton.jit
def _topk_gating_fwd_part3(
    gates,
    locations,
    masks,
    invers_k_ptr,
    indices_ks,
    denom_s, 
    clamp_denom_s,
    token_rearranged_ec_idx_ptr,
    gate_ks_ptr,
    token_sel_exp_int_mask_ptr,
    stride_ks_k,
    stride_se_s,
    stride_kse_k,
    stride_kse_s,
    CAPACITY: tl.constexpr,
    min_value: tl.constexpr,
    K: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_k = tl.arange(0, BLOCK_K)
    s_pid = tl.program_id(axis=0)
    offs_e = tl.arange(0, EXPERTS)[None,:]
    

    locations_ptrs = locations + offs_k[:,None] * stride_kse_k + s_pid * stride_kse_s + offs_e
    locations_data = tl.load(locations_ptrs, mask=offs_k[:,None] < K)
    masks_ptrs = masks + offs_k[:,None] * stride_kse_k + s_pid * stride_kse_s + offs_e
    masks_data = tl.load(masks_ptrs, mask=offs_k[:,None] < K)
    masks_data *= tl.where(locations_data < CAPACITY, 1, 0)

    # for bwd
    tl.store(masks_ptrs, masks_data, mask=offs_k[:,None] < K)

    locations_data *= masks_data
    locations_ks_data = tl.sum(locations_data, axis=1)
    
    gates_ptrs = gates + s_pid * stride_se_s + offs_e
   
    
    gates_data = tl.load(gates_ptrs)
    #gate_s = torch.einsum("se,kse->ks", gates, mask_float)
    multi = tl.broadcast_to(gates_data, (BLOCK_K, EXPERTS)) * masks_data

    # gates_s = tl.sum(multi, axis=1)
    gates_ks, indices_ks_data = tl.max(multi, axis=1, return_indices=True)
    indices_ks_ptrs = indices_ks + offs_k*stride_ks_k + s_pid
    tl.store(indices_ks_ptrs, indices_ks_data, mask=offs_k < K)
    denom_s_data = tl.sum(gates_ks, axis=0)
    denom_s_ptrs = denom_s + s_pid
    tl.store(denom_s_ptrs, denom_s_data)
    # torch.clamp
    clamp_denom_s_data = tl.where(denom_s_data < min_value, min_value, denom_s_data)
    tl.store(clamp_denom_s + s_pid, clamp_denom_s_data)
    # ks
    gates_ks /= clamp_denom_s_data
    gate_ks_ptrs = gate_ks_ptr + offs_k * stride_ks_k + s_pid
    tl.store(gate_ks_ptrs, gates_ks, mask=offs_k < K)

   
    token_rearranged_ec_idx_data = indices_ks_data.to(tl.int32) * CAPACITY + locations_ks_data.to(tl.int32)
    token_rearranged_ec_idx_ptrs = token_rearranged_ec_idx_ptr + offs_k * stride_ks_k + s_pid
    tl.store(token_rearranged_ec_idx_ptrs, token_rearranged_ec_idx_data, mask=offs_k < K)

    # se
    invers_k_data = tl.load(invers_k_ptr + offs_k, mask=offs_k < K)
    token_sel_exp_int_mask_data = masks_data * tl.broadcast_to(tl.reshape(invers_k_data,(BLOCK_K,1)), (BLOCK_K, EXPERTS))
    token_sel_exp_int_mask_data = tl.sum(token_sel_exp_int_mask_data, axis=0)
    token_sel_exp_int_mask_ptrs = token_sel_exp_int_mask_ptr + s_pid * stride_se_s + tl.arange(0, EXPERTS)
    tl.store(token_sel_exp_int_mask_ptrs, token_sel_exp_int_mask_data)
    
    


def topk_fwd_triton(logits: torch.Tensor, k: int, capacity_factor: float = 1.0, min_capacity: int = 2):
    capacity = _capacity(logits, torch.tensor(capacity_factor * k), torch.tensor(min_capacity)).item()
    gates, masks = dlblas._topk_gating_fwd_part1(logits, k)
    locations, exp_counts, res, ce = dlblas._topk_gating_fwd_part2(gates, masks, k)
    l_aux = torch.mean(res)
    
    s, e = gates.shape
    invers_k = torch.arange(k, 0, -1, device=masks.device)

    token_rearranged_ec_idx = torch.empty((k, s), dtype=torch.int32, device=gates.device)
    indices_ks = torch.empty((k, s), dtype=torch.int64, device=gates.device)
    gate_ks = torch.empty((k, s), dtype=gates.dtype, device=gates.device)
    denom_s = torch.empty((s,), dtype=gates.dtype, device=gates.device)
    clamp_denom_s = torch.empty((s,), dtype=gates.dtype, device=gates.device)
    token_sel_exp_int_mask = torch.empty((s, e), dtype=torch.int64, device=gates.device)
    stride_se_s, _ = gates.stride()
    stride_ks_k, _ = gate_ks.stride()
    stride_kse_k, stride_kse_s, _ = masks.stride()
    _topk_gating_fwd_part3[(s,)](
        gates,
        locations,
        masks,
        invers_k,
        indices_ks,
        denom_s, 
        clamp_denom_s,
        token_rearranged_ec_idx,
        gate_ks,
        token_sel_exp_int_mask,
        stride_ks_k,
        stride_se_s,
        stride_kse_k,
        stride_kse_s,
        CAPACITY = capacity,
        min_value = torch.finfo(gates.dtype).eps,
        K = k,
        EXPERTS = e,
        BLOCK_K = triton.next_power_of_2(k),
    )
    expert_sel_top_c_token_idx = torch.topk(
            token_sel_exp_int_mask, k=capacity, dim=0, sorted=True
    )[1]
    expert_select_token_idx = expert_sel_top_c_token_idx.t().reshape(e * capacity)
    token_rearranged_ec_idx = token_rearranged_ec_idx.reshape(-1)
    token_exp_weights = gate_ks.reshape(-1)
    for_bwd = (gates, ce, masks, gate_ks, indices_ks, denom_s, clamp_denom_s)
    return l_aux, token_rearranged_ec_idx, token_exp_weights, expert_select_token_idx, for_bwd


def topk_forward(logits, primals_2, primals_3, primals_4, primals_5):
    num_experts = int(logits.shape[1])
    getes = torch.ops.aten._softmax.default(logits, 1, False);  
    _tensor_constant0 = torch.tensor([6], dtype=torch.int64, device=logits.device)
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    mul = torch.ops.aten.mul.Tensor(lift_fresh_copy, 64.0);  lift_fresh_copy = None
    ceil = torch.ops.aten.ceil.default(mul);  mul = None
    capacity = torch.ops.aten._to_copy.default(ceil, dtype = torch.int64);  ceil = None
    topk = torch.ops.aten.topk.default(getes, 6, 1)
    getitem_1 = topk[1];  topk = None
    indices_s = torch.ops.aten.t.default(getitem_1);  getitem_1 = None
    masks = torch.nn.functional.one_hot(indices_s.reshape(-1), num_classes=num_experts)
    # clone = torch.ops.aten.clone.default(t, memory_format = torch.contiguous_format);  t = None
    # _unsafe_view = torch.ops.aten._unsafe_view.default(clone, [24576]);  clone = None
    # zeros = torch.ops.aten.zeros.default([24576, 64], dtype = torch.int64, layout = torch.strided, device = device_)
    # unsqueeze = torch.ops.aten.unsqueeze.default(_unsafe_view, -1);  _unsafe_view = None
    # scatter = torch.ops.aten.scatter.value(zeros, -1, unsqueeze, 1);  zeros = unsqueeze = None
    locations = torch.ops.aten.cumsum.default(masks, 0) - 1
   
    locations = torch.ops.aten.view.default(locations, [-1, 4096, 64]); 
    me = torch.ops.aten.mean.dim(getes, [0])
    masks = torch.ops.aten.view.default(masks, [-1, 4096, 64])
    # select_1 = torch.ops.aten.select.int(masks, 0, 0);  
    # _to_copy_1 = torch.ops.aten._to_copy.default(select_1, dtype = torch.float32, layout = torch.strided, device = device_);  select_1 = None
    ce = torch.mean(masks[0].type_as(logits), dim=0)
    # mean_1 = torch.ops.aten.mean.dim(_to_copy_1, [0]);  _to_copy_1 = None
    # mul_1 = torch.ops.aten.mul.Tensor(me, ce);  
    l_aux = torch.ops.aten.mean.default(me * ce) * num_experts * num_experts
    # mul_2 = torch.ops.aten.mul.Tensor(mean_2, 64);  mean_2 = None
    # mul_3 = torch.ops.aten.mul.Tensor(mul_2, 64);  mul_2 = None
    lt_1 = torch.ops.aten.lt.Tensor(locations, capacity)
    
    masks = torch.ops.aten.mul.Tensor(masks, lt_1); 

    # view_4 = torch.ops.aten.view.default(mul_4, [24576, 64]);  mul_4 = None
    # view_5 = torch.ops.aten.view.default(view_4, [-1, 4096, 64]);  view_4 = None
    mul_5 = torch.ops.aten.mul.Tensor(locations, masks);  
    sum_1 = torch.ops.aten.sum.dim_IntList(mul_5, [2]);  mul_5 = None
    # _to_copy_2 = torch.ops.aten._to_copy.default(view_5, dtype = torch.float32, layout = torch.strided, device = device_)
    mul_6 = torch.ops.aten.mul.Tensor(getes, masks.type_as(logits))
    max_1 = torch.ops.aten.max.dim(mul_6, 2);  mul_6 = None
    gate_s = max_1[0]
    indices_s = max_1[1];  max_1 = None
    denom_s = torch.ops.aten.sum.dim_IntList(gate_s, [0])
    clamp_denom_s = torch.ops.aten.clamp.default(denom_s, torch.finfo(denom_s.dtype).eps)
    div = torch.ops.aten.div.Tensor(gate_s, clamp_denom_s)
    _to_copy_3 = torch.ops.aten._to_copy.default(indices_s, dtype = torch.int32)
    mul_7 = torch.ops.aten.mul.Tensor(_to_copy_3, capacity);  _to_copy_3 = _to_copy = None
    _to_copy_4 = torch.ops.aten._to_copy.default(sum_1, dtype = torch.int32);  sum_1 = None
    add = torch.ops.aten.add.Tensor(mul_7, _to_copy_4);  mul_7 = _to_copy_4 = None
    arange = torch.ops.aten.arange.start_step(6, 0, -1, device = logits.device, pin_memory = False)
    view_6 = torch.ops.aten.view.default(arange, [6, 1, 1]);  arange = None
    mul_8 = torch.ops.aten.mul.Tensor(masks, view_6);  view_5 = view_6 = None
    sum_3 = torch.ops.aten.sum.dim_IntList(mul_8, [0]);  mul_8 = None
    topk_1 = torch.ops.aten.topk.default(sum_3, 384, 0);  sum_3 = None
    getitem_5 = topk_1[1];  topk_1 = None
    t_1 = torch.ops.aten.t.default(getitem_5);  getitem_5 = None
    clone_2 = torch.ops.aten.clone.default(t_1, memory_format = torch.contiguous_format);  t_1 = None
    _unsafe_view_1 = torch.ops.aten._unsafe_view.default(clone_2, [24576]);  clone_2 = None
    view_7 = torch.ops.aten.view.default(add, [-1]);  add = None
    view_9 = torch.ops.aten.view.default(div, [-1]);  div = None
    return [l_aux, view_7, view_9, _unsafe_view_1, getes, ce, masks, gate_s, indices_s, denom_s, clamp_denom_s]


@triton.autotune(
    configs = [
        triton.Config({'BLOCK_S': BS}, num_stages=s, num_warps=w) \
        for BS in [8, 16] \
        for s in [1] \
        for w in [1] \
    ],
    key=['k', 's', 'e'],
)
@triton.jit
def _topk_gating_bwd_kernel_0(
    gates_ks, denom_s, clamp_denom_s,
    grad_token_exp_weights_ks,
    add_1_ks,
    stride_ks_k,
    min_value: tl.constexpr,
    k:tl.constexpr, s: tl.constexpr, e: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_S: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs_k = tl.arange(0, BLOCK_K)
    offs_s = pid * BLOCK_S  + tl.arange(0, BLOCK_S)
    offs_e = tl.arange(0, e)
    grad_token_exp_weights_ks_ptrs = grad_token_exp_weights_ks + offs_k[:,None] * stride_ks_k + offs_s[None,:]
    grad_token_exp_weights_ks_data = tl.load(grad_token_exp_weights_ks_ptrs, mask=offs_k[:,None] < k and offs_s[None,:] < s)
    gates_ks_ptrs = gates_ks + offs_k[:,None] * stride_ks_k + offs_s[None,:]
    gates_ks_data = tl.load(gates_ks_ptrs, mask=offs_k[:,None] < k and offs_s[None,:] < s)
    denom_s_data = tl.load(denom_s + offs_s, mask=offs_s < s)
    clamp_denom_s_data = tl.load(clamp_denom_s + offs_s, mask=offs_s < s)
    clamp_denom_ks_data = tl.broadcast_to(tl.expand_dims(clamp_denom_s_data, axis=0), (BLOCK_K, BLOCK_S))
    div_1 = gates_ks_data / clamp_denom_ks_data
    div_2 = div_1 / clamp_denom_ks_data
    mul_10 = (-grad_token_exp_weights_ks_data) * div_2
    div_3 = grad_token_exp_weights_ks_data / clamp_denom_ks_data
    sum_4 = tl.sum(mul_10, axis=0)
    sum_4 = tl.where(denom_s_data >= min_value, sum_4, 0.0)
    add_1 = div_3 + tl.broadcast_to(tl.expand_dims(sum_4, axis=0), (BLOCK_K, BLOCK_S))
    add_1_ks_ptrs = add_1_ks + offs_k[:,None] * stride_ks_k + offs_s[None,:]
    tl.store(add_1_ks_ptrs, add_1, mask=offs_k[:,None] < k and offs_s[None,:] < s)
    

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_S': BS}, num_stages=s, num_warps=w) \
        for BS in [2, 4, 8] \
        for s in [1] \
        for w in [1] \
    ],
    key=['k', 's', 'e'],
)
@triton.jit
def _topk_gating_bwd_kernel_1(
    ce, masks_kse, grad_l_aux, scatter_kse,
    add_2_se,
    stride_se_s,
    stride_kse_k, stride_kse_s,
    k:tl.constexpr, s: tl.constexpr, e: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_S: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs_k = tl.arange(0, BLOCK_K)
    offs_s = pid * BLOCK_S  + tl.arange(0, BLOCK_S)
    offs_e = tl.arange(0, e)
    ce_ptrs = ce + offs_e
    ce_data = tl.load(ce_ptrs)
    grad_l_aux_data = tl.load(grad_l_aux)
    masks_kse_ptrs = masks_kse + offs_k[:,None,None] * stride_kse_k + offs_s[None,:,None] * stride_kse_s + offs_e[None,None,:]
    masks_kse_data = tl.load(masks_kse_ptrs, mask=offs_k[:,None,None] < k and offs_s[None,:,None] < s)
    scatter_kse_ptrs = scatter_kse + offs_k[:,None,None] * stride_kse_k + offs_s[None,:,None] * stride_kse_s + offs_e[None,None,:]
    scatter_kse_data = tl.load(scatter_kse_ptrs, mask=offs_k[:,None,None] < k and offs_s[None,:,None] < s)
    sum_5 = tl.sum(scatter_kse_data*masks_kse_data, axis=0)
    mul_14 = ce_data * grad_l_aux_data * e / s
    div_5 = tl.broadcast_to(tl.expand_dims(mul_14, axis=0), (BLOCK_S, e))
    add_2 = sum_5 + div_5
    add_2_se_ptrs = add_2_se + offs_s[:, None] * stride_se_s + offs_e[None,:]
    tl.store(add_2_se_ptrs, add_2, mask=offs_s[:, None] < s)


def topk_bwd_triton(gates_se, ce, masks_kse, gates_ks, indices_ks, denom_s, clamp_denom_s, grad_l_aux, tangents_2, grad_token_exp_weights, tangents_4):
    k, s, e = masks_kse.shape
    add_1_ks = torch.empty((k, s), dtype=gates_se.dtype, device=gates_se.device)
    stride_ks_k, _ = add_1_ks.stride()
    
    assert e == triton.next_power_of_2(e)
    
    grid = lambda META: (triton.cdiv(s, META["BLOCK_S"]), )
    _topk_gating_bwd_kernel_0[grid](
        gates_ks, denom_s, clamp_denom_s,
        grad_token_exp_weights,
        add_1_ks,
        stride_ks_k,
        torch.finfo(gates_se.dtype).eps,
        k, s, e,
        BLOCK_K=triton.next_power_of_2(k)
    )
    # print(f"_bwd_kernel.best_config ", _topk_gating_bwd_kernel_0.best_config, flush = True)
    zeros_1 = torch.ops.aten.zeros.default([k, s, e], dtype = gates_se.dtype, layout = torch.strided, device = gates_se.device)
    scatter_kse = torch.ops.aten.scatter.src(zeros_1, 2, indices_ks.view(k,s,1), add_1_ks.view(k,s,1))
    # add_2_se = topk_gating_bwd_1(gates_se, ce, masks_kse, grad_l_aux, scatter_kse)
    
    add_2_se = torch.empty([s,e], dtype=gates_se.dtype, device=gates_se.device)
    stride_se_s, _ = add_2_se.stride()
    stride_kse_k, stride_kse_s, _ = masks_kse.stride()
    grid = lambda META: (triton.cdiv(s, META["BLOCK_S"]), )
    _topk_gating_bwd_kernel_1[grid](
        ce, masks_kse, grad_l_aux, scatter_kse,
        add_2_se,
        stride_se_s,
        stride_kse_k, stride_kse_s,
        k, s, e,
        BLOCK_K=triton.next_power_of_2(k)
    )
    # print(f"_bwd_kernel.best_config ", _topk_gating_bwd_kernel_1.best_config, flush = True)
    return torch.ops.aten._softmax_backward_data.default(add_2_se, gates_se, 1, gates_se.dtype)


def topk_backward(getes, ce, masks, gate_ks, indices_s, denom_s, clamp_denom_s, grad_l_aux, tangents_2, grad_token_exp_weights, tangents_4):
    # clone_1 = torch.ops.aten.clone.default(getitem_2);  getitem_2 = None
    view_10 = torch.ops.aten.view.default(grad_token_exp_weights, [6, 4096]);  
    div_1 = torch.ops.aten.div.Tensor(gate_ks, clamp_denom_s); 
    div_2 = torch.ops.aten.div.Tensor(div_1, clamp_denom_s);  div_1 = None
    neg = torch.ops.aten.neg.default(view_10)
    mul_10 = torch.ops.aten.mul.Tensor(neg, div_2);  neg = div_2 = None
    div_3 = torch.ops.aten.div.Tensor(view_10, clamp_denom_s);  
    sum_4 = torch.ops.aten.sum.dim_IntList(mul_10, [0], True);  mul_10 = None
    view_11 = torch.ops.aten.view.default(sum_4, [4096])
    scalar_tensor = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = getes.device)
    ge = torch.ops.aten.ge.Scalar(denom_s, 1.1920928955078125e-07)
    where = torch.ops.aten.where.self(ge, view_11, scalar_tensor);  ge = view_11 = scalar_tensor = None
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(where, 0);  where = None
    expand = torch.ops.aten.expand.default(unsqueeze_1, [6, 4096]);  unsqueeze_1 = None
    add_1 = torch.ops.aten.add.Tensor(div_3, expand);  div_3 = expand = None
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(add_1, 2);  add_1 = None
    unsqueeze_3 = torch.ops.aten.unsqueeze.default(indices_s, 2);  
    zeros_1 = torch.ops.aten.zeros.default([6, 4096, 64], dtype = torch.float32, layout = torch.strided, device = getes.device)
    scatter_1 = torch.ops.aten.scatter.src(zeros_1, 2, unsqueeze_3, unsqueeze_2);  
    

    mul_11 = torch.ops.aten.mul.Tensor(scatter_1, masks)
    sum_5 = torch.ops.aten.sum.dim_IntList(mul_11, [0], True)
    
    view_12 = torch.ops.aten.view.default(sum_5, [4096, 64]);  sum_5 = None
    mul_12 = torch.ops.aten.mul.Tensor(grad_l_aux, 64)
    mul_13 = torch.ops.aten.mul.Tensor(mul_12, 64);  mul_12 = None
    expand_1 = torch.ops.aten.expand.default(mul_13, [64]);  mul_13 = None
    div_4 = torch.ops.aten.div.Scalar(expand_1, 64);  expand_1 = None
    mul_14 = torch.ops.aten.mul.Tensor(div_4, ce)
    unsqueeze_4 = torch.ops.aten.unsqueeze.default(mul_14, 0);  mul_14 = None
    expand_2 = torch.ops.aten.expand.default(unsqueeze_4, [4096, 64]);  unsqueeze_4 = None
    div_5 = torch.ops.aten.div.Scalar(expand_2, 4096);  expand_2 = None
    add_2 = torch.ops.aten.add.Tensor(view_12, div_5);  view_12 = div_5 = None
    _softmax_backward_data = torch.ops.aten._softmax_backward_data.default(add_2, getes, 1, torch.float32)
    return _softmax_backward_data, None, None, None, None

# device_ = torch.device('cuda:3')
# torch.cuda.set_device(device_)
# logitsa =  torch.randn((4096, 64), device=device_, requires_grad=True)
# mul_3, view_7, view_9, _unsafe_view_1, _softmax, ce_test, masks, getitem_2, getitem_3, denom_s, clamp_denom_s = topk_forward(logitsa, 1,1,1,1)
# g0 = mul_3
# g1 = view_9
class TopKGatingFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.Any, logits: torch.Tensor, k: int, capacity_factor: float = 1.0, min_capacity: int = 2, enable_token_rearrange_opt: bool = False):
        # compute the capacity
        # capacity = _capacity(logits, torch.tensor(capacity_factor * k), torch.tensor(min_capacity)).item()
        # gates, masks = dlblas._topk_gating_fwd_part1(logits, k)
        # locations, exp_counts, res, ce = dlblas._topk_gating_fwd_part2(gates, masks, k)
        # part3_res = dlblas._topk_gating_fwd_part3(gates, masks, locations, k, capacity, enable_token_rearrange_opt)
        # l_aux = torch.mean(res)
        # ctx.save_for_backward(locations, masks, gates, ce)
        # return l_aux, *part3_res
        # test
        l_aux, token_rearranged_ec_idx, token_exp_weights, expert_select_token_idx, for_bwd = topk_fwd_triton(logits, k, capacity_factor, min_capacity)
        ctx.save_for_backward(*for_bwd)
        return l_aux, token_rearranged_ec_idx, token_exp_weights, expert_select_token_idx
        # l_aux_test, view_7, view_9, _unsafe_view_1, getes, ce, masks, gate_s, indices_s, denom_s, clamp_denom_s = topk_forward(logits, 1,1,1,1)
        # assert torch.allclose(l_aux, l_aux_test)
        # assert torch.allclose(token_rearranged_ec_idx)
        # assert torch.allclose()
        # assert torch.allclose()
        # assert torch.allclose()
        # assert torch.allclose()
        # assert torch.allclose()
        # ctx.save_for_backward(getes, ce, masks, gate_s, indices_s, denom_s, clamp_denom_s)
        # return l_aux, view_7.int(), view_9, _unsafe_view_1
 
    

    @staticmethod
    def backward(ctx: torch.Any, *grad_outputs: torch.Any) -> torch.Any:
        # grad_l_aux = grad_outputs[0].item()
        # # grad_combine = grad_outputs[1]
        # locations, masks, gates, ce = ctx.saved_tensors
        # grad_logits = dlblas._topk_gating_bwd(grad_l_aux, locations, masks, gates, ce)
        # return grad_logits, None, None, None, None
        # test
        # 
        grad_l_aux = grad_outputs[0]
        grad_token_exp_weights = grad_outputs[1]
        # g0.backward(mul_3,retain_graph=True)
        # res1 = logitsa.grad
        # logitsa.grad.zero_()
        # g1.backward(view_9, retain_graph=True)
        # res2 = logitsa.grad
        # logitsa.grad.zero_()
        # return res1*res2, None, None, None, None
        getes, ce, masks, gate_ks, indices_s, denom_s, clamp_denom_s = ctx.saved_tensors
        # print(f"_softmax:{_softmax.shape}, {_softmax.dtype}")
        # print(f"ce_test:{ce_test.shape}, {ce_test.dtype}")
        # print(f"masks:{masks.shape}, {masks.dtype}")
        # print(f"getitem_2:{getitem_2.shape}, {getitem_2.dtype}")
        # print(f"getitem_3:{getitem_3.shape}, {getitem_3.dtype}")
        # print(f"denom_s:{denom_s.shape}, {denom_s.dtype}")
        # print(f"clamp_denom_s:{clamp_denom_s.shape}, {clamp_denom_s.dtype}")

        # print(f"mul_3:{mul_3.shape}, {mul_3.dtype}, {mul_3}")
        # print(f"view_9:{view_9.shape}, {view_9.dtype}")
        # res = topk_backward(_softmax, ce_test, masks, getitem_2, getitem_3, denom_s, clamp_denom_s, mul_3, 1, view_9, 1)
        # print(f"res[0]:{res[0].shape}, {res[0].dtype}")
        # fn = lambda: topk_backward(_softmax, ce_test, masks, getitem_2, getitem_3, denom_s, clamp_denom_s, mul_3, 1, view_9, 1)
        # ms = triton.testing.do_bench(fn, warmup=100, rep=100)
        # print(ms)
        # quit()
        # return None, None, None, None, None
        a1 = topk_bwd_triton(getes, ce, masks, gate_ks, indices_s, denom_s, clamp_denom_s, grad_l_aux, None, grad_token_exp_weights, None)
        return a1, None, None, None, None
        a2 = topk_backward(getes, ce, masks, gate_ks, indices_s, denom_s, clamp_denom_s, grad_l_aux, 1, grad_token_exp_weights, 1)
        assert torch.allclose(a1, a2[0])
        fn = lambda:topk_bwd_triton(getes, ce, masks, gate_ks, indices_s, denom_s, clamp_denom_s, grad_l_aux, None, grad_token_exp_weights, None)
        ms = triton.testing.do_bench(fn, warmup=100, rep=100)
        print(ms)
        fn = lambda:topk_backward(getes, ce, masks, gate_ks, indices_s, denom_s, clamp_denom_s, grad_l_aux, 1, grad_token_exp_weights, 1)
        ms = triton.testing.do_bench(fn, warmup=100, rep=100)
        print(ms)
        quit()

        return topk_backward(getes, ce, masks, gate_ks, indices_s, denom_s, clamp_denom_s, grad_l_aux, 1, grad_token_exp_weights, 1)
        
    

def call(logits: torch.Tensor, k: int, capacity_factor: float = 1.0, min_capacity: int = 2, enable_token_rearrange_opt: bool = False):
    return TopKGatingFunc.apply(logits, k, capacity_factor, min_capacity, enable_token_rearrange_opt)


def bench_fn(logits: torch.Tensor, k: int, capacity_factor: float = 1.0, min_capacity: int = 2, enable_token_rearrange_opt: bool = False):
    fn = lambda: call(logits, k, capacity_factor, min_capacity, enable_token_rearrange_opt)
    ms = triton.testing.do_bench(fn, warmup=100, rep=100)
    return ms


# register
name = 'topk_gating'
for dtype in [torch.float16, torch.float32]:
    for device in ['cuda']:
        seqLen, experts = SymVar('seqLen'), SymVar('experts')
        k = SymVar('k')
        capacity_factor = SymVar('capacity_factor')
        min_capacity = SymVar('min_capacity')
        # we dont' actually allocate tensor
        logits = Tensor((seqLen, experts), dtype=dtype, device=device)
        # space = ChoiceSpace([])
        register_dlblas_op(name, None, (logits, torch.SymInt, torch.SymFloat, torch.SymInt, torch.SymBool), call, bench_fn, call)

