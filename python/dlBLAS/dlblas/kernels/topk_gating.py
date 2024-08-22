from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
import triton
import triton.language as tl

from dlblas.utils import register_dlblas_op, SymVar, Tensor, ChoiceSpace
from dlblas.op_registry import op_registry
import dlblas

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


def topk_fwd_triton(gates: torch.Tensor, masks, locations, capacity):
    s, e = gates.shape
    k = masks.shape[0]
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
    for_bwd = (gate_ks, indices_ks, denom_s, clamp_denom_s)
    return token_rearranged_ec_idx, token_exp_weights, expert_select_token_idx, for_bwd


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


def topk_bwd_triton(gates_se, ce, masks_kse, gates_ks, indices_ks, denom_s, clamp_denom_s, grad_l_aux, grad_token_exp_weights):
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


class TopKGatingFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.Any, logits: torch.Tensor, k: int, capacity_factor: float = 1.0, min_capacity: int = 2, higher_precision: bool = False):
        # compute the capacity
        capacity = _capacity(logits, torch.tensor(capacity_factor * k), torch.tensor(min_capacity)).item()
        gates, masks = dlblas._topk_gating_fwd_part1(logits, k)
        locations, res, ce = dlblas._topk_gating_fwd_part2(gates, masks, k)
        l_aux = torch.mean(res)
        if higher_precision:
            token_rearranged_ec_idx, token_exp_weights, expert_select_token_idx, for_bwd = topk_fwd_triton(gates, masks, locations, capacity)
            ctx.save_for_backward(gates, ce, masks, *for_bwd)
            return l_aux, token_rearranged_ec_idx, token_exp_weights, expert_select_token_idx
        else:
            part3_res = dlblas._topk_gating_fwd_part3(gates, masks, locations, k, capacity, True)
            l_aux = torch.mean(res)
            ctx.save_for_backward(locations, masks, gates, ce)
            return l_aux, *part3_res


    @staticmethod
    def backward(ctx: torch.Any, *grad_outputs: torch.Any) -> torch.Any:
     
        if len(ctx.saved_tensors) == 4:
            grad_l_aux = grad_outputs[0]
            locations, masks, gates, ce = ctx.saved_tensors
            grad_logits = dlblas._topk_gating_bwd(grad_l_aux, locations, masks, gates, ce)
            return grad_logits, None, None, None, None
        else:
            grad_l_aux = grad_outputs[0]
            grad_token_exp_weights = grad_outputs[1]
            getes, ce, masks, gate_ks, indices_s, denom_s, clamp_denom_s = ctx.saved_tensors
            res = topk_bwd_triton(getes, ce, masks, gate_ks, indices_s, denom_s, clamp_denom_s, grad_l_aux, grad_token_exp_weights)
            return res, None, None, None, None
    

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

