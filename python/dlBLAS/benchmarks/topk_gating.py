from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
import triton
import time

import dlblas
from dlblas.utils.op_collector import InnerCompilerOpCollectorContext 

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

def fused_topkgating(
    logits: Tensor,
    k: int,
    capacity_factor: float = 1.0, 
    min_capacity: int = 2
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements TopKGating on logits."""
    # everything is in fp32 in this function
   
    gates = F.softmax(logits, dim=1)
    
    num_experts = int(gates.shape[1])

    capacity = _capacity(gates, torch.tensor(capacity_factor * k), torch.tensor(min_capacity))

    # Create a mask by top-k experts
    indices_s = torch.topk(gates, k, dim=1).indices
    indices_s = indices_s.permute(1, 0).reshape(-1)
    masks = F.one_hot(indices_s, num_classes=num_experts)
    
    # Compute locations in capacity buffer
    locations = torch.cumsum(masks, dim=0) - 1
    # reshape (s,e) to (k,s,e)
    masks = masks.reshape(-1, gates.shape[0], num_experts)
    locations = locations.reshape(-1, gates.shape[0], num_experts)

    # gating decisions
    exp_counts = torch.sum(masks[0], dim=0).detach().to("cpu")

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(masks[0].type_as(logits), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts
    # Remove locations outside capacity from mask
    masks *= torch.lt(locations, capacity)
    # Store the capacity location for each token
    locations_s = torch.sum(locations * masks, dim=2)
    # Normalize gate probabilities
    mask_float = masks.type_as(logits)
    gate_s = torch.einsum("se,kse->ks", gates, mask_float)
    denom_s = torch.sum(gate_s, dim=0)
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gate_s /= denom_s
    # Calculate combine_weights and dispatch_mask
    gate_all = torch.einsum("ks,kse->kse", gate_s, mask_float)
    
    # ---- test begin ----
    # k, s, e, c= locations_s.shape[0], locations_s.shape[1], logits.shape[1], capacity
    # combine_weights_test = torch.zeros((s, e, c), device=logits.device, dtype=logits.dtype)
    # for idx_k in range(k):
    #     for idx_s in range(s):
    #         combine_weights_test[idx_s,:,locations_s[idx_k][idx_s]] += gate_all[idx_k, idx_s,:]
    # dispatch_mask = combine_weights_test.bool()
    # return l_aux, combine_weights_test, dispatch_mask, exp_counts
    # --replace---
    locations_sc = F.one_hot(locations_s, num_classes=capacity).type_as(logits)
    combine_sec = torch.einsum("kse,ksc->ksec", gate_all, locations_sc)
    combine_weights = torch.sum(combine_sec, dim=0)
    
    # assert torch.allclose(combine_weights, combine_weights_test)
    # --- test end ----
    
    # torch.cuda.synchronize(logits.device)
    # t0 = time.time()

    dispatch_mask = combine_weights.bool()

    # torch.cuda.synchronize(logits.device)
    # print(f"torch time:{(time.time() - t0) * 1000.0}")
    
    # return l_aux, masks, locations_s, exp_counts
    return l_aux, combine_weights, dispatch_mask, exp_counts


# def topk_forward(logits, primals_2, primals_3, primals_4, primals_5):
#     num_experts = int(logits.shape[1])
#     getes = torch.ops.aten._softmax.default(logits, 1, False);  
#     _tensor_constant0 = torch.tensor([6], dtype=torch.int64, device=device_)
#     lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
#     mul = torch.ops.aten.mul.Tensor(lift_fresh_copy, 64.0);  lift_fresh_copy = None
#     ceil = torch.ops.aten.ceil.default(mul);  mul = None
#     capacity = torch.ops.aten._to_copy.default(ceil, dtype = torch.int64);  ceil = None
#     print(f"torch test capacity:{capacity}")
#     topk = torch.ops.aten.topk.default(getes, 6, 1)
#     getitem_1 = topk[1];  topk = None
#     indices_s = torch.ops.aten.t.default(getitem_1);  getitem_1 = None
#     masks = F.one_hot(indices_s.reshape(-1), num_classes=num_experts)
#     # clone = torch.ops.aten.clone.default(t, memory_format = torch.contiguous_format);  t = None
#     # _unsafe_view = torch.ops.aten._unsafe_view.default(clone, [24576]);  clone = None
#     # zeros = torch.ops.aten.zeros.default([24576, 64], dtype = torch.int64, layout = torch.strided, device = device_)
#     # unsqueeze = torch.ops.aten.unsqueeze.default(_unsafe_view, -1);  _unsafe_view = None
#     # scatter = torch.ops.aten.scatter.value(zeros, -1, unsqueeze, 1);  zeros = unsqueeze = None
#     locations = torch.ops.aten.cumsum.default(masks, 0) - 1
   
#     locations = torch.ops.aten.view.default(locations, [-1, 4096, 64]); 
#     me = torch.ops.aten.mean.dim(getes, [0])
#     masks = torch.ops.aten.view.default(masks, [-1, 4096, 64])
#     # select_1 = torch.ops.aten.select.int(masks, 0, 0);  
#     # _to_copy_1 = torch.ops.aten._to_copy.default(select_1, dtype = torch.float32, layout = torch.strided, device = device_);  select_1 = None
#     ce = torch.mean(masks[0].type_as(logits), dim=0)
#     # mean_1 = torch.ops.aten.mean.dim(_to_copy_1, [0]);  _to_copy_1 = None
#     # mul_1 = torch.ops.aten.mul.Tensor(me, ce);  
#     l_aux = torch.ops.aten.mean.default(me * ce) * num_experts * num_experts
#     # mul_2 = torch.ops.aten.mul.Tensor(mean_2, 64);  mean_2 = None
#     # mul_3 = torch.ops.aten.mul.Tensor(mul_2, 64);  mul_2 = None
#     lt_1 = torch.ops.aten.lt.Tensor(locations, capacity)
    
#     masks = torch.ops.aten.mul.Tensor(masks, lt_1); 

#     # view_4 = torch.ops.aten.view.default(mul_4, [24576, 64]);  mul_4 = None
#     # view_5 = torch.ops.aten.view.default(view_4, [-1, 4096, 64]);  view_4 = None
#     mul_5 = torch.ops.aten.mul.Tensor(locations, masks);  
#     sum_1 = torch.ops.aten.sum.dim_IntList(mul_5, [2]);  mul_5 = None
#     # _to_copy_2 = torch.ops.aten._to_copy.default(view_5, dtype = torch.float32, layout = torch.strided, device = device_)
#     mul_6 = torch.ops.aten.mul.Tensor(getes, masks.type_as(logits))
#     max_1 = torch.ops.aten.max.dim(mul_6, 2);  mul_6 = None
#     gate_s = max_1[0]
#     indices_s = max_1[1];  max_1 = None
#     denom_s = torch.ops.aten.sum.dim_IntList(gate_s, [0])
#     clamp_denom_s = torch.ops.aten.clamp.default(denom_s, torch.finfo(denom_s.dtype).eps)
#     div = torch.ops.aten.div.Tensor(gate_s, clamp_denom_s)
#     _to_copy_3 = torch.ops.aten._to_copy.default(indices_s, dtype = torch.int32)
#     mul_7 = torch.ops.aten.mul.Tensor(_to_copy_3, capacity);  _to_copy_3 = _to_copy = None
#     _to_copy_4 = torch.ops.aten._to_copy.default(sum_1, dtype = torch.int32);  sum_1 = None
#     add = torch.ops.aten.add.Tensor(mul_7, _to_copy_4);  mul_7 = _to_copy_4 = None
#     arange = torch.ops.aten.arange.start_step(6, 0, -1, device = device_, pin_memory = False)
#     view_6 = torch.ops.aten.view.default(arange, [6, 1, 1]);  arange = None
#     mul_8 = torch.ops.aten.mul.Tensor(masks, view_6);  view_5 = view_6 = None
#     sum_3 = torch.ops.aten.sum.dim_IntList(mul_8, [0]);  mul_8 = None
#     topk_1 = torch.ops.aten.topk.default(sum_3, 384, 0);  sum_3 = None
#     getitem_5 = topk_1[1];  topk_1 = None
#     t_1 = torch.ops.aten.t.default(getitem_5);  getitem_5 = None
#     clone_2 = torch.ops.aten.clone.default(t_1, memory_format = torch.contiguous_format);  t_1 = None
#     _unsafe_view_1 = torch.ops.aten._unsafe_view.default(clone_2, [24576]);  clone_2 = None
#     view_7 = torch.ops.aten.view.default(add, [-1]);  add = None
#     view_9 = torch.ops.aten.view.default(div, [-1]);  div = None
#     return [l_aux, view_7, view_9, _unsafe_view_1, getes, ce, masks, gate_s, indices_s, denom_s, clamp_denom_s, locations]

def topk_forward(primals_1, primals_2, primals_3, primals_4, primals_5):
    _softmax = torch.ops.aten._softmax.default(primals_1, 1, False);  primals_1 = None
    _tensor_constant0 = torch.tensor([6], dtype=torch.int64, device=device_)
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    mul = torch.ops.aten.mul.Tensor(lift_fresh_copy, 64.0);  lift_fresh_copy = None
    ceil = torch.ops.aten.ceil.default(mul);  mul = None
    _to_copy = torch.ops.aten._to_copy.default(ceil, dtype = torch.int64);  ceil = None
    topk = torch.ops.aten.topk.default(_softmax, 6, 1)
    getitem_1 = topk[1];  topk = None
    t = torch.ops.aten.t.default(getitem_1);  getitem_1 = None
    clone = torch.ops.aten.clone.default(t, memory_format = torch.contiguous_format);  t = None
    _unsafe_view = torch.ops.aten._unsafe_view.default(clone, [24576]);  clone = None
    zeros = torch.ops.aten.zeros.default([24576, 64], dtype = torch.int64, layout = torch.strided, device = device_)
    unsqueeze = torch.ops.aten.unsqueeze.default(_unsafe_view, -1);  _unsafe_view = None
    scatter = torch.ops.aten.scatter.value(zeros, -1, unsqueeze, 1);  zeros = unsqueeze = None
    cumsum = torch.ops.aten.cumsum.default(scatter, 0)
    sub = torch.ops.aten.sub.Tensor(cumsum, 1);  cumsum = None
    view_1 = torch.ops.aten.view.default(sub, [-1, 4096, 64]);  sub = None
    mean = torch.ops.aten.mean.dim(_softmax, [0])
    view_2 = torch.ops.aten.view.default(scatter, [-1, 4096, 64])
    select_1 = torch.ops.aten.select.int(view_2, 0, 0);  view_2 = None
    _to_copy_1 = torch.ops.aten._to_copy.default(select_1, dtype = torch.float32, layout = torch.strided, device = device_);  select_1 = None
    mean_1 = torch.ops.aten.mean.dim(_to_copy_1, [0]);  _to_copy_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(mean, mean_1);  mean = None
    mean_2 = torch.ops.aten.mean.default(mul_1);  mul_1 = None
    mul_2 = torch.ops.aten.mul.Tensor(mean_2, 64);  mean_2 = None
    mul_3 = torch.ops.aten.mul.Tensor(mul_2, 64);  mul_2 = None
    lt_1 = torch.ops.aten.lt.Tensor(view_1, _to_copy)
    view_3 = torch.ops.aten.view.default(scatter, [-1, 4096, 64]);  scatter = None
    mul_4 = torch.ops.aten.mul.Tensor(view_3, lt_1);  view_3 = lt_1 = None
    view_4 = torch.ops.aten.view.default(mul_4, [24576, 64]);  mul_4 = None
    view_5 = torch.ops.aten.view.default(view_4, [-1, 4096, 64]);  view_4 = None
    mul_5 = torch.ops.aten.mul.Tensor(view_1, view_5);  view_1 = None
    sum_1 = torch.ops.aten.sum.dim_IntList(mul_5, [2]);  mul_5 = None
    _to_copy_2 = torch.ops.aten._to_copy.default(view_5, dtype = torch.float32, layout = torch.strided, device = device_)
    mul_6 = torch.ops.aten.mul.Tensor(_softmax, _to_copy_2)
    max_1 = torch.ops.aten.max.dim(mul_6, 2);  mul_6 = None
    getitem_2 = max_1[0]
    getitem_3 = max_1[1];  max_1 = None
    sum_2 = torch.ops.aten.sum.dim_IntList(getitem_2, [0])
    clamp = torch.ops.aten.clamp.default(sum_2, 1.1920928955078125e-07)
    div = torch.ops.aten.div.Tensor(getitem_2, clamp)
    _to_copy_3 = torch.ops.aten._to_copy.default(getitem_3, dtype = torch.int32)
    mul_7 = torch.ops.aten.mul.Tensor(_to_copy_3, _to_copy);  _to_copy_3 = _to_copy = None
    _to_copy_4 = torch.ops.aten._to_copy.default(sum_1, dtype = torch.int32);  sum_1 = None
    add = torch.ops.aten.add.Tensor(mul_7, _to_copy_4);  mul_7 = _to_copy_4 = None
    arange = torch.ops.aten.arange.start_step(6, 0, -1, device = device_, pin_memory = False)
    view_6 = torch.ops.aten.view.default(arange, [6, 1, 1]);  arange = None
    mul_8 = torch.ops.aten.mul.Tensor(view_5, view_6);  view_5 = view_6 = None
    sum_3 = torch.ops.aten.sum.dim_IntList(mul_8, [0]);  mul_8 = None
    topk_1 = torch.ops.aten.topk.default(sum_3, 384, 0);  sum_3 = None
    getitem_5 = topk_1[1];  topk_1 = None
    t_1 = torch.ops.aten.t.default(getitem_5);  getitem_5 = None
    clone_2 = torch.ops.aten.clone.default(t_1, memory_format = torch.contiguous_format);  t_1 = None
    _unsafe_view_1 = torch.ops.aten._unsafe_view.default(clone_2, [24576]);  clone_2 = None
    view_7 = torch.ops.aten.view.default(add, [-1]);  add = None
    view_9 = torch.ops.aten.view.default(div, [-1]);  div = None
    return [mul_3, view_7, view_9, _unsafe_view_1, _softmax, mean_1, _to_copy_2, getitem_2, getitem_3, sum_2, clamp]
    
    
def topk_backward(_softmax, mean_1, _to_copy_2, getitem_2, getitem_3, sum_2, clamp, tangents_1, tangents_2, tangents_3, tangents_4):
    clone_1 = torch.ops.aten.clone.default(getitem_2);  getitem_2 = None
    view_10 = torch.ops.aten.view.default(tangents_3, [6, 4096]);  tangents_3 = None
    div_1 = torch.ops.aten.div.Tensor(clone_1, clamp);  clone_1 = None
    div_2 = torch.ops.aten.div.Tensor(div_1, clamp);  div_1 = None
    neg = torch.ops.aten.neg.default(view_10)
    mul_10 = torch.ops.aten.mul.Tensor(neg, div_2);  neg = div_2 = None
    div_3 = torch.ops.aten.div.Tensor(view_10, clamp);  view_10 = clamp = None
    sum_4 = torch.ops.aten.sum.dim_IntList(mul_10, [0], True);  mul_10 = None
    view_11 = torch.ops.aten.view.default(sum_4, [4096]);  sum_4 = None
    scalar_tensor = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device_)
    ge = torch.ops.aten.ge.Scalar(sum_2, 1.1920928955078125e-07);  sum_2 = None
    where = torch.ops.aten.where.self(ge, view_11, scalar_tensor);  ge = view_11 = scalar_tensor = None
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(where, 0);  where = None
    expand = torch.ops.aten.expand.default(unsqueeze_1, [6, 4096]);  unsqueeze_1 = None
    add_1 = torch.ops.aten.add.Tensor(div_3, expand);  div_3 = expand = None
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(add_1, 2);  add_1 = None
    unsqueeze_3 = torch.ops.aten.unsqueeze.default(getitem_3, 2);  getitem_3 = None
    zeros_1 = torch.ops.aten.zeros.default([6, 4096, 64], dtype = torch.float32, layout = torch.strided, device = device_)
    scatter_1 = torch.ops.aten.scatter.src(zeros_1, 2, unsqueeze_3, unsqueeze_2);  zeros_1 = unsqueeze_3 = unsqueeze_2 = None
    mul_11 = torch.ops.aten.mul.Tensor(scatter_1, _to_copy_2);  scatter_1 = _to_copy_2 = None
    sum_5 = torch.ops.aten.sum.dim_IntList(mul_11, [0], True);  mul_11 = None
    view_12 = torch.ops.aten.view.default(sum_5, [4096, 64]);  sum_5 = None
    mul_12 = torch.ops.aten.mul.Tensor(tangents_1, 64);  tangents_1 = None
    mul_13 = torch.ops.aten.mul.Tensor(mul_12, 64);  mul_12 = None
    expand_1 = torch.ops.aten.expand.default(mul_13, [64]);  mul_13 = None
    div_4 = torch.ops.aten.div.Scalar(expand_1, 64);  expand_1 = None
    mul_14 = torch.ops.aten.mul.Tensor(div_4, mean_1);  div_4 = mean_1 = None
    unsqueeze_4 = torch.ops.aten.unsqueeze.default(mul_14, 0);  mul_14 = None
    expand_2 = torch.ops.aten.expand.default(unsqueeze_4, [4096, 64]);  unsqueeze_4 = None
    div_5 = torch.ops.aten.div.Scalar(expand_2, 4096);  expand_2 = None
    add_2 = torch.ops.aten.add.Tensor(view_12, div_5);  view_12 = div_5 = None
    _softmax_backward_data = torch.ops.aten._softmax_backward_data.default(add_2, _softmax, 1, torch.float32);  add_2 = _softmax = None
    return [_softmax_backward_data, None, None, None, None]


from functorch.compile import aot_module, make_boxed_func, aot_function
from torch._dynamo.backends.common import aot_autograd
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(">>> my_compiler() invoked:")
    # print(">>> FX graph:")
    # gm.graph.print_tabular()
    print(f">>> Code:\n{gm.code}")
    return make_boxed_func(gm.forward)  # return a python callable

my_aot_backend = aot_autograd(fw_compiler=my_compiler)

from tutel import moe as tutel_moe
from collections import namedtuple
GatingTokenRearrangeInfo = namedtuple(
    "GatingTokenRearrangeInfo", ["token_rearranged_ec_idx", "token_exp_weights", "expert_select_token_idx"]
)
# @torch.compile(backend=my_aot_backend)
def fused_topkgating_opt(
    logits: Tensor,
    k: int,
    capacity_factor: float,
    min_capacity: int,
    enable_token_rearrange_opt: bool = False,
    use_tutel: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements TopKGating on logits."""
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)
    num_experts = int(gates.shape[1])

    capacity = _capacity(gates, torch.tensor(capacity_factor * k), torch.tensor(min_capacity))
    # Create a mask by top-k experts
    indices_s = torch.topk(gates, k, dim=1).indices.t()
    masks = F.one_hot(indices_s.reshape(-1), num_classes=num_experts)

    # Compute locations in capacity buffer
    # if use_tutel and TUTEL_INSTALLED:
    locations = tutel_moe.fast_cumsum_sub_one(masks)
    # else:
    # locations = torch.cumsum(masks, dim=0) - 1

    # reshape (s,e) to (k,s,e)
    masks = masks.reshape(-1, gates.shape[0], num_experts)
    locations = locations.reshape(-1, gates.shape[0], num_experts)

    # gating decisions
    # exp_counts = torch.sum(masks[0], dim=0).detach()

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(masks[0].type_as(logits), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts

    # Remove locations outside capacity from mask
    masks *= torch.lt(locations, capacity)

    # Store the capacity location for each token
    locations_s = torch.sum(locations * masks, dim=2)

    # Normalize gate probabilities
    mask_float = masks.type_as(logits)
    # gate_s = einsum("se,kse->ks", gates, mask_float)
    gate_s, indices_s = torch.max(gates * mask_float, dim=2)
    denom_s = torch.sum(gate_s, dim=0)
    # Avoid divide-by-zero
    clamp_denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gate_s /= clamp_denom_s

    # if enable_token_rearrange_opt:
    token_rearranged_ec_idx = indices_s.int() * capacity + locations_s.int()
    # shape：[S, E]->[C, E]->[E, C]->[E*C]
    # import pdb; pdb.set_trace()
    token_sel_exp_int_mask = masks * torch.arange(k, 0, -1, device=masks.device).reshape(k, 1, 1)
    expert_sel_top_c_token_idx = torch.topk(
        torch.sum(token_sel_exp_int_mask, dim=0), k=capacity, dim=0, sorted=True
    )[1]
    expert_select_token_idx = expert_sel_top_c_token_idx.t().reshape(num_experts * capacity)
    token_rearranged_ec_idx = token_rearranged_ec_idx.reshape(-1)
    token_exp_weights = gate_s.reshape(-1)

    top2_gating_token_infos = GatingTokenRearrangeInfo(
        token_rearranged_ec_idx=token_rearranged_ec_idx,
        token_exp_weights=token_exp_weights,
        expert_select_token_idx=expert_select_token_idx,
    )
    return l_aux, top2_gating_token_infos
    # else:
    #     # Calculate combine_weights and dispatch_mask
    #     gate_all = torch.einsum("ks,kse->kse", gate_s, mask_float)
    #     locations_sc = F.one_hot(locations_s, num_classes=capacity).type_as(logits)
    #     combine_sec = torch.einsum("kse,ksc->ksec", gate_all, locations_sc)
    #     combine_weights = torch.sum(combine_sec, dim=0)
    #     dispatch_mask = combine_weights.bool()

    #     return l_aux, combine_weights, dispatch_mask

device_ = torch.device('cuda:3')
torch.cuda.set_device(device_)


def test():
    
    # k, SeqLen, NumberExperts = 4, 16, 8
    k, SeqLen, NumberExperts = 6, 4096, 64
    shape = (SeqLen, NumberExperts)
    logits_torch = torch.randn(shape, device=device_, requires_grad=True)
    capacity_factor: float = 1.0
    min_capacity: int = 2
    enable_token_rearrange_opt = True
    
    with torch.no_grad():
        logits_triton = logits_torch.clone()
        logits_test = logits_torch.clone()

    logits_triton.requires_grad = True
    logits_test.requires_grad = True
    
    model_torch = fused_topkgating_opt
    model_triton = dlblas.topk_gating
    
    output1_torch, out_torch_pack = model_torch(logits_torch, k, capacity_factor, min_capacity, enable_token_rearrange_opt)
    output2_torch = out_torch_pack.token_rearranged_ec_idx
    output3_torch = out_torch_pack.token_exp_weights
    output4_torch = out_torch_pack.expert_select_token_idx
    mul_3, view_7, view_9, _unsafe_view_1, _softmax, ce_test, masks, getitem_2, getitem_3, denom_s, clamp_denom_s = topk_forward(logits_test, 1,1,1,1)
    assert torch.allclose(output1_torch, mul_3)
    assert torch.allclose(output3_torch, view_9)
    back_res = topk_backward(_softmax, ce_test, masks, getitem_2, getitem_3, denom_s, clamp_denom_s, mul_3, 1, view_9, 1)
    
    output1_triton, output2_triton, output3_triton, output4_triton = model_triton(logits_triton, k, capacity_factor, min_capacity, enable_token_rearrange_opt)
    # output1_triton, output2_triton, output3_triton, output4_triton = fused_topkgating_opt(logits_triton, k, capacity_factor, min_capacity, True)
    
    assert output1_torch.shape == output1_triton.shape
    assert torch.allclose(output1_torch, output1_triton)
    assert output2_torch.shape == output2_triton.shape
    assert torch.allclose(output2_torch, output2_triton)
    assert output3_torch.shape == output3_triton.shape
    assert torch.allclose(output3_torch, output3_triton)
    assert torch.allclose(output4_torch, output4_triton)
   
    loss_torch = torch.sum(torch.mean(output1_torch * output3_torch))
    loss_triton = torch.sum(torch.mean(output1_triton * output3_triton))
    
    assert torch.allclose(loss_torch, loss_triton)

    # # for backward
    dout_torch = torch.randn_like(loss_torch)
    with torch.no_grad():
        dout_triton = dout_torch.clone()
    loss_torch.backward(dout_torch, retain_graph=True)
    # tmp_grad = logits_torch.grad.clone()
    # logits_torch.grad.zero_()
    # loss_torch.backward(dout_triton, retain_graph=True)
    # logits_triton.grad = logits_torch.grad
    # loss_test = torch.sum(torch.mean(mul_3 * view_9))
    # loss_test.backward(retain_graph=True)
    # loss_torch.backward(retain_graph=True)
    # output1_torch.backward()
    # print(f"logits grad max diff: {(back_res[0] - logits_torch.grad).abs().max().item()}")
    # print(f"logits grad mean diff: {(back_res[0] - logits_torch.grad).abs().mean().item()}")
    loss_triton.backward(dout_triton, retain_graph=True)
    # output1_torch.backward(retain_graph=True)
    # print(logits_torch.grad)
    # output1_triton.backward(retain_graph=True)


    print(f"logits grad max diff: {(logits_torch.grad - logits_triton.grad).abs().max().item()}")
    print(f"logits grad mean diff: {(logits_torch.grad - logits_triton.grad).abs().mean().item()}")
    
    assert logits_torch.grad.shape == logits_triton.grad.shape
    assert torch.allclose(logits_torch.grad, logits_triton.grad, rtol = 1e-8, atol = 1e-8)
    # vary seq length for fixed head and batch=4
    configs = []

    configs.append(
        triton.testing.Benchmark(
            x_names=["op"],
            x_vals=['fwd', 'bwd'],
            line_arg="provider",

            line_vals=["triton", "pytorch"],
            line_names=["Triton", "PyTorch"],

            styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="ms",
            plot_name=f"Experts{NumberExperts}-top{k}-gating-seqLen:{SeqLen}",
            args={
                "SeqLen": SeqLen
            },
        ))
    @triton.testing.perf_report(configs)
    def bench_top2gating(SeqLen, op, provider, device=device_):
        warmup = 100
        rep = 100
        shape = (SeqLen, NumberExperts)
        logits = torch.randn(shape, device=device, requires_grad=True)


        if "triton" in provider:
            if 'fwd' == op:
                fn = lambda: model_triton(logits, k, capacity_factor, min_capacity, enable_token_rearrange_opt)
                ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            elif 'bwd' == op:
                out0, out1, out2, _ =  model_triton(logits, k, capacity_factor, min_capacity, enable_token_rearrange_opt)
                loss = torch.sum(torch.mean(out0*out2))
                dout = torch.randn_like(loss)
                bwd_fn = lambda: loss.backward(dout, retain_graph=True)
                ms = triton.testing.do_bench(bwd_fn, warmup=warmup, rep=rep)
            else:
                raise Exception()
        
        
        if "pytorch" in provider:
            if 'fwd' == op:
                fn = lambda : model_torch(logits, k, capacity_factor, min_capacity, enable_token_rearrange_opt)
                ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            elif 'bwd' == op:
                # def iter(logits, k, capacity_factor, min_capacity, enable_token_rearrange_opt):
                out0, out_torch_pack =  model_torch(logits, k, capacity_factor, min_capacity, enable_token_rearrange_opt)
                output2_torch = out_torch_pack.token_rearranged_ec_idx
                output3_torch = out_torch_pack.token_exp_weights
                output4_torch = out_torch_pack.expert_select_token_idx
                loss = torch.sum(torch.mean(out0*output3_torch))
                dout = torch.randn_like(loss)
                loss.backward(dout, retain_graph=True)

                bwd_fn = lambda: loss.backward(dout, retain_graph=True)
                # bwd_fn = lambda: iter(logits, k, capacity_factor, min_capacity, enable_token_rearrange_opt)
                ms = triton.testing.do_bench(bwd_fn, warmup=warmup, rep=rep)
            else:
                raise Exception()

        return ms
    
    bench_top2gating.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    test()
    print("sucessfully!")
