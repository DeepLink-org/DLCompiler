from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
import triton

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

def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


class TopKGatingFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.Any, logits: torch.Tensor, k: int, capacity_factor: float = 1.0, min_capacity: int = 2):
        # compute the capacity
        capacity = _capacity(logits, torch.tensor(capacity_factor * k), torch.tensor(min_capacity)).item()
        gates, masks = dlblas._topk_gating_fwd_part1(logits, k)
        locations, exp_counts, res, ce = dlblas._topk_gating_fwd_part2(gates, masks, k)
        combine_weights, dispatch_mask = dlblas._topk_gating_fwd_part3(gates, masks, locations, k, capacity)
        l_aux = torch.mean(res)
        ctx.save_for_backward(locations, masks, gates, ce)
        return l_aux, combine_weights, dispatch_mask, exp_counts.to('cpu')
    

    @staticmethod
    def backward(ctx: torch.Any, *grad_outputs: torch.Any) -> torch.Any:
        grad_l_aux = grad_outputs[0].item()
        grad_combine = grad_outputs[1]
        locations, masks, gates, ce = ctx.saved_tensors
        grad_logits = dlblas._topk_gating_bwd(grad_l_aux, grad_combine, locations, masks, gates, ce)
        return grad_logits, None, None, None
    

def call(logits: torch.Tensor, k: int, capacity_factor: float = 1.0, min_capacity: int = 2):
    return TopKGatingFunc.apply(logits, k, capacity_factor, min_capacity)


def bench_fn(logits: torch.Tensor, k: int, capacity_factor: float = 1.0, min_capacity: int = 2):
    fn = lambda: call(logits, k, capacity_factor, min_capacity)
    ms = triton.testing.do_bench(fn, warmup=20, rep=20)
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
        register_dlblas_op(name, None, (logits, torch.SymInt, torch.SymFloat, torch.SymInt), call, bench_fn, call)

