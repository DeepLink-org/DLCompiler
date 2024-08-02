from typing import Tuple
from torch import Tensor

# this import all kernels dynamically
import dlblas.kernels
from dlblas.utils import get_op
__version__ = "0.0.1"

def topk_gating(logits: Tensor, k: int, capacity_factor: float = 1.0, min_capacity: int = 2) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    op = get_op("topk_gating", (logits, k, capacity_factor, min_capacity))
    return op(logits, k, capacity_factor, min_capacity)


def layernorm_gated(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, is_rms_norm=False):
    op = get_op("layernorm_gated", (x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm))
    return op(x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm)


def selective_state_update(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
    op = get_op("selective_state_update", (state, x, dt, A, B, C, D, z, dt_bias, dt_softplus))
    return op(state, x, dt, A, B, C, D, z, dt_bias, dt_softplus)


def matmul(a:Tensor, b: Tensor, activation=""):
    if activation == "leaky_relu":
        op = get_op("matmul_leaky_relu", (a, b, activation))
        return op(a, b, activation)
    elif activation == "":
        op = get_op("matmul", (a, b))
        return op(a, b)
    else:
        raise f"matmul_{activation} not impl."


def _topk_gating_fwd_part1(logits: Tensor, k: int):
    op = get_op("_topk_gating_fwd_part1", (logits, k))
    return op(logits, k)


def _topk_gating_fwd_part2(gates: Tensor, masks: Tensor, k: int):
    op = get_op("_topk_gating_fwd_part2", (gates, masks, k))
    return op(gates, masks, k)


def _topk_gating_fwd_part3(gates: Tensor, masks: Tensor, locations: Tensor, k: int, capacity: int):
    op = get_op("_topk_gating_fwd_part3", (gates, masks, locations, k, capacity))
    return op(gates, masks, locations, k, capacity)

