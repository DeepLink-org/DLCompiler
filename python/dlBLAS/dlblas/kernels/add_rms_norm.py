import torch
import triton
import triton.language as tl
from torch import Tensor
from dlblas.utils import register_dlblas_op, SymVar, Tensor


@triton.jit
def add_rms_norm_kernel(
    hidden_states,
    weight,
    residual,
    residual_out,
    output,
    stride_s: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    """rms norm kernel."""
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    weight_data = tl.load(weight + offsets, mask=offsets < N_COLS)
    hidden_states_ptrs = hidden_states + prog_id * stride_s + offsets
    hidden_states_data = tl.load(hidden_states_ptrs, mask=offsets < N_COLS)
    if HAS_RESIDUAL:
        residual_ptrs = residual + prog_id * stride_s + offsets
        residual_data = tl.load(residual_ptrs, mask=offsets < N_COLS)
        hidden_states_data = hidden_states_data + residual_data
        residual_out_ptrs = residual_out + prog_id * stride_s + offsets
        tl.store(residual_out_ptrs, hidden_states_data, mask=offsets < N_COLS)
    xf = hidden_states_data.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / float(N_COLS)
    out = xf * tl.rsqrt(var + eps)
    out = out.to(hidden_states_data.dtype) * weight_data
    out_ptr = output + prog_id * stride_s + offsets
    tl.store(out_ptr, out, mask=offsets < N_COLS)


def call(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: torch.Tensor = None,
):
    """add rms norm."""

    feat_size = weight.shape[0]
    seq_len = hidden_states.shape[0]
    assert hidden_states.shape[1] == feat_size
    stride_s, _ = hidden_states.stride()

    BLOCK_N = triton.next_power_of_2(feat_size)
    out = torch.empty_like(hidden_states)
    residual_out = None
    if residual is not None:
        residual_out = torch.empty_like(residual)

    grid = (seq_len,)
    add_rms_norm_kernel[grid](
        hidden_states,
        weight,
        residual,
        residual_out,
        out,
        stride_s=stride_s,
        eps=eps,
        N_COLS=feat_size,
        BLOCK_N=BLOCK_N,
        HAS_RESIDUAL=(residual is not None),
        num_warps=4,
        num_stages=2,
    )
    if residual is not None:
        return out, residual_out
    else:
        return out


def bench_fn(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: torch.Tensor = None,
):
    fn = lambda: call(hidden_states, weight, eps, residual)
    ms = triton.testing.do_bench(fn, warmup=10, rep=10)
    return ms


# register
for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    for device in ["cuda"]:
        H, C = SymVar("H"), SymVar("C")
        # we dont' actually allocate tensor
        hidden_states = Tensor((H, C), dtype=dtype, device=device)
        residual = Tensor((H, C), dtype=dtype, device=device)
        weight = Tensor((C,), dtype=dtype, device=device)
        # space = ChoiceSpace([])
        register_dlblas_op(
            "add_rms_norm",
            None,
            (hidden_states, weight, torch.SymFloat, residual),
            call,
            bench_fn,
            call,
        )
        register_dlblas_op(
            "rms_norm",
            None,
            (hidden_states, weight, torch.SymFloat),
            call,
            bench_fn,
            call,
        )
