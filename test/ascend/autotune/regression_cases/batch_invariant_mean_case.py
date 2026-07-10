import torch
import triton
import triton.language as tl
from common import (
    assert_close,
    bench,
    case_values,
    dtype_from_name,
    init,
    parse_args,
    report,
)

MODELS = "Runtime batch-invariant replacement; kernel coverage does not depend on model availability"


def mean_reference_check(tuner, config, *args, **kwargs):
    input_3d = args[0]
    output_2d = args[1]
    M = args[7]
    N = args[8]
    K = args[9]
    output_2d.zero_()
    tuner._make_kernel_call(*args, config=config, **kwargs)(warmup=False)
    ref = torch.mean(input_3d.reshape(M, N, K).to(torch.float16), dim=1)
    return torch.allclose(output_2d, ref.reshape_as(output_2d), atol=1e-2, rtol=1e-2)


MEAN_SEARCH_HINTS = {
    "search_params": {
        "params": ["BLOCK_SIZE"],
        "reference_fn": mean_reference_check,
    },
}


@triton.autotune(
    configs=[],
    key=[],
    hints=MEAN_SEARCH_HINTS,
)
@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,  # size before reduction dim
    N,  # size of reduction dim
    K,  # size after reduction dim
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing mean along a single dimension.
    Input is viewed as (M, N, K) where N is the dimension being reduced.
    """
    # Program ID gives us which output element we're computing
    pid = tl.program_id(0)

    # Compute output indices
    m_idx = pid // K
    k_idx = pid % K

    # Bounds check
    if m_idx >= M or k_idx >= K:
        return
    # Accumulate sum across reduction dimension
    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N

        # Calculate input indices
        input_idx = (
            m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2
        )
        # Load and accumulate
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    # Compute mean and store
    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)


def mean_dim(
    input_: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    assert (
        -input_.ndim <= dim < input_.ndim
    ), f"Invalid dimension {dim} for tensor with {input_.ndim} dimensions"
    if dim < 0:
        dim = dim + input_.ndim
    if dtype is None:
        if input_.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            dtype = torch.float32
        else:
            dtype = input_.dtype
    if input_.dtype != dtype:
        input_ = input_.to(dtype)
    shape = list(input_.shape)
    M = 1
    for i in range(dim):
        M *= shape[i]
    N = shape[dim]
    K = 1
    for i in range(dim + 1, len(shape)):
        K *= shape[i]
    input_3d = input_.reshape(M, N, K)
    if keepdim:
        output_shape = shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = shape[:dim] + shape[dim + 1 :]
    output = torch.empty(output_shape, dtype=dtype, device=input_.device)
    if keepdim:
        output_2d = output.reshape(M, 1, K).squeeze(1)
    else:
        output_2d = output.reshape(M, K)
    grid = (M * K,)
    mean_kernel[grid](
        input_3d,
        output_2d,
        input_3d.stride(0),
        input_3d.stride(1),
        input_3d.stride(2),
        output_2d.stride(0),
        output_2d.stride(1) if output_2d.ndim > 1 else 0,
        M,
        N,
        K,
    )
    return output


def main():
    args = parse_args()
    init(args.seed)
    dtype = dtype_from_name(args.dtype)
    for rows, hidden in case_values(
        ((512, 4096), (1024, 5120), (2048, 7168), (4096, 8192))
    ):
        x = torch.randn(rows, hidden, dtype=dtype, device=args.device)
        tri = mean_dim(x, dim=-1, dtype=torch.float16)
        ref = torch.mean(x.to(torch.float16), dim=-1)
        assert_close("mean_kernel", tri, ref, args.rtol or 1e-2, args.atol or 1e-2)
        torch_ms, _ = bench(
            lambda: torch.mean(x.to(torch.float16), dim=-1), args.warmup, args.repeat
        )
        triton_ms, _ = bench(
            lambda: mean_dim(x, dim=-1, dtype=torch.float16), args.warmup, args.repeat
        )
        report(f"mean_kernel shape=({rows}, {hidden})", torch_ms, triton_ms, MODELS)


def test_benchmark():
    main()


if __name__ == "__main__":
    main()
