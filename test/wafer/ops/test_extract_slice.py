import torch
import torch_txda  # noqa: F401

import triton
import triton.language as tl
from triton.experimental.tle.language import dsa as dl


@triton.jit
def triton_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    out_sub = dl.extract_slice(output, [block_start], [32], [1])
    out_idx = block_start + tl.arange(0, 32)
    out_msk = out_idx < n_elements
    tl.store(output_ptr + out_idx, out_sub, mask=out_msk)


def triton_func(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    x_txda = x.to("txda")
    y_txda = y.to("txda")
    output_txda = output.to("txda")
    triton_kernel[grid](x_txda, y_txda, output_txda, n_elements, BLOCK_SIZE=1024)
    with torch.no_grad():
        output.copy_(output_txda.cpu())
    return output


def test_extract_slice():
    size = 1024
    x = torch.rand(size, device="cpu")
    y = torch.rand(size, device="cpu")
    torch_ref = x + y
    triton_cal = triton_func(x, y)
    print("max diff", (triton_cal[:32] - torch_ref[:32]).abs().max())
    torch.testing.assert_close(triton_cal[:32], torch_ref[:32])
