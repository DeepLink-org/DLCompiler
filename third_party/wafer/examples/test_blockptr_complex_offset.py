import torch
import torch_txda  # noqa: F401

import triton
import triton.language as tl


@triton.jit
def block_copy_kernel(a_ptr, b_ptr):
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr + 8,
        shape=(2, 2),
        strides=(2, 1),
        offsets=(0, 0),
        block_shape=(2, 2),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(2, 2),
        strides=(2, 1),
        offsets=(0, 0),
        block_shape=(2, 2),
        order=(1, 0),
    )
    a = tl.load(a_block_ptr, boundary_check=(0, ))
    tl.store(b_block_ptr, a, boundary_check=(0, ))


def test(device):
    input = torch.arange(0, 16, device="cpu", dtype=torch.float32)
    output = torch.full((4, ), -1, device="cpu", dtype=torch.float32)
    expected = torch.arange(8, 12, device="cpu")
    grid = lambda meta: (1, )

    input_txda = input.to("txda")
    output_txda = output.to("txda")
    block_copy_kernel[grid](input_txda, output_txda)
    with torch.no_grad():
        output.copy_(output_txda.cpu())
    assert torch.equal(expected, output)
