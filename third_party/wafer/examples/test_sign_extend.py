import torch
import torch_txda  # noqa: F401

import triton

import triton.language as tl



@triton.jit
def sign_extend(off, in0, out0, in0_size):
    offset = tl.load(off).to(tl.int64)
    offsets = offset + tl.arange(0, 4)
    a = tl.load(in0 + offsets, mask=offsets < in0_size, other=11)
    tl.store(out0 + tl.arange(0, 4), a)


def compile():
    src = triton.compiler.ASTSource(
        fn=sign_extend,
        signature={'off': '*i32', 'in0': '*fp32', 'out0': '*fp32', 'in0_size': 'i32'},
    )
    ret = triton.compile(src, )
    print(ret.asm["ttir"])


def test_sign_extend(device):
    if device == 'cpu':
        pass  # Wafer driver is selected by conftest.py.

    SIZE = 4
    offsets = torch.full((1, ), 1, device="cpu", dtype=torch.int32)
    input = torch.arange(0, SIZE, device="cpu", dtype=torch.int32)
    output = torch.full((SIZE, ), -1, device="cpu", dtype=torch.int32)
    grid = lambda meta: (1, )
    print(output)
    offsets_txda = offsets.to("txda")
    input_txda = input.to("txda")
    output_txda = output.to("txda")
    sign_extend[grid](offsets_txda, input_txda, output_txda, SIZE)
    with torch.no_grad():
        output.copy_(output_txda.cpu())
    print(input)
    print(output)
    torch.testing.assert_close(torch.tensor([1, 2, 3, 11], device="cpu", dtype=torch.int32), output)
