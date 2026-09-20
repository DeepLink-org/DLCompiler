import torch
import torch_txda  # noqa: F401

import triton
import triton.language as tl



@triton.jit
def early_return(in0, out0):
    pid = tl.program_id(0)
    id = tl.load(in0 + pid)
    if id == -1:
        return
    offs = 1 + tl.arange(0, 4)
    out_offs = tl.arange(0, 4)
    tl.store(out0 + out_offs, offs)


def compile(device):
    src = triton.compiler.ASTSource(
        fn=early_return,
        signature={'in0': '*fp32', 'out0': '*fp32'},
    )
    ret = triton.compile(src, )
    print(ret.asm["ttir"])


def test_return_case(device):
    if device == 'cpu':
        pass  # Wafer driver is selected by conftest.py.

    SIZE = 8
    input = torch.full((SIZE, ), -1, device="cpu", dtype=torch.int32)
    output = torch.full((SIZE, ), -1, device="cpu", dtype=torch.int32)
    grid = lambda meta: (1, )
    print(output)
    input_txda = input.to("txda")
    output_txda = output.to("txda")
    early_return[grid](input_txda, output_txda)
    with torch.no_grad():
        output.copy_(output_txda.cpu())
    print(input)
    print(output)
    torch.testing.assert_close(torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1], dtype=torch.int32), output)


def test_normal_case(device):
    if device == 'cpu':
        pass  # Wafer driver is selected by conftest.py.

    SIZE = 8
    input = torch.arange(0, SIZE, device="cpu", dtype=torch.int32)
    output = torch.full((SIZE, ), -1, device="cpu", dtype=torch.int32)
    grid = lambda meta: (1, )
    print(output)
    input_txda = input.to("txda")
    output_txda = output.to("txda")
    early_return[grid](input_txda, output_txda)
    with torch.no_grad():
        output.copy_(output_txda.cpu())
    print(input)
    print(output)
    torch.testing.assert_close(torch.tensor([1, 2, 3, 4, -1, -1, -1, -1], dtype=torch.int32), output)
