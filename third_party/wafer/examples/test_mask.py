import torch
import torch_txda  # noqa: F401

import triton
import triton.language as tl



def test_mask(device):

    @triton.jit
    def test(in0, out0):
        offs = 100 + tl.arange(0, 4)
        out_offs = tl.arange(0, 4)
        a = tl.load(in0 + offs, mask=offs < 4, other=-1)
        tl.store(out0 + out_offs, a)

    SIZE = 8
    input = torch.arange(0, SIZE, device="cpu", dtype=torch.int32)
    output = torch.full((SIZE, ), -2, device="cpu", dtype=torch.int32)

    if device == 'cpu':
        pass  # Wafer driver is selected by conftest.py.

    grid = lambda meta: (1, )

    src = triton.compiler.ASTSource(
        fn=test,
        signature={'in0': '*fp32', 'out0': '*fp32'},
    )
    ret = triton.compile(src, )
    print(ret.asm["ttir"])

    print(output)
    input_txda = input.to("txda")
    output_txda = output.to("txda")
    test[grid](input_txda, output_txda)
    with torch.no_grad():
        output.copy_(output_txda.cpu())
    print(input)
    print(output)
    torch.testing.assert_close(output, torch.tensor([-1, -1, -1, -1, -2, -2, -2, -2], device="cpu", dtype=torch.int32))
