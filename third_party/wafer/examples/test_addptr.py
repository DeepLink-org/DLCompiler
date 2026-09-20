import torch
import torch_txda  # noqa: F401

import triton
import triton.language as tl


@triton.jit
def addptr(in0, out0):
    for i in range(0, 10, 2):
        in1 = in0 + 1 + i
        in2 = in1 + 1

        out1 = out0 + 1 + i
        out2 = out1 + 1

        a1 = tl.load(in1)
        a2 = tl.load(in2)

        tl.store(out1, a1)
        tl.store(out2, a2)


def test(device):
    input = torch.arange(0, 11, device="cpu", dtype=torch.float32)
    output = torch.full((11, ), 0, device="cpu", dtype=torch.float32)
    grid = lambda meta: (1, )

    print(output)
    input_txda = input.to("txda")
    output_txda = output.to("txda")
    addptr[grid](input_txda, output_txda)
    with torch.no_grad():
        output.copy_(output_txda.cpu())
    print(input)
    print(output)
    assert torch.equal(input, output)

    # TODO: need to check some conditions otherwise the code below does not make any difference for the test
    src = triton.compiler.ASTSource(
        fn=addptr,
        signature={'in0': '*fp32', 'out0': '*fp32'},
    )
    ret = triton.compile(src, )
    print(ret.asm["ttir"])
