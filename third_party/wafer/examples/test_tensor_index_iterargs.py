import torch
import torch_txda  # noqa: F401

import triton
import triton.language as tl



def test_tensor_indices_nested_with_mask(device):

    @triton.jit
    def addptr_with_masks(in0, out0, mask_bound):
        offs = tl.arange(0, 4)
        out_offs = tl.arange(0, 4)
        # We're loading 16 elements here, the bound is set to 14 so that
        # the mask only applies to the last iteration's load
        # TODO: The current mask implementation in triton-shared does not seem
        # to work when the mask applies to the entire tensor load, perhaps
        # the lowerings for subviews with 0-dimensions do not work?
        for i in range(0, 4):
            mask = offs < mask_bound
            a = tl.load(in0 + offs, mask=mask, other=-11)
            tl.store(out0 + out_offs, a)
            offs += 4
            out_offs += 4

    SIZE = 17
    input = torch.arange(0, SIZE, device="cpu", dtype=torch.int32)
    output = torch.full((SIZE, ), -1, device="cpu", dtype=torch.int32)

    if device == 'cpu':
        pass  # Wafer driver is selected by conftest.py.

    grid = lambda meta: (1, )

    print(output)
    input_txda = input.to("txda")
    output_txda = output.to("txda")
    addptr_with_masks[grid](input_txda, output_txda, 14)
    with torch.no_grad():
        output.copy_(output_txda.cpu())
    expected_output = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -11, -11, -1], dtype=torch.int32,
                                   device="cpu")
    torch.testing.assert_close(output, expected_output)
    print(input)
    print(output)


def test_tensor_indices_nested(device):

    @triton.jit
    def tensor_indices_nested(in0, out0):
        offs = tl.arange(0, 4)
        out_offs = tl.arange(0, 4)
        for i in range(0, 2):
            offs += i * 2
            a = tl.load(in0 + offs)
            tl.store(out0 + out_offs, a)
            offs += 4
            out_offs += 4
            for j in range(0, 3):
                offs += j * 3
                a = tl.load(in0 + offs)
                tl.store(out0 + out_offs, a)
                offs += 4
                out_offs += 4

    SIZE = 64
    input = torch.arange(0, SIZE, device="cpu", dtype=torch.int32)
    output = torch.full((SIZE, ), -1, device="cpu", dtype=torch.int32)

    if device == 'cpu':
        pass  # Wafer driver is selected by conftest.py.

    grid = lambda meta: (1, )

    print(output)
    input_txda = input.to("txda")
    output_txda = output.to("txda")
    tensor_indices_nested[grid](input_txda, output_txda)
    with torch.no_grad():
        output.copy_(output_txda.cpu())
    expected_output = torch.tensor([
        0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 38, 39, 40, 41, 48, 49,
        50, 51, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1
    ], device="cpu", dtype=torch.int32)
    torch.testing.assert_close(output, expected_output)
    print(input)
    print(output)


def test_integer_tensor(device):

    @triton.jit
    def test_1(out0):
        offs = tl.arange(0, 4)
        out_offs = tl.arange(0, 4)
        for i in range(0, 2):
            tl.store(out0 + out_offs, offs)
            out_offs += 4
            offs += 4

    SIZE = 8
    input = torch.arange(0, SIZE, device="cpu", dtype=torch.int32)
    output = torch.full((SIZE, ), -1, device="cpu", dtype=torch.int32)

    if device == 'cpu':
        pass  # Wafer driver is selected by conftest.py.

    grid = lambda meta: (1, )

    print(output)
    output_txda = output.to("txda")
    test_1[grid](output_txda)
    with torch.no_grad():
        output.copy_(output_txda.cpu())
    print(input)
    print(output)
    torch.testing.assert_close(input, output)
    src = triton.compiler.ASTSource(
        fn=test_1,
        signature={'out0': '*fp32'},
    )
    ret = triton.compile(src, )
    print(ret.asm["ttir"])
