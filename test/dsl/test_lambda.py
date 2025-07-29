import torch
import triton
import triton.language as tl
import pytest

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # use lambda function to add x and y
    fn = lambda a, b: a + b
    output = fn(x, y)
    # output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=128)
    return output

def test_add():
    torch.manual_seed(0)
    size = 1024
    x = torch.rand(size, device='npu')
    y = torch.rand(size, device='npu')
    output_torch = x + y
    output_triton = add(x, y)
    # print(output_torch)
    # print(output_triton)
    assert torch.allclose(output_torch, output_triton, atol=1e-5)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')

NBLOCKS = 1
XS : tl.constexpr = 128
YS : tl.constexpr = 4
ZS : tl.constexpr = 8
NUMEL : tl.constexpr = XS * ZS

@triton.jit
def fn_broadcast(output_ptr, x_ptr, length):
    col_offsets = tl.arange(0, NUMEL)
    input = tl.load(x_ptr + col_offsets)
    count = lambda a, b, c: a * b * c
    # result = input.reshape((XS, 1, ZS)).broadcast_to((XS, YS, ZS)).reshape((XS * YS * ZS))
    result = input.reshape((XS, 1, ZS)).broadcast_to((XS, YS, ZS)).reshape((count(XS, YS, ZS)))
    brc_col_offsets = tl.arange(0, NUMEL * YS)
    tl.store(output_ptr + brc_col_offsets, result)

def test_broadcast():
    length = NUMEL

    x = torch.randn((XS, 1, ZS), dtype=torch.float32).npu()
    output = torch.randn((XS, YS, ZS), dtype=torch.float32).npu()
    fn_broadcast[NBLOCKS,1,1](output, x, length, debug=True)
    assert(torch.equal(output, x.repeat(1, YS, 1)))

if __name__ == "__main__":
    test_add()
    test_broadcast()
