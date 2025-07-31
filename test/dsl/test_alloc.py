import torch
import triton
import triton.language as tl
import triton.language.extra.deeplink as dl
import pytest

@triton.jit
def custom_func_kernel(x_ptr,  # *Pointer* to first input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = dl.alloc([BLOCK_SIZE], 1.68, dtype=tl.float32, layout=dl.ND, scope=dl.UB)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def custom_func(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    custom_func_kernel[grid](x, output, n_elements, BLOCK_SIZE=128)
    return output

def test_add():
    torch.manual_seed(0)
    size = 1024
    x = torch.rand(size, device='npu')
    output_torch = x + 1.68
    output_triton = custom_func(x)
    
    assert torch.allclose(output_torch, output_triton, atol=1e-5)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')

if __name__ == "__main__":
    test_add()
