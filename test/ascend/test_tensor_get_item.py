import torch
import torch_npu

import triton
import triton.language as tl
import triton.backends.dicp_triton.driver as dicp
import triton.language.extra.deeplink as dl

triton.runtime.driver.set_active(dicp.DICPDriver('ascend'))

import pytest

@triton.jit
def triton_kernel(x_ptr, y_ptr, output_ptr, POS: tl.constexpr, N: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * N
    offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < N
    x = tl.load(x_ptr + start + offsets, mask=mask)
    y = tl.load(y_ptr + start + offsets, mask=mask)
    out_left = x[:POS] + y[:POS]
    out_right = x[POS:] - y[POS:]
    out_left_offsets = tl.arange(0, POS)
    tl.store(output_ptr + start + out_left_offsets, out_left)
    out_right_offsets = POS + out_left_offsets
    tl.store(output_ptr + start + out_right_offsets, out_right, mask=out_right_offsets < N)

def triton_func(x: torch.Tensor, y: torch.Tensor, pos: int):
    output = torch.empty_like(x)
    M = x.size(0)
    N = x.size(1)
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    triton_kernel[(M,)](x, y, output, POS=pos, N=N, BLOCK_SIZE_N=BLOCK_SIZE_N)
    return output

def test_tensor_get_item():
    size = (128, 128)
    mid_pos = size[1] // 2
    x = torch.rand(size, device='npu')
    y = torch.rand(size, device='npu')
    torch_add_ref = x + y
    torch_sub_ref = x - y
    triton_cal = triton_func(x, y, mid_pos)
    torch.testing.assert_close(triton_cal[:,:mid_pos], torch_add_ref[:,:mid_pos])
    torch.testing.assert_close(triton_cal[:,mid_pos:], torch_sub_ref[:,mid_pos:])
