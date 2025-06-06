import pytest
import triton
import triton.language as tl
import torch
import torch_npu
import test_common
from test_common import TestUtils
import math
import logging

def torch_eq(x0, x1):
    if x0.dtype != torch.uint32:
        return x0 == x1
    else:
        return x0.to(torch.float32) == x1.to(torch.float32)

@triton.jit
def triton_eq(in_ptr0, in_ptr1, out_ptr0, N: tl.constexpr, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x_index = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + x_index, mask=x_index < N)
        tmp1 = tl.load(in_ptr1 + x_index, mask=x_index < N)
        tmp2 = tmp0 == tmp1
        tl.store(out_ptr0 + x_index, tmp2, mask=x_index < N)

@pytest.mark.parametrize('shape', TestUtils.test_shape1_2_3d)
@pytest.mark.parametrize('dtype', ['int8','int16','int32','int64','float16','bfloat16','float32'])
def test_eq(shape, dtype):
    logging.debug(f'dtype:{dtype} shape:{shape}')
    # 生成数据
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype).npu()

    numel = x0.numel()
    ncore = 1 if numel <= 32 else 32
    xblock = math.ceil(numel / ncore)
    xblock_sub = numel if numel <= ncore else math.ceil(numel / ncore)

    # torch结果
    torch_res = torch_eq(x0, x1).to(eval('torch.' + dtype))
    # triton结果
    triton_res = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    N = triton_res.numel()
    triton_eq[ncore, 1, 1](x0, x1, triton_res, N, xblock, xblock_sub)
    # 比较结果
    torch_res = torch_res if dtype != 'uint32' else torch_res.to(torch.float32)
    triton_res = triton_res if dtype != 'uint32' else triton_res.to(torch.float32)
    cmp_dtype = dtype if dtype != 'uint32' else 'float32'
    test_common.validate_cmp(cmp_dtype, triton_res, torch_res)

if __name__ == "__main__":
    for dtype in TestUtils.dtype_list:
        for shape in [(37,), (37, 3), (1, 22, 39)]:
            test_eq(shape, dtype)
