import triton
import triton.language as tl
import torch
import numpy as np
import pytest
import test_common
from test_common import TestUtils
import math

@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, z_ptr,
            XB : tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
            XNUMEL:tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx=tl.arange(0,XB) + xoffs
    yidx=tl.arange(0,YB) + yoffs
    zidx=tl.arange(0,ZB) + zoffs

    idx=xidx[:,None,None]*YNUMEL*ZNUMEL+yidx[None,:,None]*ZNUMEL+zidx[None,None,:]

    X = tl.load(x_ptr+idx)

    ret = tl.rsqrt(X)

    tl.store(output_ptr+idx,ret)

import logging

@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('dtype', ['float32','float16','bfloat16'])
def test_rsqrt(dtype, shape,):
    x = test_common.generate_tensor(shape,dtype).abs().npu()
    y = test_common.generate_tensor(shape,dtype).abs().npu()
    z = test_common.generate_tensor(shape,dtype).abs().npu()
    new_shape = shape

    output = torch.randint(1, new_shape, dtype=eval('torch.'+dtype)).npu()
    output1 = output
    logging.log(logging.DEBUG, f"output.dtype={output.dtype}")

    ans = torch.rsqrt(x)

    if len(shape) == 1:
        XB = 1;xnumel = 1
        YB = 1;ynumel = 1
        ZB = shape[0];znumel = shape[0]
    elif len(shape) == 2:
        XB = 1;xnumel = 1
        YB = shape[0]; ynumel = shape[0]
        ZB = shape[1];znumel = shape[1]
    else:
        XB = shape[0];xnumel = shape[0]
        YB = shape[1];ynumel = shape[1]
        ZB = shape[2];znumel = shape[2]

    grid = (1,1,1)
    if x.numel()*x.element_size() >= 8192:
        grid = (1,1,ZB)
        ZB = 1

    fn_npu_[grid](output, x, y, z, XB, YB, ZB, xnumel, ynumel, znumel)

    test_common.validate_cmp(dtype, ans, output)

invalid_dtypes = [
    'int8',
    'int16',
    'int32',
    'uint32',
    'int64',
    'bool',
]
@pytest.mark.parametrize("dtype", invalid_dtypes)
@test_common.raises_with_match(triton.compiler.errors.CompilationError, "Expected dtype")
def test_rsqrt_invalid_dtype_case(dtype):
    x = test_common.generate_tensor((1,), dtype).npu()
    y = test_common.generate_tensor((1,), dtype).npu()
    z = test_common.generate_tensor((1,), dtype).npu()

    output = torch.randint(1, (1,), dtype=eval('torch.'+dtype)).npu()
    fn_npu_[1, 1, 1](output, x, y, z, 1, 1, 1, 1, 1, 1)
