import pytest
import triton
import triton.language as tl
import torch
import torch_npu
import triton.language.extra.ascend.libdevice as libdevice
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

    ret = tl.log(X)

    tl.store(output_ptr+idx,ret)

import logging

@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('dtype', ['float32','float16','bfloat16'])
def test_log(dtype, shape):
    logging.log(logging.DEBUG, f"shape = {shape}")
    x = torch.rand(shape, dtype=eval('torch.'+dtype)).npu()
    y = torch.rand(shape, dtype=eval('torch.'+dtype)).npu()
    z = torch.rand(shape, dtype=eval('torch.'+dtype)).npu()

    new_shape = shape

    output = torch.randint(1, new_shape, dtype=eval('torch.'+dtype)).npu()
    output1 = output
    logging.log(logging.DEBUG, f"output.dtype={output.dtype}")

    ans = torch.log(x).to(eval('torch.' + dtype))

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
