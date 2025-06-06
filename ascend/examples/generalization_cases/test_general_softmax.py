# 实际实现与官网定义不符，可能和triton submodule版本有关， 当前的submodule 不接受指定dim，都是按第0维做softmax
# arith.maximum 不支持类似 1x3 -> 3 和 1 -> 1 的reduce
import triton
import triton.language as tl
import torch
import pytest
import test_common
from test_common import TestUtils
import math

def torch_softmax_d0(x1):
    res = torch.softmax(x1, axis=0).to(x1.dtype)
    return res

@triton.jit
def tt_softmax_1d(in_ptr, out_ptr,
    xnumel:tl.constexpr, ynumel:tl.constexpr, znumel:tl.constexpr,
    XB:tl.constexpr, YB:tl.constexpr, ZB:tl.constexpr):
    idx = tl.arange(0,XB)
    x = tl.load(in_ptr + idx)
    ret = tl.softmax(x)
    tl.store(out_ptr + idx, ret)

@triton.jit
def tt_softmax_2d(in_ptr, out_ptr,
    xnumel:tl.constexpr, ynumel:tl.constexpr, znumel:tl.constexpr,
    XB:tl.constexpr, YB:tl.constexpr, ZB:tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    xidx = tl.arange(0,XB) + xoffs
    yidx = tl.arange(0,YB) + yoffs
    idx = xidx[:,None]*ynumel + yidx[None,:]
    
    a = tl.load(in_ptr + idx)
    ret = tl.softmax(a)

    tl.store(out_ptr + idx, ret)

@triton.jit
def tt_softmax_3d(in_ptr, out_ptr,
    xnumel:tl.constexpr, ynumel:tl.constexpr, znumel:tl.constexpr,
    XB:tl.constexpr, YB:tl.constexpr, ZB:tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0,XB) + xoffs
    yidx = tl.arange(0,YB) + yoffs
    zidx = tl.arange(0,ZB) + zoffs

    idx = xidx[:,None,None]*ynumel*znumel + yidx[None,:,None]*znumel + zidx[None,None,:]
    
    a = tl.load(in_ptr + idx)
    ret = tl.softmax(a)

    tl.store(out_ptr + idx, ret)


import logging

@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
def test_softmax(dtype, shape):
    logging.log(logging.DEBUG, f"shape = {shape}",flush=True)
    torch.manual_seed(0)
    x = torch.rand(shape,dtype=eval('torch.' + dtype),device="npu")*10
    grid = (1,1,1)

    y_cal = torch.rand(shape,dtype=eval('torch.'+dtype),device="npu")

    y_ref = torch_softmax_d0(x)
    if len(shape) == 1:
        tt_softmax_1d[grid](x, y_cal, x.numel(), 1, 1, x.numel(), 1, 1)
    elif len(shape) == 2:
        xnumel, ynumel, znumel = shape + (1,)
        XB,YB,ZB = xnumel, ynumel, znumel
        if x.numel()*x.element_size()>8192:
            grid = (1,ynumel,1)
            YB = 1
        tt_softmax_2d[grid](x, y_cal, xnumel, ynumel, znumel, XB, YB, ZB)
        
    elif len(shape) == 3:
        xnumel, ynumel, znumel = shape
        XB,YB,ZB = xnumel, ynumel, znumel
        tt_softmax_3d[grid](x, y_cal, xnumel, ynumel, znumel, XB, YB, ZB)
    
    test_common.validate_cmp(dtype, y_cal, y_ref)

invalid_types = [
    'int8',
    'int16',
    'int32',
    'uint32',
    'int64',
    'bool',
]
@pytest.mark.parametrize("dtype", invalid_types)
@test_common.raises_with_match(triton.compiler.errors.CompilationError, "Expected dtype")
def test_softmax_invalid_dtype_case(dtype):
    x0 = test_common.generate_tensor((1,), dtype).npu()

    y_cal = torch.zeros((1,), dtype=eval('torch.' + dtype)).npu()
    tt_softmax_1d[1, 1, 1](x0, y_cal, 0, 0, 0, 1, 0, 0)
