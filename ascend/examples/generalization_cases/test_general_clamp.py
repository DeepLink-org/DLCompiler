# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Only floating point clamp is supported
import pytest

import triton
import triton.language as tl
import time
import torch
import torch_npu
import test_common
from test_common import TestUtils
import logging

def torch_clamp(x0,min_,max_):
    res = torch.clamp(x0, min_, max_)
    return res

@triton.jit
def tt_clamp_1d(in_ptr, out_ptr, min_ptr, max_ptr, 
    xnumel:tl.constexpr, ynumel:tl.constexpr, znumel:tl.constexpr,
    XB:tl.constexpr, YB:tl.constexpr, ZB:tl.constexpr):
    idx = tl.arange(0,XB)
    
    x = tl.load(in_ptr + idx)
    min_ = tl.load(min_ptr + idx)
    max_ = tl.load(max_ptr + idx)
    ret = tl.clamp(x,min_,max_)

    tl.store(out_ptr + idx, ret)

@triton.jit
def tt_clamp_2d(in_ptr, out_ptr, min_ptr, max_ptr, 
    xnumel:tl.constexpr, ynumel:tl.constexpr, znumel:tl.constexpr,
    XB:tl.constexpr, YB:tl.constexpr, ZB:tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    xidx = tl.arange(0,XB) + xoffs
    yidx = tl.arange(0,YB) + yoffs
    idx = xidx[:,None]*ynumel + yidx[None,:]
    
    x = tl.load(in_ptr + idx)
    min_ = tl.load(min_ptr + idx)
    max_ = tl.load(max_ptr + idx)
    ret = tl.clamp(x,min_,max_)

    tl.store(out_ptr + idx, ret)

@triton.jit
def tt_clamp_3d(in_ptr, out_ptr, min_ptr, max_ptr, 
    xnumel:tl.constexpr, ynumel:tl.constexpr, znumel:tl.constexpr,
    XB:tl.constexpr, YB:tl.constexpr, ZB:tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0,XB) + xoffs
    yidx = tl.arange(0,YB) + yoffs
    zidx = tl.arange(0,ZB) + zoffs

    idx = xidx[:,None,None]*ynumel*znumel + yidx[None,:,None]*znumel + zidx[None,None,:]
    
    x = tl.load(in_ptr + idx)
    min_ = tl.load(min_ptr + idx)
    max_ = tl.load(max_ptr + idx)
    ret = tl.clamp(x,min_,max_)

    tl.store(out_ptr + idx, ret)

@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
def test_clamp(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    torch.manual_seed(0)
    x = test_common.generate_tensor(shape,dtype).npu()
    a = test_common.generate_tensor(shape,dtype)
    b = test_common.generate_tensor(shape,dtype)
    min_ = torch.min(a,b).npu()
    max_ = torch.max(a,b).npu()

    grid = (1,1,1)

    y_cal = torch.empty(shape,dtype=eval('torch.'+dtype),device="npu")

    y_ref = torch_clamp(x, min_, max_)
    if len(shape) == 1:
        tt_clamp_1d[grid](x, y_cal, min_, max_, x.numel(), 1, 1, x.numel(), 1, 1)
    elif len(shape) == 2:
        xnumel, ynumel, znumel = shape + (1,)
        XB,YB,ZB = xnumel, ynumel, znumel
        if x.numel()*x.element_size()>8192:
            grid = (1,ynumel,1)
            YB = 1
        tt_clamp_2d[grid](x, y_cal, min_, max_, xnumel, ynumel, znumel, XB, YB, ZB)
        
    elif len(shape) == 3:
        xnumel, ynumel, znumel = shape
        XB,YB,ZB = xnumel, ynumel, znumel
        tt_clamp_3d[grid](x, y_cal, min_, max_, xnumel, ynumel, znumel, XB, YB, ZB)
    
    test_common.validate_cmp(dtype, y_cal, y_ref)