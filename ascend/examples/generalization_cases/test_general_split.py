# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import triton
import triton.language as tl
import torch
import pytest
import test_common
from test_common import TestUtils
import math



@triton.jit
def fn_npu_(output_ptr, x_ptr, output_ptr1,
            XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
            XNUMEL:tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx=tl.arange(0,XB) + xoffs
    yidx=tl.arange(0,YB) + yoffs
    zidx=tl.arange(0,ZB) + zoffs

    idx=xidx[:,None,None,None]*YNUMEL*ZNUMEL*2+yidx[None,:,None,None]*ZNUMEL*2+ \
         zidx[None,None,:,None]*2 + tl.arange(0,2)[None,None,None,:]

    X = tl.load(x_ptr+idx)

    xx, yy = tl.split(X)

    oidx=xidx[:,None,None]*YNUMEL*ZNUMEL+yidx[None,:,None]*ZNUMEL+zidx[None,None,:]

    tl.store(output_ptr + oidx, xx)
    tl.store(output_ptr1 + oidx, yy)

import logging

@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
def test_split(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    x = torch.full(shape, 100, dtype=eval('torch.'+dtype)).npu()
    y = torch.full(shape, 30, dtype=eval('torch.'+dtype)).npu()
    xx = torch.stack((x, y), dim=-1)


    a, b = torch.split(xx, 1, dim=-1)

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

    a = a.reshape(XB, YB, ZB)
    b = b.reshape(XB, YB, ZB)
    output = torch.randint(1, (XB,YB,ZB), dtype=eval('torch.'+dtype)).npu()
    output1 = torch.randint(1, (XB,YB,ZB), dtype=eval('torch.'+dtype)).npu()

    grid = (1,1,1)
    if x.numel()*x.element_size() >= 8192:
        if xnumel > 1:
            grid = (XB,1,1)
            XB = 1
        elif ynumel > 1:
            grid = (1,YB,1)
            YB = 1
        else:
            grid = (1,1,ZB)
            ZB = 1

    fn_npu_[grid](output, xx, output1, XB, YB, ZB, xnumel, ynumel, znumel)

    test_common.validate_cmp(dtype, a, output)
    test_common.validate_cmp(dtype, b, output1)


    

