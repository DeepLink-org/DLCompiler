# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Only floating point clamp is supported
import pytest

import triton
import triton.language as tl
import torch
import test_common
from test_common import TestUtils
import logging


@triton.jit
def triton_div(output_ptr, x_ptr, y_ptr, z_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
               XNUMEL: tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)

    ret = X / Y

    tl.store(output_ptr + idx, ret)


@pytest.mark.parametrize('shape', TestUtils.full_shape) # some shape with int8 over ub
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16'])
def test_add(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    x = test_common.generate_tensor(shape, dtype).npu()
    y = test_common.generate_tensor(shape, dtype).npu()
    z = test_common.generate_tensor(shape, dtype).npu()
    #x[x == 0] = 1
    y[y == 0] = 1

    new_shape = shape
    if dtype == 'int8' or dtype == 'int16' or dtype == 'int32' or dtype == 'int64':
        output = torch.randint(1, new_shape, dtype=eval('torch.float32')).npu()
        dtype = 'float32'
    else:
        output = torch.randint(1, new_shape, dtype=eval('torch.' + dtype)).npu()

    ans = x / y

    if len(shape) == 1:
        XB = 1
        xnumel = 1
        YB = 1
        ynumel = 1
        ZB = shape[0]
        znumel = shape[0]
    elif len(shape) == 2:
        XB = 1
        xnumel = 1
        YB = shape[0]
        ynumel = shape[0]
        ZB = shape[1]
        znumel = shape[1]
    else:
        XB = shape[0]
        xnumel = shape[0]
        YB = shape[1]
        ynumel = shape[1]
        ZB = shape[2]
        znumel = shape[2]

    grid = (1, 1, 1)
    if x.numel() * x.element_size() >= 8192:
        grid = (1, 1, ZB)
        ZB = 1

    triton_div[grid](output, x, y, z, XB, YB, ZB, xnumel, ynumel, znumel)
    test_common.validate_cmp(dtype, ans, output)
