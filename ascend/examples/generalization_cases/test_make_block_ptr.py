# -*- coding: utf-8 -*-
# # Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import triton
import triton.language as tl

import torch
import torch_npu
import pytest
import test_common
from test_common import TestUtils
@triton.jit
def fn_npu_1d(output_ptr, x_ptr, y_ptr, z_ptr, output_ptr1, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    block_ptr_in = tl.make_block_ptr(
        base=x_ptr,
        shape=(XB,),
        strides=(1,),
        offsets=(0,),
        block_shape=(XB,),
        order=(0,),
    )
    X = tl.load(block_ptr_in)

    block_ptr_out = tl.make_block_ptr(
        base=output_ptr,
        shape=(XB,),
        strides=(1,),
        offsets=(0,),
        block_shape=(XB,),
        order=(0,),
    )
    tl.store(block_ptr_out, X)

@triton.jit
def fn_npu_2d(output_ptr, x_ptr, y_ptr, z_ptr, output_ptr1, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xoffset = tl.program_id(0)
    block_ptr_in = tl.make_block_ptr(
        base=x_ptr,
        shape=(XB, YB),
        strides=(YB, 1),
        offsets=(xoffset, 0),
        block_shape=(XB, YB),
        order=(1, 0),
    )
    X = tl.load(block_ptr_in)

    block_ptr_out = tl.make_block_ptr(
        base=output_ptr,
        shape=(XB, YB),
        strides=(YB, 1),
        offsets=(xoffset, 0),
        block_shape=(XB, YB),
        order=(1, 0),
    )
    tl.store(block_ptr_out, X)

@triton.jit
def fn_npu_3d(output_ptr, x_ptr, y_ptr, z_ptr, output_ptr1, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    block_ptr_in = tl.make_block_ptr(
        base=x_ptr,
        shape=(XB, YB, ZB),
        strides=(YB * ZB, ZB, 1),
        offsets=(0, 0, 0),
        block_shape=(XB, YB, ZB),
        order=(2, 1, 0),
    )
    X = tl.load(block_ptr_in)

    block_ptr_out = tl.make_block_ptr(
        base=output_ptr,
        shape=(XB, YB, ZB),
        strides=(YB * ZB, ZB, 1),
        offsets=(0, 0, 0),
        block_shape=(XB, YB, ZB),
        order=(2, 1, 0),
    )
    tl.store(block_ptr_out, X)

temporarily_not_support_dtype=['bool']

@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
@pytest.mark.parametrize('shape', TestUtils.full_shape)
def test_npu(dtype, shape):
    if dtype in temporarily_not_support_dtype:
        return
    x = test_common.generate_tensor(shape, dtype).npu()
    y = test_common.generate_tensor(shape, dtype).npu()
    z = test_common.generate_tensor(shape, dtype).npu()

    output = torch.randint(1, shape, dtype=eval('torch.' + dtype)).npu()
    output1 = output

    a = x
    if len(shape)==3:
        fn_npu_3d[1, 1, 1](output, x, y, z, output1, XB=shape[0], YB=shape[1], ZB=shape[2])
    elif len(shape)==2:
        if x.numel() * x.element_size() > 8192:
            fn_npu_2d[shape[0],1,1](output, x, y, z, output1, XB=1, YB=shape[1], ZB=1)
        else:
            fn_npu_2d[1,1,1](output, x, y, z, output1, XB=shape[0], YB=shape[1], ZB=1)
    else:
        fn_npu_1d[1,1,1](output, x, y, z, output1, XB=shape[0], YB=1, ZB=1)
    torch.testing.assert_close(output, a)
