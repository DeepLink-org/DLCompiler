# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 
import triton
import triton.language as tl
import test_common

from test_common import TestUtils
import torch
import torch_npu
import pytest
import math
import random
    
@triton.jit
def fn_npu_int8_3d(output_ptr,X : tl.constexpr,Y : tl.constexpr,Z : tl.constexpr,XNUMEL : tl.constexpr,YNUMEL : tl.constexpr,ZNUMEL : tl.constexpr):
    xidx=tl.arange(0,XNUMEL)
    yidx=tl.arange(0,YNUMEL)
    zidx=tl.arange(0,ZNUMEL)
    Xmask = xidx<X
    Ymask = yidx<Y
    Zmask = zidx<Z
    mask = (Xmask[:,None,None])&(Ymask[None,:,None])&(Zmask[None,None,:])
    ret = tl.zeros((XNUMEL,YNUMEL,ZNUMEL),dtype = tl.int8)
    oidx=xidx[:,None,None]*Y*Z+yidx[None,:,None]*Z+zidx[None,None,:]
    tl.store(output_ptr+oidx,ret, mask = mask)

@triton.jit
def fn_npu_int16_3d(output_ptr,X : tl.constexpr,Y : tl.constexpr,Z : tl.constexpr,XNUMEL : tl.constexpr,YNUMEL : tl.constexpr,ZNUMEL : tl.constexpr):
    xidx=tl.arange(0,XNUMEL)
    yidx=tl.arange(0,YNUMEL)
    zidx=tl.arange(0,ZNUMEL)
    Xmask = xidx<X
    Ymask = yidx<Y
    Zmask = zidx<Z
    mask = (Xmask[:,None,None])&(Ymask[None,:,None])&(Zmask[None,None,:])
    ret = tl.zeros((XNUMEL,YNUMEL,ZNUMEL),dtype = tl.int16)
    oidx=xidx[:,None,None]*Y*Z+yidx[None,:,None]*Z+zidx[None,None,:]
    tl.store(output_ptr+oidx,ret, mask = mask)
    
@triton.jit
def fn_npu_int32_3d(output_ptr,X : tl.constexpr,Y : tl.constexpr,Z : tl.constexpr,XNUMEL : tl.constexpr,YNUMEL : tl.constexpr,ZNUMEL : tl.constexpr):
    xidx=tl.arange(0,XNUMEL)
    yidx=tl.arange(0,YNUMEL)
    zidx=tl.arange(0,ZNUMEL)
    Xmask = xidx<X
    Ymask = yidx<Y
    Zmask = zidx<Z
    mask = (Xmask[:,None,None])&(Ymask[None,:,None])&(Zmask[None,None,:])
    ret = tl.zeros((XNUMEL,YNUMEL,ZNUMEL),dtype = tl.int32)
    oidx=xidx[:,None,None]*Y*Z+yidx[None,:,None]*Z+zidx[None,None,:]
    tl.store(output_ptr+oidx,ret, mask = mask)
    
@triton.jit
def fn_npu_int64_3d(output_ptr,X : tl.constexpr,Y : tl.constexpr,Z : tl.constexpr,XNUMEL : tl.constexpr,YNUMEL : tl.constexpr,ZNUMEL : tl.constexpr):
    xidx=tl.arange(0,XNUMEL)
    yidx=tl.arange(0,YNUMEL)
    zidx=tl.arange(0,ZNUMEL)
    Xmask = xidx<X
    Ymask = yidx<Y
    Zmask = zidx<Z
    mask = (Xmask[:,None,None])&(Ymask[None,:,None])&(Zmask[None,None,:])
    ret = tl.zeros((XNUMEL,YNUMEL,ZNUMEL),dtype = tl.int64)
    oidx=xidx[:,None,None]*Y*Z+yidx[None,:,None]*Z+zidx[None,None,:]
    tl.store(output_ptr+oidx,ret, mask = mask)
    
@triton.jit
def fn_npu_fp16_3d(output_ptr,X : tl.constexpr,Y : tl.constexpr,Z : tl.constexpr,XNUMEL : tl.constexpr,YNUMEL : tl.constexpr,ZNUMEL : tl.constexpr):
    xidx=tl.arange(0,XNUMEL)
    yidx=tl.arange(0,YNUMEL)
    zidx=tl.arange(0,ZNUMEL)
    Xmask = xidx<X
    Ymask = yidx<Y
    Zmask = zidx<Z
    mask = (Xmask[:,None,None])&(Ymask[None,:,None])&(Zmask[None,None,:])
    ret = tl.zeros((XNUMEL,YNUMEL,ZNUMEL),dtype = tl.float16)
    oidx=xidx[:,None,None]*Y*Z+yidx[None,:,None]*Z+zidx[None,None,:]
    tl.store(output_ptr+oidx,ret, mask = mask)
    
@triton.jit
def fn_npu_fp32_3d(output_ptr,X : tl.constexpr,Y : tl.constexpr,Z : tl.constexpr,XNUMEL : tl.constexpr,YNUMEL : tl.constexpr,ZNUMEL : tl.constexpr):
    xidx=tl.arange(0,XNUMEL)
    yidx=tl.arange(0,YNUMEL)
    zidx=tl.arange(0,ZNUMEL)
    Xmask = xidx<X
    Ymask = yidx<Y
    Zmask = zidx<Z
    mask = (Xmask[:,None,None])&(Ymask[None,:,None])&(Zmask[None,None,:])
    ret = tl.zeros((XNUMEL,YNUMEL,ZNUMEL),dtype = tl.float32)
    oidx=xidx[:,None,None]*Y*Z+yidx[None,:,None]*Z+zidx[None,None,:]
    tl.store(output_ptr+oidx,ret, mask = mask)
    
@triton.jit
def fn_npu_bf16_3d(output_ptr,X : tl.constexpr,Y : tl.constexpr,Z : tl.constexpr,XNUMEL : tl.constexpr,YNUMEL : tl.constexpr,ZNUMEL : tl.constexpr):
    xidx=tl.arange(0,XNUMEL)
    yidx=tl.arange(0,YNUMEL)
    zidx=tl.arange(0,ZNUMEL)
    Xmask = xidx<X
    Ymask = yidx<Y
    Zmask = zidx<Z
    mask = (Xmask[:,None,None])&(Ymask[None,:,None])&(Zmask[None,None,:])
    ret = tl.zeros((XNUMEL,YNUMEL,ZNUMEL),dtype = tl.bfloat16)
    oidx=xidx[:,None,None]*Y*Z+yidx[None,:,None]*Z+zidx[None,None,:]
    tl.store(output_ptr+oidx,ret, mask = mask)
    
@triton.jit
def fn_npu_bool_3d(output_ptr,X : tl.constexpr,Y : tl.constexpr,Z : tl.constexpr,XNUMEL : tl.constexpr,YNUMEL : tl.constexpr,ZNUMEL : tl.constexpr):
    xidx=tl.arange(0,XNUMEL)
    yidx=tl.arange(0,YNUMEL)
    zidx=tl.arange(0,ZNUMEL)
    Xmask = xidx<X
    Ymask = yidx<Y
    Zmask = zidx<Z
    mask = (Xmask[:,None,None])&(Ymask[None,:,None])&(Zmask[None,None,:])
    ret = tl.zeros((XNUMEL,YNUMEL,ZNUMEL),dtype = tl.int1)
    oidx=xidx[:,None,None]*Y*Z+yidx[None,:,None]*Z+zidx[None,None,:]
    tl.store(output_ptr+oidx,ret, mask = mask)


@triton.jit
def fn_npu_int8_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr,
                   YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Ymask[:, None]) & (Zmask[None, :])
    ret = tl.zeros((YNUMEL, ZNUMEL), dtype=tl.int8)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)

@triton.jit
def fn_npu_int16_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr,
                   YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Ymask[:, None]) & (Zmask[None, :])
    ret = tl.zeros((YNUMEL, ZNUMEL), dtype=tl.int16)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)

@triton.jit
def fn_npu_int32_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr,
                   YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Ymask[:, None]) & (Zmask[None, :])
    ret = tl.zeros((YNUMEL, ZNUMEL), dtype=tl.int32)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)

@triton.jit
def fn_npu_int64_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr,
                   YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Ymask[:, None]) & (Zmask[None, :])
    ret = tl.zeros((YNUMEL, ZNUMEL), dtype=tl.int64)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_fp16_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr,
                   YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Ymask[:, None]) & (Zmask[None, :])
    ret = tl.zeros((YNUMEL, ZNUMEL), dtype=tl.float16)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_fp32_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr,
                   YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Ymask[:, None]) & (Zmask[None, :])
    ret = tl.zeros((YNUMEL, ZNUMEL), dtype=tl.float32)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_bf16_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr,
                   YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Ymask[:, None]) & (Zmask[None, :])
    ret = tl.zeros((YNUMEL, ZNUMEL), dtype=tl.bfloat16)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_bool_2d(output_ptr, Y: tl.constexpr, Z: tl.constexpr,
                   YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    yidx = tl.arange(0, YNUMEL)
    zidx = tl.arange(0, ZNUMEL)
    Ymask = yidx < Y
    Zmask = zidx < Z
    mask = (Ymask[:, None]) & (Zmask[None, :])
    ret = tl.zeros((YNUMEL, ZNUMEL), dtype=tl.int1)
    oidx = yidx[:, None] * Z + zidx[None, :]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_int8_1d(output_ptr, Z: tl.constexpr, ZNUMEL: tl.constexpr):
    zidx = tl.arange(0, ZNUMEL)
    Zmask = zidx < Z
    mask = (Zmask[:])
    ret = tl.zeros((ZNUMEL, ), dtype=tl.int8)
    oidx = zidx[:]
    tl.store(output_ptr + oidx, ret, mask=mask)

@triton.jit
def fn_npu_int16_1d(output_ptr, Z: tl.constexpr, ZNUMEL: tl.constexpr):
    zidx = tl.arange(0, ZNUMEL)
    Zmask = zidx < Z
    mask = (Zmask[:])
    ret = tl.zeros((ZNUMEL, ), dtype=tl.int16)
    oidx = zidx[:]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_int32_1d(output_ptr, Z: tl.constexpr, ZNUMEL: tl.constexpr):
    zidx = tl.arange(0, ZNUMEL)
    Zmask = zidx < Z
    mask = (Zmask[:])
    ret = tl.zeros((ZNUMEL, ), dtype=tl.int32)
    oidx = zidx[:]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_int64_1d(output_ptr, Z: tl.constexpr, ZNUMEL: tl.constexpr):
    zidx = tl.arange(0, ZNUMEL)
    Zmask = zidx < Z
    mask = (Zmask[:])
    ret = tl.zeros((ZNUMEL, ), dtype=tl.int64)
    oidx = zidx[:]
    tl.store(output_ptr + oidx, ret, mask=mask)

@triton.jit
def fn_npu_fp16_1d(output_ptr, Z: tl.constexpr, ZNUMEL: tl.constexpr):
    zidx = tl.arange(0, ZNUMEL)
    Zmask = zidx < Z
    mask = (Zmask[:])
    ret = tl.zeros((ZNUMEL, ), dtype=tl.float16)
    oidx = zidx[:]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_fp32_1d(output_ptr, Z: tl.constexpr, ZNUMEL: tl.constexpr):
    zidx = tl.arange(0, ZNUMEL)
    Zmask = zidx < Z
    mask = (Zmask[:])
    ret = tl.zeros((ZNUMEL, ), dtype=tl.float32)
    oidx = zidx[:]
    tl.store(output_ptr + oidx, ret, mask=mask)


@triton.jit
def fn_npu_bf16_1d(output_ptr, Z: tl.constexpr, ZNUMEL: tl.constexpr):
    zidx = tl.arange(0, ZNUMEL)
    Zmask = zidx < Z
    mask = (Zmask[:])
    ret = tl.zeros((ZNUMEL, ), dtype=tl.bfloat16)
    oidx = zidx[:]
    tl.store(output_ptr + oidx, ret, mask=mask)

@triton.jit
def fn_npu_bool_1d(output_ptr, Z: tl.constexpr, ZNUMEL: tl.constexpr):
    zidx = tl.arange(0, ZNUMEL)
    Zmask = zidx < Z
    mask = (Zmask[:])
    ret = tl.zeros((ZNUMEL, ), dtype=tl.int1)
    oidx = zidx[:]
    tl.store(output_ptr + oidx, ret, mask=mask)


test_dtype = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool']
test_shape1d = TestUtils.test_shape1d
test_shape2d = TestUtils.test_shape2d
test_shape3d = TestUtils.test_shape3d

# 定义 dtype 到 (test_func, test_sigtype) 的映射
dtype_mapping3d = {
    'int8': (fn_npu_int8_3d, torch.int8),
    'int16': (fn_npu_int16_3d, torch.int16),
    'int32': (fn_npu_int32_3d, torch.int32),
    'int64': (fn_npu_int64_3d, torch.int64),
    'float16': (fn_npu_fp16_3d, torch.float16),
    'float32': (fn_npu_fp32_3d, torch.float32),
    'bfloat16': (fn_npu_bf16_3d, torch.bfloat16),
    'bool': (fn_npu_bool_3d, torch.bool),
}
dtype_mapping2d = {
    'int8': (fn_npu_int8_2d, torch.int8),
    'int16': (fn_npu_int16_2d, torch.int16),
    'int32': (fn_npu_int32_2d, torch.int32),
    'int64': (fn_npu_int64_2d, torch.int64),
    'float16': (fn_npu_fp16_2d, torch.float16),
    'float32': (fn_npu_fp32_2d, torch.float32),
    'bfloat16': (fn_npu_bf16_2d, torch.bfloat16),
    'bool': (fn_npu_bool_2d, torch.bool),
}
dtype_mapping1d = {
    'int8': (fn_npu_int8_1d, torch.int8),
    'int16': (fn_npu_int16_1d, torch.int16),
    'int32': (fn_npu_int32_1d, torch.int32),
    'int64': (fn_npu_int64_1d, torch.int64),
    'float16': (fn_npu_fp16_1d, torch.float16),
    'float32': (fn_npu_fp32_1d, torch.float32),
    'bfloat16': (fn_npu_bf16_1d, torch.bfloat16),
    'bool': (fn_npu_bool_1d, torch.bool),
}

# 生成测试用例
testlist = [
    (func, sigtype, dtype, shape)
    for sigtype in test_dtype
    for shape in test_shape1d
    for func, dtype in [dtype_mapping1d[sigtype]]  # 直接解包映射结果
]

testlist += [
    (func, sigtype, dtype, shape)
    for sigtype in test_dtype
    for shape in test_shape2d
    for func, dtype in [dtype_mapping2d[sigtype]]  # 直接解包映射结果
]

testlist += [
    (func, sigtype, dtype, shape)
    for sigtype in test_dtype
    for shape in test_shape3d
    for func, dtype in [dtype_mapping3d[sigtype]]  # 直接解包映射结果
]

@pytest.mark.parametrize('testfunc, sigtype, dtype, shape', testlist)
def test_npu(testfunc, sigtype, dtype, shape):
    x = 0;
    output = 0
    if len(shape) == 3:
        x = torch.full((shape[0], shape[1], shape[2]), 0, dtype=dtype).npu()
        output = torch.randint(1, (shape[0], shape[1], shape[2]), dtype=dtype).npu()
        testfunc[(1, 1, 1)](output, shape[0], shape[1], shape[2], shape[0], shape[1], shape[2],debug=True)
    if len(shape) == 2:
        x = torch.full((shape[0], shape[1]), 0, dtype=dtype).npu()
        output = torch.randint(1, (shape[0], shape[1]), dtype=dtype).npu()
        shape0 = shape[0]
        shape1 = shape[1]
        if x.numel() * x.element_size() >= 8192:
            grid = (shape0, 1, 1)
            shape0 = 1
        else:
            grid = (1, 1, 1)
        testfunc[grid](output, shape0, shape1, shape0, shape1, debug=True)
    if len(shape) == 1:
        x = torch.full((shape[0], ), 0, dtype=dtype).npu()
        output = torch.randint(1, (shape[0],), dtype=dtype).npu()
        testfunc[1, 1, 1](output, shape[0], shape[0], debug=True)
    test_common.validate_cmp(sigtype, output, x)
