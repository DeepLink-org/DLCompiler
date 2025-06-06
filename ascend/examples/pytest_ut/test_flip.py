# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 
import triton
import triton.language as tl

import torch
import torch_npu
import pytest
import test_common
import triton.language.extra.ascend.libdevice as libdevice

@triton.jit
def fn_npu_(output_ptr, x_ptr,XB : tl.constexpr,YB : tl.constexpr,ZB : tl.constexpr):
    xidx=tl.arange(0,XB)
    yidx=tl.arange(0,YB)
    zidx=tl.arange(0,ZB)

    idx=xidx[:,None,None]*YB*ZB+yidx[None,:,None]*ZB+zidx[None,None,:]

    X = tl.load(x_ptr+idx)

    ret = libdevice.flip(X,2)

    oidx=xidx[:,None,None]*YB*ZB+yidx[None,:,None]*ZB+zidx[None,None,:]

    tl.store(output_ptr+idx,ret)


@pytest.mark.parametrize('para_type,data_type,XB,YB,ZB',
                         [
                             ['float32',torch.float32,2,256,16],
                             ['float32',torch.float32,4,8,8],
                             ['float16',torch.float16,2,256,16],
                             ['float16',torch.float16,4,4,8],
                             ['int8',torch.int8,2,256,16],
                             ['int8',torch.int8,4,4,8],
                         ]
                         )
def test_flip(para_type,data_type,XB,YB,ZB):
    x = torch.randint(low=-128,high=128,size=(XB,YB,ZB),dtype=data_type).npu()

    ans = torch.flip(x,dims=(-1,))
    print(f"toch_npu: {ans}")

    
    output = torch.randint(1, (XB,YB,ZB), dtype=data_type).npu()
    fn_npu_[1,1,1](output,x, XB, YB, ZB)
    print(f"triton: {output}")

    test_common.validate_cmp(para_type, ans, output)
