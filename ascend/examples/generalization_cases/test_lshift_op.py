# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import pytest
import triton
import triton.language as tl
import time
import torch
import torch_npu
import test_common
from test_common import TestUtils

@triton.jit
def triton_lshift_1d(in_ptr0, out_ptr0, L : tl.constexpr):
    lblk_idx = tl.arange(0,L)
    idx = lblk_idx[:]
    x0=tl.load(in_ptr0+idx)
    ret = x0 << 2
    odx = lblk_idx[:]
    tl.store(out_ptr0+odx, ret)

@triton.jit
def triton_lshift_2d(in_ptr0, out_ptr0, M : tl.constexpr, N : tl.constexpr):
    moffs = tl.program_id(0) * M
    mblk_idx = tl.arange(0,M) + moffs
    nblk_idx = tl.arange(0,N)
    idx = mblk_idx[:,None]*N+nblk_idx[None,:]
    x0=tl.load(in_ptr0+idx)
    ret = x0 << 2
    odx = mblk_idx[:,None]*N+nblk_idx[None,:]
    tl.store(out_ptr0+odx, ret)

@triton.jit
def triton_lshift_3d(in_ptr0, out_ptr0, L : tl.constexpr, M : tl.constexpr, N : tl.constexpr):
    loffs = tl.program_id(0) * L
    lblk_idx = tl.arange(0,L) + loffs
    mblk_idx = tl.arange(0,M)
    nblk_idx = tl.arange(0,N)
    idx = lblk_idx[:,None,None]*N*M+mblk_idx[None,:,None]*N+nblk_idx[None,None,:]
    x0=tl.load(in_ptr0+idx)
    ret = x0 << 2
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    tl.store(out_ptr0+odx, ret)

dtype_mapping = {
    'int8': (torch.int8),
    'int16': (torch.int16),
    'int32': (torch.int32),
    'uint32': (torch.uint32),
    'int64': (torch.int64),
    'float16': (torch.float16),
    'float32': (torch.float32),
    'bfloat16': (torch.bfloat16),
    'bool': (torch.bool),
}

typelist = ['int8','int16','int32','int64',]


@pytest.mark.parametrize('shape', TestUtils.test_shape1_2_3d)
@pytest.mark.parametrize('sigtype',typelist)
def test_lshift(sigtype, shape):
    dtype = dtype_mapping[sigtype]
    x0 = test_common.generate_tensor(shape = shape, dtype = sigtype).npu()
    # ncore, xblock, xblock_sub = 2, 32768, 1024
    y_ref = x0 << 2
    output = torch.zeros(shape, dtype=dtype).npu()
    if len(shape) == 3:
        shape0 = shape[0]
        shape1 = shape[1]
        shape2 = shape[2]
        if x0.numel() * x0.element_size() >= 1024:
            grid = (shape0, 1, 1)
            shape0 = 1
        else:
            grid = (1, 1, 1)
        triton_lshift_3d[grid](x0, output, shape0, shape1, shape2)
    if len(shape) == 2:
        shape0 = shape[0]
        shape1 = shape[1]
        if x0.numel() * x0.element_size() >= 1024:
            grid = (shape0, 1, 1)
            shape0 = 1
        else:
            grid = (1, 1, 1)
        triton_lshift_2d[grid](x0, output, shape0, shape1)
    if len(shape) == 1:
        triton_lshift_1d[1, 1, 1](x0, output, shape[0])
    test_common.validate_cmp(sigtype, output, y_ref)
