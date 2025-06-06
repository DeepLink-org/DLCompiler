# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import pytest
import triton
import triton.language as tl
import time
import torch
import torch_npu
import test_common
from test_common import TestUtils, generate_tensor
import logging

@triton.jit
def triton_logical_or_1d(in_ptr0, in_ptr1, out_ptr0, L : tl.constexpr):
    lblk_idx = tl.arange(0, L)
    idx = lblk_idx
    x0=tl.load(in_ptr0 + idx)
    x1=tl.load(in_ptr1 + idx)
    ret = x0.logical_or(x1)
    odx = lblk_idx
    tl.store(out_ptr0 + odx, ret)

@triton.jit
def triton_logical_or_2d(in_ptr0, in_ptr1, out_ptr0, L : tl.constexpr, M : tl.constexpr):
    pid = tl.program_id(0)
    lblk_idx = tl.arange(0, L) + pid * L
    mblk_idx = tl.arange(0, M)
    idx = lblk_idx[:,None] * M + mblk_idx[None, :]
    x0=tl.load(in_ptr0 + idx)
    x1=tl.load(in_ptr1 + idx)
    ret = x0.logical_or(x1)
    odx = lblk_idx[:, None] * M + mblk_idx[None, :]
    tl.store(out_ptr0 + odx, ret)

@triton.jit
def triton_logical_or_3d(in_ptr0, in_ptr1, out_ptr0, L : tl.constexpr, M : tl.constexpr, N : tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = lblk_idx[:,None,None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x0=tl.load(in_ptr0 + idx)
    x1=tl.load(in_ptr1 + idx)
    ret = x0.logical_or(x1)
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    tl.store(out_ptr0 + odx, ret)

support_typelist = ['bool',]

@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('sigtype', support_typelist)
def test_logical_or(shape, sigtype):
    logging.debug(f"dtype:{sigtype} shape:{shape}")
    dtype = eval('torch.' + sigtype)
    x0 = generate_tensor(shape=shape, dtype=sigtype).npu()
    x1 = generate_tensor(shape=shape, dtype=sigtype).npu()
    # ncore, xblock, xblock_sub = 2, 32768, 1024
    y_ref = torch.logical_or(x0, x1)
    output = torch.zeros(shape, dtype=dtype).npu()
    if len(shape) == 1:
        triton_logical_or_1d[1, 1, 1](x0, x1, output, shape[0])
    elif len(shape) == 2:
        triton_logical_or_2d[shape[0], 1, 1](x0, x1, output, 1, shape[1])
    elif len(shape) == 3:
        triton_logical_or_3d[1, 1, 1](x0, x1, output, shape[0], shape[1], shape[2])

    test_common.validate_cmp(sigtype, output, y_ref)
