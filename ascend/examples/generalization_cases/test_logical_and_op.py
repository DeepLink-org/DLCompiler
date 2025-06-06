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
def triton_logical_and_1d(in_ptr0, in_ptr1, out_ptr0, L : tl.constexpr):
    lblk_idx = tl.arange(0, L)
    idx = lblk_idx
    x0=tl.load(in_ptr0 + idx)
    x1=tl.load(in_ptr1 + idx)
    ret = x0.logical_and(x1)
    odx = lblk_idx
    tl.store(out_ptr0 + odx, ret)

@triton.jit
def triton_logical_and_2d(in_ptr0, in_ptr1, out_ptr0, L : tl.constexpr, M : tl.constexpr):
    loffs = tl.program_id(0) * L
    lblk_idx = tl.arange(0, L) + loffs
    mblk_idx = tl.arange(0, M)
    idx = lblk_idx[:,None] * M + mblk_idx[None, :]
    x0=tl.load(in_ptr0 + idx)
    x1=tl.load(in_ptr1 + idx)
    ret = x0.logical_and(x1)
    odx = lblk_idx[:, None] * M + mblk_idx[None, :]
    tl.store(out_ptr0 + odx, ret)

@triton.jit
def triton_logical_and_3d(in_ptr0, in_ptr1, out_ptr0, L : tl.constexpr, M : tl.constexpr, N : tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = lblk_idx[:,None,None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x0=tl.load(in_ptr0 + idx)
    x1=tl.load(in_ptr1 + idx)
    ret = x0.logical_and(x1)
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    tl.store(out_ptr0 + odx, ret)

support_typelist = ['bool',]

@pytest.mark.parametrize('shape', TestUtils.test_shape1_2_3d)
@pytest.mark.parametrize('sigtype', support_typelist)
def test_logical_and(shape, sigtype):
    logging.debug(f"dtype:{sigtype} shape:{shape}")
    dtype = eval('torch.' + sigtype)
    x0 = generate_tensor(shape=shape, dtype=sigtype).npu()
    x1 = generate_tensor(shape=shape, dtype=sigtype).npu()
    # ncore, xblock, xblock_sub = 2, 32768, 1024
    y_ref = torch.logical_and(x0, x1)
    output = torch.zeros(shape, dtype=dtype).npu()
    if len(shape) == 1:
        triton_logical_and_1d[1, 1, 1](x0, x1, output, shape[0])
    elif len(shape) == 2:
        shape0 = shape[0]
        shape1 = shape[1]
        if x0.numel() * x0.element_size() >= 8192:
            grid = (shape0, 1, 1)
            shape0 = 1
        else:
            grid = (1, 1, 1)
        triton_logical_and_2d[grid](x0, x1, output, shape0, shape1)
    elif len(shape) == 3:
        triton_logical_and_3d[1, 1, 1](x0, x1, output, shape[0], shape[1], shape[2])

    test_common.validate_cmp(sigtype, output, y_ref)

if __name__ == "__main__":
    for dtype in typelist:
        for shape in [(37,), (37, 3), (1,22,39)]:
            test_logical_and(shape, dtype)
