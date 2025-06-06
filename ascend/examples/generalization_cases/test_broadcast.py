# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import triton
import triton.language as tl
import torch
import pytest
import test_common
from test_common import TestUtils


@triton.jit
def fn_broadcast_1d(output_ptr, x_ptr,  XS: tl.constexpr, YS: tl.constexpr):
    xidx = tl.arange(0, XS)[None, :]
    base = tl.load(x_ptr + xidx)
    out = base.broadcast_to((YS, XS))
    oidx = tl.arange(0, YS)[:, None] * XS + tl.arange(0, XS)[None, :]
    tl.store(output_ptr + oidx, out)


@pytest.mark.parametrize('shape', TestUtils.test_shape1d)
@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
def test_npu_1d(shape, dtype):
    XS = shape[0]
    YS = 4

    x = test_common.generate_tensor((XS, ), dtype=dtype).npu()
    std = torch.broadcast_to(x, (YS, XS))
    output = test_common.generate_tensor((YS, XS), dtype=dtype).npu()
    fn_broadcast_1d[1, 1, 1](output, x, XS, YS)
    test_common.validate_cmp(dtype, std, output)


@triton.jit
def fn_broadcast_2d(output_ptr, x_ptr, NUMEL: tl.constexpr, XS: tl.constexpr, YS: tl.constexpr, ZS: tl.constexpr):
    zoffset = tl.program_id(0) * ZS
    zidx = tl.arange(0, ZS)[None, :]
    base = tl.load(x_ptr + zoffset + zidx)
    out = base.broadcast_to((YS, ZS))
    oidx = zoffset * YS + tl.arange(0, YS)[:, None] * ZS + zidx
    tl.store(output_ptr + oidx, out)


@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
def test_npu_2d(shape, dtype):
    XS = shape[0]
    ZS = shape[1]
    YS = 4
    NUMEL = XS * ZS

    x = test_common.generate_tensor((XS, 1, ZS), dtype=dtype).npu()  # randn not support int type
    std = torch.broadcast_to(x, (XS, YS, ZS))
    output = test_common.generate_tensor((XS, YS, ZS), dtype=dtype).npu()
    fn_broadcast_2d[XS, 1, 1](output, x, NUMEL, XS, YS, ZS)
    test_common.validate_cmp(dtype, std, output)


@triton.jit
def triton_broadcast_to_dim0(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = tl.arange(0, 1)[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x = tl.load(in_ptr0 + idx)
    x1 = tl.load(out_ptr0 + odx)
    ret = tl.broadcast(x, x1)
    tl.store(out_ptr0 + odx, ret)


@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
@pytest.mark.parametrize('shape', TestUtils.test_shape3d)
def test_broadcast_to_dim0(shape, dtype):
    L, M, N = shape
    x0 = test_common.generate_tensor(shape=(1, M, N), dtype=dtype).npu()
    ans = x0.repeat(L, 1, 1)
    output = torch.zeros((L, M, N), dtype=eval('torch.' + dtype)).npu()
    triton_broadcast_to_dim0[1, 1, 1](x0, output, L, M, N, debug=True)
    test_common.validate_cmp(dtype, output, ans)


@triton.jit
def triton_broadcast_to_dim1(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = lblk_idx[:, None, None] * N * 1 + tl.arange(0, 1)[None, :, None] * N + nblk_idx[None, None, :]
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x = tl.load(in_ptr0 + idx)
    x1 = tl.load(out_ptr0 + odx)
    ret = tl.broadcast(x, x1)
    tl.store(out_ptr0 + odx, ret)


@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
@pytest.mark.parametrize('shape', TestUtils.test_shape3d)
def test_broadcast_to_dim1(shape, dtype):
    L, M, N = shape
    x0 = test_common.generate_tensor(shape=(L, 1, N), dtype=dtype).npu()
    ans = x0.repeat(1, M, 1)
    output = torch.zeros((L, M, N), dtype=eval('torch.' + dtype)).npu()
    triton_broadcast_to_dim1[1, 1, 1](x0, output, L, M, N, debug=True)
    test_common.validate_cmp(dtype, output, ans)


@triton.jit
def triton_broadcast_to_dim2(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = lblk_idx[:, None, None] * 1 * M + mblk_idx[None, :, None] * 1 + tl.arange(0, 1)[None, None, :]
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x = tl.load(in_ptr0 + idx)
    x1 = tl.load(out_ptr0 + odx)
    ret = tl.broadcast(x, x1)
    tl.store(out_ptr0 + odx, ret)


@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
@pytest.mark.parametrize('shape', TestUtils.test_shape3d)
def test_broadcast_to_dim2(shape, dtype):
    L, M, N = shape
    x0 = test_common.generate_tensor(shape=(L, M, 1), dtype=dtype).npu()
    ans = x0.repeat(1, 1, N)
    output = torch.zeros((L, M, N), dtype=eval('torch.' + dtype)).npu()
    triton_broadcast_to_dim2[1, 1, 1](x0, output, L, M, N, debug=True)
    test_common.validate_cmp(dtype, output, ans)


@triton.jit
def triton_broadcast_to_dim01(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = tl.arange(0, 1)[:, None, None] * N * 1 + tl.arange(0, 1)[None, :, None] * N + nblk_idx[None, None, :]
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x = tl.load(in_ptr0 + idx)
    x1 = tl.load(out_ptr0 + odx)
    ret = tl.broadcast(x, x1)
    tl.store(out_ptr0 + odx, ret)


@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
@pytest.mark.parametrize('shape', TestUtils.test_shape3d)
def test_broadcast_to_dim01(shape, dtype):
    L, M, N = shape
    x0 = test_common.generate_tensor(shape=(1, 1, N), dtype=dtype).npu()
    ans = x0.repeat(L, M, 1)
    output = torch.zeros((L, M, N), dtype=eval('torch.' + dtype)).npu()
    triton_broadcast_to_dim01[1, 1, 1](x0, output, L, M, N, debug=True)
    test_common.validate_cmp(dtype, output, ans)


@triton.jit
def triton_broadcast_to_dim02(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = tl.arange(0, 1)[:, None, None] * M * 1 + mblk_idx[None, :, None] * 1 + tl.arange(0, 1)[None, None, :]
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x = tl.load(in_ptr0 + idx)
    x1 = tl.load(out_ptr0 + odx)
    ret = tl.broadcast(x, x1)
    tl.store(out_ptr0 + odx, ret)


@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
@pytest.mark.parametrize('shape', TestUtils.test_shape3d)
def test_broadcast_to_dim02(shape, dtype):
    L, M, N = shape
    x0 = test_common.generate_tensor(shape=(1, M, 1), dtype=dtype).npu()
    ans = x0.repeat(L, 1, N)
    output = torch.zeros((L, M, N), dtype=eval('torch.' + dtype)).npu()
    triton_broadcast_to_dim02[1, 1, 1](x0, output, L, M, N, debug=True)
    test_common.validate_cmp(dtype, output, ans)


@triton.jit
def triton_broadcast_to_dim12(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = lblk_idx[:, None, None] * 1 * 1 + tl.arange(0, 1)[None, :, None] * 1 + tl.arange(0, 1)[None, None, :]
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x = tl.load(in_ptr0 + idx)
    x1 = tl.load(out_ptr0 + odx)
    ret = tl.broadcast(x, x1)
    tl.store(out_ptr0 + odx, ret)


@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
@pytest.mark.parametrize('shape', TestUtils.test_shape3d)
def test_broadcast_to_dim12(shape, dtype):
    L, M, N = shape
    x0 = test_common.generate_tensor(shape=(L, 1, 1), dtype=dtype).npu()
    ans = x0.repeat(1, M, N)
    output = torch.zeros((L, M, N), dtype=eval('torch.' + dtype)).npu()
    triton_broadcast_to_dim12[1, 1, 1](x0, output, L, M, N, debug=True)
    test_common.validate_cmp(dtype, output, ans)
