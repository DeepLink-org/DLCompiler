# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
import triton
import triton.language as tl
import torch
import torch_npu
import pytest
import test_common
import functools
from test_common import TestUtils, check_ub_mem_overflow

# <<<<<<< test_xorsum_1d
def torch_xorsum(tensor, dim=None, keepdim=False):
    if dim is None:
        result = tensor.flatten()[0]
        for x in tensor.flatten()[1:]:
            result = result ^ x
        return result
    else:
        assert dim < tensor.dim(), f"Invalid dim {dim} for tensor shape {tensor.shape}"
        result = tensor.select(dim, 0)
        for i in range(1, tensor.size(dim)):
            result = result ^ tensor.select(dim, i)
        if keepdim:
            result = result.unsqueeze(dim)
        return result

@triton.jit
def triton_xorsum_1d(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) + tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr0 + xoffset, None)
    tmp4 = tl.xor_sum(tmp0, 0)
    tl.store(out_ptr1, tmp4, None)

@pytest.mark.parametrize('shape', TestUtils.test_shape1d)
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64'])
def test_xorsum_1d(dtype, shape):
    if check_ub_mem_overflow(dtype, shape):
        return
    x0 = test_common.generate_tensor(shape, dtype).npu()
    triton_res = torch.empty(1, dtype=eval("torch." + dtype)).npu()
    numel = shape[0]
    triton_xorsum_1d[1,1,1](x0, triton_res, numel, numel)
    torch_res = torch_xorsum(x0, dim=0, keepdim=True)
    test_common.validate_cmp(dtype, triton_res, torch_res)

# >>>>>>> test_xorsum_1d

# <<<<<<< test_xorsum_2d
@triton.jit
def triton_xorsum_2d(in_ptr0, out_ptr0, dim : tl.constexpr, M : tl.constexpr, N : tl.constexpr, MNUMEL: tl.constexpr, NNUMEL: tl.constexpr):
    mblk_idx = tl.arange(0,MNUMEL)
    nblk_idx = tl.arange(0,NNUMEL)
    mmask = mblk_idx < M
    nmask = nblk_idx < N
    mask = (mmask[:,None]) & (nmask[None,:])
    idx = mblk_idx[:,None] * N + nblk_idx[None,:]
    x = tl.load(in_ptr0 + idx, mask = mask, other = -float('inf'))
    tmp4 = tl.xor_sum(x, dim)
    if dim == 0:
        tl.store(out_ptr0 + tl.arange(0,N), tmp4, None)
    else:
        tl.store(out_ptr0 + tl.arange(0,M), tmp4, None)

@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64'])
@pytest.mark.parametrize('dim', [0, 1])
def test_xorsum_2d(dtype, shape, dim):
    if check_ub_mem_overflow(dtype, shape):
        return
    shapex, shapey = shape
    x0 = test_common.generate_tensor(shape, dtype).npu()
    triton_res = torch.empty([shape[1-dim], ], dtype=eval("torch." + dtype)).npu()
    triton_xorsum_2d[1,1,1](x0, triton_res, dim, shapex, shapey, shapex, shapey)
    torch_res = torch_xorsum(x0, dim=dim, keepdim=False)
    test_common.validate_cmp(dtype, triton_res, torch_res)

# >>>>>>> test_xorsum_2d

# <<<<<<< test_xorsum_3d
def torch_xorsum_3d(x0, no_reduce_dim):
    inp = x0 if x0.device == "cpu" else x0.cpu()
    if no_reduce_dim == 0:
        return torch_xorsum(torch_xorsum(inp, 1), 1).npu()
    elif no_reduce_dim == 1:
        return torch_xorsum(torch_xorsum(inp, 0), 1).npu()
    elif no_reduce_dim == 2:
        return torch_xorsum(torch_xorsum(inp, 0), 0).npu()
    else:
        assert False, f"no reduce dim not right, no_reduce_dim = {no_reduce_dim}"

@triton.jit
def triton_xorsum_3d_0_1(in_ptr, out_ptr,
    xnumel:tl.constexpr, ynumel:tl.constexpr, znumel:tl.constexpr,
    XB:tl.constexpr, YB:tl.constexpr, ZB:tl.constexpr):
    xidx = tl.arange(0,XB)
    yidx = tl.arange(0,YB)
    zidx = tl.arange(0,ZB)
    idx = xidx[:,None,None]*ynumel*znumel + yidx[None,:,None]*znumel + zidx[None,None,:]
    x = tl.load(in_ptr + idx)
    tmp = tl.xor_sum(x, 0)
    ret = tl.xor_sum(tmp, 0)
    oidx = zidx
    tl.store(out_ptr + oidx, ret)

@triton.jit
def triton_xorsum_3d_0_2(in_ptr, out_ptr,
    xnumel:tl.constexpr, ynumel:tl.constexpr, znumel:tl.constexpr,
    XB:tl.constexpr, YB:tl.constexpr, ZB:tl.constexpr):
    xidx = tl.arange(0,XB)
    yidx = tl.arange(0,YB)
    zidx = tl.arange(0,ZB)
    idx = xidx[:,None,None]*ynumel*znumel + yidx[None,:,None]*znumel + zidx[None,None,:]
    x = tl.load(in_ptr + idx)
    tmp = tl.xor_sum(x, 0)
    ret = tl.xor_sum(tmp, 1)
    oidx = yidx
    tl.store(out_ptr + oidx, ret)

@triton.jit
def triton_xorsum_3d_1_2(in_ptr, out_ptr,
    xnumel:tl.constexpr, ynumel:tl.constexpr, znumel:tl.constexpr,
    XB:tl.constexpr, YB:tl.constexpr, ZB:tl.constexpr):
    xidx = tl.arange(0,XB)
    yidx = tl.arange(0,YB)
    zidx = tl.arange(0,ZB)
    idx = xidx[:,None,None]*ynumel*znumel + yidx[None,:,None]*znumel + zidx[None,None,:]
    x = tl.load(in_ptr + idx)
    tmp = tl.xor_sum(x, 1)
    ret = tl.xor_sum(tmp, 1)
    oidx = xidx
    tl.store(out_ptr + oidx, ret)

def triton_xorsum_3d(in_ptr, out_ptr, xnumel, ynumel, znumel, XB, YB, ZB, no_reduce_dim):
    if no_reduce_dim == 0:
        triton_xorsum_3d_1_2[1, 1, 1](in_ptr, out_ptr, xnumel, ynumel, znumel, XB, YB, ZB)
    elif no_reduce_dim == 1:
        triton_xorsum_3d_0_2[1, 1, 1](in_ptr, out_ptr, xnumel, ynumel, znumel, XB, YB, ZB)
    elif no_reduce_dim == 2:
        triton_xorsum_3d_0_1[1, 1, 1](in_ptr, out_ptr, xnumel, ynumel, znumel, XB, YB, ZB)

@pytest.mark.parametrize('shape', TestUtils.test_shape3d)
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64'])
@pytest.mark.parametrize('no_reduce_dim', [0, 1, 2])
def test_xorsum_3d(dtype, shape, no_reduce_dim):
    x0 = test_common.generate_tensor(shape, dtype).npu()
    triton_res = torch.empty([shape[no_reduce_dim], ], dtype=eval("torch." + dtype)).npu()
    triton_xorsum_3d(x0, triton_res, shape[0], shape[1], shape[2], shape[0], shape[1], shape[2], no_reduce_dim)
    torch_res = torch_xorsum_3d(x0, no_reduce_dim)
    test_common.validate_cmp(dtype, triton_res, torch_res)

if __name__ == "__main__":
    test_xorsum_3d('int8', (3,3,3), 0)
# >>>>>>> test_xorsum_3d
