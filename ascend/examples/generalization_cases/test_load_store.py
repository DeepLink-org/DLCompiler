# -*- coding: utf-8 -*-
# # Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import triton
import triton.language as tl
import torch
import torch_npu
import pytest
import test_common
from test_common import TestUtils
import logging


@triton.jit
def fn_npu_1d(output_ptr, x_ptr, YB: tl.constexpr):
    idx = tl.arange(0, YB)
    X = tl.load(x_ptr + idx)
    tl.store(output_ptr + idx, X)

def torch_fn_npu_1d(x):
    return x

@triton.jit
def fn_npu_2d(output_ptr, x_ptr, YB: tl.constexpr, ZB: tl.constexpr):
    pid = tl.program_id(0)
    y_idx = tl.arange(0, YB)[:, None] + pid * YB
    z_idx = tl.arange(0, ZB)[None, :]
    idx = y_idx * ZB + z_idx

    X = tl.load(x_ptr + idx)

    tl.store(output_ptr + idx, X)

def torch_fn_npu_2d(x):
    return x


@triton.jit
def fn_npu_3d(output_ptr, x_ptr, YB: tl.constexpr, ZB: tl.constexpr, KB: tl.constexpr):
    y = tl.arange(0, YB)[:, None, None]
    z = tl.arange(0, ZB)[None, :, None]
    k = tl.arange(0, KB)[None, None, :]

    idx = y * ZB * KB + z * KB + k

    X = tl.load(x_ptr + idx)

    tl.store(output_ptr + idx, X)

def torch_fn_npu_3d(x):
    return x

@pytest.mark.parametrize('shape', TestUtils.test_shape1_2_3d)
@pytest.mark.parametrize('dtype', TestUtils.dtype_list)
def test_npu(shape, dtype):
    logging.debug(f'dtype:{dtype} shape:{shape}')
    data_type = eval('torch.' + dtype)
    x = torch.randint(low=0, high=2, size=shape, dtype=data_type).npu()
    triton_res = torch.empty(shape, dtype=data_type).npu()
    torch_res = x
    if len(shape) == 1:
        torch_res = torch_fn_npu_1d(x)
        fn_npu_1d[1, 1, 1](triton_res, x, shape[0])
        # uint32 转成 float32算精度，因为torch_npu不支持uint32类型张量的slice
        torch_res = torch_res if dtype != 'uint32' else torch_res.to(torch.float32)
        triton_res = triton_res if dtype != 'uint32' else triton_res.to(torch.float32)
        cmp_type = dtype if dtype != 'uint32' else 'float32'
        test_common.validate_cmp(cmp_type, triton_res[:2 * shape[0] // 3], torch_res[:2 * shape[0] // 3])
    elif len(shape) == 2:
        torch_res = torch_fn_npu_2d(x)
        fn_npu_2d[shape[0], 1, 1](triton_res, x, 1, shape[1])
        torch_res = torch_res if dtype != 'uint32' else torch_res.to(torch.float32)
        triton_res = triton_res if dtype != 'uint32' else triton_res.to(torch.float32)
        cmp_type = dtype if dtype != 'uint32' else 'float32'
        test_common.validate_cmp(cmp_type, triton_res[:2 * shape[0] // 3, :2 * shape[1] // 3],
                                 torch_res[:2 * shape[0] // 3, :2 * shape[1] // 3])
    elif len(shape) == 3:
        torch_res = torch_fn_npu_3d(x)
        fn_npu_3d[1, 1, 1](triton_res, x, shape[0], shape[1], shape[2])
        torch_res = torch_res if dtype != 'uint32' else torch_res.to(torch.float32)
        triton_res = triton_res if dtype != 'uint32' else triton_res.to(torch.float32)
        cmp_type = dtype if dtype != 'uint32' else 'float32'
        test_common.validate_cmp(cmp_type, triton_res[:2 * shape[0] // 3, :2 * shape[1] // 3, :2 * shape[2] // 3],
                                 torch_res[:2 * shape[0] // 3, :2 * shape[1] // 3, :2 * shape[2] // 3])

