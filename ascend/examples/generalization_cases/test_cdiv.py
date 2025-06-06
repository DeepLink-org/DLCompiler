import logging

import pytest
import triton
import triton.language as tl
import torch
import torch_npu
import test_common
from test_common import TestUtils
import math
import logging


def torch_cdiv(x0, x1, dtype):
   return (x0 + x1 - 1) // x1

@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, z_ptr,
            XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
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

    ret = tl.cdiv(X, Y)

    tl.store(output_ptr + idx, ret)


@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('dtype',
                         ['int8', 'int16', 'int32', 'int64'])
def test_case2(dtype, shape):
    # 生成数据, cdiv int8 溢出的行为triton与torch_cpu不一致
    x = test_common.generate_tensor(shape, dtype).abs().npu() // 2
    y = test_common.generate_tensor(shape, dtype).npu()
    z = test_common.generate_tensor(shape, dtype).npu()
    y = (y.abs() // 2 + 1)
    new_shape = shape

    output = torch.randint(1, new_shape, dtype=eval('torch.' + dtype)).npu()
    output1 = output
    logging.debug(f"output.dtype={output.dtype}")

    ans = torch_cdiv(x.cpu(), y.cpu(), eval('torch.' + dtype))

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

    fn_npu_[grid](output, x, y, z, XB, YB, ZB, xnumel, ynumel, znumel)

    test_common.validate_cmp(dtype, ans, output)

