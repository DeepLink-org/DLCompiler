import triton
import triton.language as tl
import torch
import pytest
import test_common
from test_common import TestUtils
import math

@triton.jit
def atomic_add(in_ptr0, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE : tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_add(out_ptr0 + (x1), tmp0, xmask)
    # tl.store(out_ptr1 + (x1), tmp1, xmask)


@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16', 'int32', 'int16', 'int8'])
def test_atomic_add(dtype, shape):
    ncore = 1
    block_size = shape[0] * shape[1] / ncore
    split_size = shape[0] // ncore
    x0_value = 3
    x0 = torch.full(shape, x0_value, dtype = eval('torch.float32')).npu()
    x1 = torch.full((split_size, shape[1]), 2, dtype = eval('torch.float32')).npu()
    y = torch.full((split_size, shape[1]), -10, dtype = eval('torch.float32')).npu()

    y_ref = x1 + 0
    x1_ref = x1 + ncore * x0_value

    n_elements = shape[0] * shape[1]
    atomic_add[shape[0], 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=shape[1])
    test_common.validate_cmp(dtype, x1, x1_ref)

# 3d
testlist = [
    (1,22,39),
    (27,1,39),
    (27,22,1),
    (1,1,23),
    (23,1,1),
    (1,23,1),
    (27,5,3),
    (2,29,4),
    (7,31,7),
    (3,5,8),
    (7,17,15),
    (25,5,16),
    (23,5,31),
    (7,11,32),
    (7,11,33),
    (2,3,255),
    (3,3,256),
    (3,2,257),
]

@pytest.mark.parametrize('shape',testlist)
@pytest.mark.parametrize('dtype',['float32', 'float16', 'bfloat16', 'int32', 'int16', 'int8'])
def test_atomic_add_3d(dtype, shape):
    ncore = 1
    split_size = shape[0] // ncore
    x0_value = 3
    x0 = torch.full(shape, x0_value, dtype = eval('torch.' + dtype)).npu()
    x1 = torch.full((split_size, shape[1], shape[2]), 2, dtype=eval('torch.' + dtype)).npu()
    y = torch.full((split_size, shape[1], shape[2]), -10, dtype=eval('torch.' + dtype)).npu()

    x1_ref = x1 + ncore * x0_value

    n_elements = shape[0] * shape [1] * shape[2]
    atomic_add[ncore, 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=shape[0] * shape[1] * shape[2])
    test_common.validate_cmp(dtype, x1, x1_ref)
