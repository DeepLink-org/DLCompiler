
import triton
import triton.language as tl
import torch
import pytest
import test_common
from test_common import TestUtils

@triton.jit
def triton_test_fn_atomic_max_dma(in_ptr0, out_ptr0, out_ptr1, n_elements : tl.constexpr, BLOCK_SIZE : tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_max(out_ptr0 + (x1), tmp0, xmask)

# torch.max do not support int
@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', ['float32', 'int32', 'int8', 'int16', 'bfloat16', 'float16'])
def test_atomic_max(dtype, shape):
    ncore = 1
    block_size = shape[0] * shape[1] / ncore
    split_size = shape[0] // ncore
    # old size: (32768, 256)
    # tensor of (1024, 256) is too big, and it will lead to failure in the backend
    # so here we make it smaller
    x0 = test_common.generate_tensor(shape, dtype)
    x1 = test_common.generate_tensor(shape, dtype)
    y = test_common.generate_tensor(shape, dtype)

    x1_ref = torch.maximum(x0, x1)
    x0 = x0.npu()
    x1 = x1.npu()
    y = y.npu()

    n_elements = shape[0] * shape[1]
    triton_test_fn_atomic_max_dma[shape[0], 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=shape[1])
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

@pytest.mark.parametrize('shape', testlist)
@pytest.mark.parametrize('dtype', ['float32', 'int32', 'int8', 'int16', 'bfloat16', 'float16'])
def test_atomic_max_3d(dtype, shape):
    ncore = 1
    split_size = shape[0] // ncore
    x0 = test_common.generate_tensor(shape, dtype)
    x1 = test_common.generate_tensor(shape, dtype)
    y = test_common.generate_tensor(shape, dtype)

    x1_ref = torch.maximum(x0, x1)
    x0 = x0.npu()
    x1 = x1.npu()
    y = y.npu()

    n_elements = shape[0] * shape[1] * shape[2]
    triton_test_fn_atomic_max_dma[ncore, 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=split_size * shape[1] * shape[2])
    test_common.validate_cmp(dtype, x1, x1_ref)
# if __name__ == "__main__":
#     test_atomic_max(['int32', (8, 8), 2])