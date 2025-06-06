import triton
import triton.language as tl
import torch
import pytest
import test_common
from test_common import TestUtils
import numpy as np

# inp the two 32 bit signed integers.
@triton.jit
def umulhi_kernel(X, Y, Z, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    z = tl.umulhi(x, y)
    tl.store(Z + tl.arange(0, N), z)


# accuracy reference
def umulhi32(a, b):
    a_64 = a.astype(np.int64)
    b_64 = b.astype(np.int64)
    product_64 = a_64 * b_64
    # get the high part
    result_high_32 = product_64 >> 32
    return result_high_32.astype(np.int32)


@pytest.mark.parametrize('dtype', ['int32'])
@pytest.mark.parametrize('shape', TestUtils.full_shape)
def test_case2(dtype, shape):
    N = shape[0]
    dtypes = eval('torch.' + dtype)
    x = torch.randint(low=0, high=2000, size=(N,), dtype=dtypes)
    y = torch.randint(low=0, high=2000, size=(N,), dtype=dtypes)
    xx = x.npu()
    yy = y.npu()
    z_tri = torch.zeros(size=(N,), dtype=dtypes).npu()
    umulhi_kernel[(1,)](xx, yy, z_tri, N=N)

    xxx = x.numpy()
    yyy = y.numpy()
    z_ref = umulhi32(xxx, yyy)
    z_ref1 = torch.from_numpy(z_ref).npu()
    torch.equal(z_tri, z_ref1)

invalid_types = [
    'int8',
    'int16',
    'int64',
    'float16',
    'float32',
    'bfloat16',
    'bool',
]
@pytest.mark.parametrize("dtype", invalid_types)
@test_common.raises_with_match(triton.compiler.errors.CompilationError, "Expected dtype")
def test_umulhi_invalid_dtype_case(dtype):
    x0 = test_common.generate_tensor((1,), dtype).npu()
    x1 = test_common.generate_tensor((1,), dtype).npu()

    y_cal = torch.zeros((1,), dtype=eval('torch.' + dtype)).npu()
    umulhi_kernel[(1,)](x0, x1, y_cal, 1)
