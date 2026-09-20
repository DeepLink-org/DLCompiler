import pytest
import torch
import torch_txda  # noqa: F401
import triton
import triton.language as tl
import numpy as np
from numpy.random import RandomState

from triton._internal_testing import (
    integral_dtypes,
    int_dtypes,
    str_to_triton_dtype,
    uint_dtypes,
    float_dtypes,
    float_dtypes_with_bfloat16,
    dtypes,
    dtypes_with_bfloat16,
    is_cuda,
    is_interpreter,
    is_hopper,
    is_hip,
    is_hip_cdna,
    is_hip_cdna2,
    is_hip_cdna3,
    is_hip_cdna4,
    is_xpu,
    get_arch,
    torch_float8_dtypes,
    torch_dtypes,
    numpy_random,
    to_triton,
    torch_dtype_name,
    to_numpy,
)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", ['int32'])
def test_umulhi(dtype_str, device):

    @triton.jit
    def kernel(X, Y, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        y = tl.load(Y + offs)
        z = tl.umulhi(x, y)
        tl.store(Z + tl.arange(0, N), z)

    def umulhi32(a, b):
        # Convert to 64-bit unsigned integers to prevent overflow
        a_64 = a.astype(np.int64)
        b_64 = b.astype(np.int64)

        # Perform the multiplication in 64-bit
        product_64 = a_64 * b_64

        # Shift right by 32 bits to get the high part of the product
        result_high_32 = product_64 >> 32
        return result_high_32

    rs = RandomState(17)
    N = 128
    x = numpy_random((N, ), dtype_str=dtype_str, rs=rs, low=0)
    x_tri = to_triton(x, device=device)
    y = numpy_random((N, ), dtype_str=dtype_str, rs=rs, low=0)
    y_tri = to_triton(y, device=device)
    z_tri = torch.zeros_like(x_tri)
    x_tri_txda = x_tri.to("txda")
    y_tri_txda = y_tri.to("txda")
    z_tri_txda = z_tri.to("txda")
    kernel[(1, )](x_tri_txda, y_tri_txda, z_tri_txda, N=N)
    with torch.no_grad():
        z_tri.copy_(z_tri_txda.cpu())

    z_ref = umulhi32(x, y)
    np.testing.assert_equal(z_ref, to_numpy(z_tri))


if __name__ == "__main__":
    # Run the test
    # pytest.main([__file__, "-v", "-s", "--tb=short"])
    test_umulhi('int32', 'cpu')
