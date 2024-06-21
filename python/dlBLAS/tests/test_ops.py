import torch
import triton
import triton.language as tl

import pytest

from dlblas import get_op, OpImpl, OpParams



@pytest.mark.parametrize("m", [32, 64, 128])
@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("k", [4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_mm(m, n, k, dtype):
    torch.manual_seed(20)

    x = torch.randn(
        (m, k),
        dtype=dtype,
        device='cuda',

    )
    args = OpParams()
    
    out = get_op('mm', args)

    assert out is None


    # compare
    # tol = {}
    # torch.testing.assert_close(o, ref_o, **tol)
