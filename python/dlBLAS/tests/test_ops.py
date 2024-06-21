import torch
import triton
import triton.language as tl

import pytest

from dlblas import get_op, OpImpl, OpParams


@pytest.mark.parametrize("m", [32, 64, 128])
@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("k", [4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("device", ['cuda'])
def test_mm(m, n, k, dtype, device):
    torch.manual_seed(20)

    a = torch.randn(
        (m, k),
        dtype=dtype,
        device=device,
    )
    b = torch.randn(
        (k, n),
        dtype=dtype,
        device=device,
    )
    args = OpParams()

    out = get_op('mm', args)

    assert out is None

    ref_out = a @ b

    # compare
    # tol = {}
    # torch.testing.assert_close(o, ref_o, **tol)
