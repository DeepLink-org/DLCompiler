import torch
import triton
import triton.language as tl

import pytest

from dlblas import get_op, get_list_op_names, OpImpl, OpParams


@pytest.mark.parametrize("m", [32, 64, 128])
@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("k", [4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ['cuda'])
def test_mm(m, n, k, dtype, device):
    torch.manual_seed(20)

    op_list = get_list_op_names()
    assert 'matmul' in op_list

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
    args = (a, b)

    dlblas_op = get_op('matmul', args)
    assert isinstance(dlblas_op, OpImpl)

    # compare
    out = dlblas_op(a, b)
    ref_out = a @ b

    tol = {
        'atol': 1.0,
    }
    assert torch.allclose(out, ref_out, **tol)
