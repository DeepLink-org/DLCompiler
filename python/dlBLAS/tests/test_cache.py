import torch

import pytest

from dlblas import get_op, get_list_op_names
from dlblas.op_registry import op_registry
from dlblas.cache import Cache


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ['cuda'])
def test_gen_key(dtype, device):
    cache = Cache()

    a = torch.randn(
        (1, 2),
        dtype=dtype,
        device=device,
    )
    b = torch.randn(
        (2, 1),
        dtype=dtype,
        device=device,
    )
    activation = 'leaky_relu'
    op = 'matmul'

    key = cache.gen_key(op, (a, b, activation))
    assert key == f'{op}-0:f16_1x2-1:f16_2x1-2:leaky_relu-{device}'


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ['cuda'])
def test_cache_hit(dtype, device):
    a = torch.randn(
        (3, 5),
        dtype=dtype,
        device=device,
    )
    b = torch.randn(
        (5, 2),
        dtype=dtype,
        device=device,
    )

    # args = (a, b)
    activation = 'leaky_relu'
    args = (a, b, activation)

    # import pdb; pdb.set_trace()
    dlblas_op = get_op('matmul', args)

    # cache hit TODO considering to add a counter utility to counter every cache hit objects
    c = torch.randn(
        (3, 5),
        dtype=dtype,
        device=device,
    )
    d = torch.randn(
        (5, 2),
        dtype=dtype,
        device=device,
    )
    # args = (a, b)
    activation = 'leaky_relu'
    args = (a, b, activation)
    dlblas_op2 = get_op('matmul', args)

    assert torch.allclose(dlblas_op2(c, d), dlblas_op(c, d))
