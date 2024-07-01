import torch

import pytest

from dlblas import (get_op, get_list_op_names, OpImpl, OpParams)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ['cuda'])
def test_compile_key(dtype, device):
    '''this tests how we can get the generated compile key
        it is a hash of the input args
    
        a potential solution is:
            https://github.com/triton-lang/triton/blob/8e96b71b1b47a5d09f1cfb1826a16178f58dbef0/python/test/unit/runtime/test_cache.py#L515
        
        but it seems too complex for our purpose, for now we simply fetch from the `cache` field
    '''

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

    # import pdb; pdb.set_trace()

    # args = (a, b)
    activation = 'leaky_relu'
    args = (a, b, activation)

    # import pdb; pdb.set_trace()
    dlblas_op = get_op('matmul', args)
