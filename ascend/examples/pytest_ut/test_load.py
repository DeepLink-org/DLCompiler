import triton
import triton.language as tl
import numpy as np
import torch
import pytest
import test_common

# eg: pytest -v test.py::test_add
#############################

@triton.jit
def triton_load_store(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tmp2 = tmp0
        tl.store(out_ptr0 + (xindex), tmp2, xmask)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (2, 4096, 8), 2, 32768, 1024],
                             ['float16', (2, 4096, 8), 2, 32768, 1024],
                             ['int8', (2, 4096, 8), 2, 32768, 1024],
                             ['float32', (8, 8, 4), 2, 128, 64],
                             ['float16', (8, 8, 4), 2, 128, 64],
                             ['int8', (8, 8, 4), 2, 128, 64],
                         ]
                         )
def test_load_store(param_list):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    y_ref = x0
    y_cal = test_common.generate_tensor(shape, dtype).npu()
    triton_load_store[ncore, 1, 1](x0, y_cal, x0.numel(), xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)


