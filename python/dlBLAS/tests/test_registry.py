import re

import torch
import triton
import triton.language as tl

import pytest

import dlblas
from dlblas import get_op, get_list_op_names, get_args_from_op_name


def test_op_registry():
    torch.manual_seed(20)

    # test matmul kernel is registed when import
    op_list = get_list_op_names()
    assert 'matmul' in op_list


@pytest.mark.skip(reason="no need")
def test_regex():
    src = """
@triton.jit
def whatever(a, b, c):
    hahahah
    
# register
name = 'matmul'
for dtype in [torch.float16, torch.float32]:
    for activation in ["", "leaky_relu"]:
        # for now, epilogue is not added to op name
        for device in ['cuda']:
            m, n, k = SymVar('m'), SymVar('n'), SymVar('k')
            # we dont' actually allocate tensor
            a = Tensor((m, k), dtype=dtype, device=device)
            b = Tensor((k, n), dtype=dtype, device=device)

            # NOTE: the underlying kernel is the same jit'ed function, but Triton
            # will dispatch to different kernels based on the input params
            #
            # why do we still need another dispatch layer in op_registry?
            # because e.g. matmul may have different Triton implemetation...
            #
            if activation == '':
                register_dlblas_op(name, (a, b), call, bench_fn, matmul_kernel)
            else:
                register_dlblas_op(name, (a, b, activation), call, bench_fn, matmul_kernel)
"""
    replaced_data = re.sub(r'register_dlblas_op.*?(?=\n|$)',
                           'pass',
                           src,
                           flags=re.MULTILINE)

    assert "register_dlblas_op" not in replaced_data

    src2 = """
name = 'matmul'
for dtype in [torch.float16, torch.float32]:
    for activation in ["", "leaky_relu"]:
        # for now, epilogue is not added to op name
        for device in ['cuda']:
            m, n, k = SymVar('m'), SymVar('n'), SymVar('k')
            a = Tensor((m, k), dtype=dtype, device=device)
            b = Tensor((k, n), dtype=dtype, device=device)
            if activation == '':
                register_dlblas_op(name, (a, b), call, bench_fn, matmul_kernel) # what about now 
            else:
                register_dlblas_op(name, (a, b, activation), call, bench_fn, matmul_kernel)
"""
    replace2 = re.sub(r'register_dlblas_op.*?(?=\n|$)',
                      'pass',
                      src2,
                      flags=re.MULTILINE)

    assert "register_dlblas_op" not in replace2
