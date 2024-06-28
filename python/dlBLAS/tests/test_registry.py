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