import torch

import pytest

from dlblas.symbolic_var import SymVar, Tensor
from dlblas.op_struct import violate_symbolic_constraints


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ['cuda'])
def test_no_violation(dtype, device):
    m, n, k = 1, 2, 3
    sm, sn, sk = SymVar('m'), SymVar('n'), SymVar('k')
    a = Tensor((sm, sk), dtype=dtype, device=device)
    b = Tensor((sk, sn), dtype=dtype, device=device)

    sym_shapes = [i.shape for i in (a, b)]
    concrete_shapes = [(m, k), (k, n)]
    assert not violate_symbolic_constraints(concrete_shapes, sym_shapes)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ['cuda'])
def test_same_name_no_violation(dtype, device):
    m, n, k = 1, 2, 3
    sm, sn, sk = SymVar('m'), SymVar('n'), SymVar('k')
    skk = SymVar('k')
    a = Tensor((sm, sk), dtype=dtype, device=device)
    b = Tensor((skk, sn), dtype=dtype, device=device)

    sym_shapes = [i.shape for i in (a, b)]
    concrete_shapes = [(m, k), (k, n)]
    assert not violate_symbolic_constraints(concrete_shapes, sym_shapes)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ['cuda'])
def test_violation(dtype, device):
    sm, sn, sk = SymVar('m'), SymVar('n'), SymVar('k')
    a = Tensor((sm, sk), dtype=dtype, device=device)
    b = Tensor((sk, sn), dtype=dtype, device=device)

    sym_shapes = [i.shape for i in (a, b)]
    concrete_shapes = [(2, 3), (4, 2)]
    assert violate_symbolic_constraints(concrete_shapes, sym_shapes)
