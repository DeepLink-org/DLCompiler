import torch

import pytest

from dlblas.symbolic_var import SymVar, Tensor


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ['cuda'])
def test_symint_as_shape(dtype, device):
    pass
