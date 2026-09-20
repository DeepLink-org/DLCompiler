"""Ascend-derived cases use TXDA tensors and their original deterministic inputs."""
import pytest


@pytest.fixture(autouse=True)
def require_device(wafer_device):
    import numpy as np
    import torch
    # The removed harness reset both generators before every test. Keep its
    # input baseline explicitly; test-local seeds can still override it.
    np.random.seed(0)
    torch.manual_seed(0)
    return wafer_device
