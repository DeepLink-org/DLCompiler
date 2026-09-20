"""Native Wafer examples retain their original deterministic input baseline."""
import pytest


@pytest.fixture(autouse=True)
def require_device(wafer_device):
    import numpy as np
    import torch
    # Preserve initialization formerly supplied by _wafer_harness.
    np.random.seed(0)
    torch.manual_seed(0)
    return wafer_device
