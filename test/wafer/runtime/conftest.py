"""Native math tests use ordinary TXDA tensors and the production launcher."""
import pytest


@pytest.fixture(autouse=True)
def require_device(wafer_device):
    return wafer_device
