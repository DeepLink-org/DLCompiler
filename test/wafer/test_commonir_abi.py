"""Keep the shared CommonIR loader independent of Wafer's five-value ABI."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    "target,mix_mode", [("mlu", False), ("maca", False), ("ascend", True)]
)
def test_commonir_legacy_loader_contract(monkeypatch, target, mix_mode):
    calls = []
    handle = object()

    def load_binary(name, binary, shared, device, *, mix_mode):
        calls.append((name, binary, shared, device, mix_mode))
        return handle, 123, 4, 0

    driver = SimpleNamespace(
        get_current_device=lambda: 0,
        launcher_cls=Mock(return_value=object()),
        utils=SimpleNamespace(load_binary=load_binary),
    )
    # Import the real loader while isolating the unavailable vendor drivers.
    package = ModuleType("_wafer_commonir_abi_test")
    package.__path__ = []
    dependency = ModuleType(package.__name__ + ".backend")
    dependency.commonir_backend = SimpleNamespace(get_driver=lambda: driver)
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, dependency.__name__, dependency)
    path = Path(__file__).resolve().parents[2] / "backend/commonir/compiler.py"
    spec = importlib.util.spec_from_file_location(package.__name__ + ".compiler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = module.CompiledKernel.__new__(module.CompiledKernel)
    kernel.module = None
    kernel.name = target + "_kernel"
    kernel.kernel = b"device binary"
    kernel.src = object()
    kernel.metadata = SimpleNamespace(shared=32, mix_mode=mix_mode)
    kernel._init_handles()
    kernel.launch_metadata((1, 1, 1), None)
    assert calls == [(kernel.name, kernel.kernel, 32, 0, mix_mode)]
    assert kernel.module is handle
    assert kernel.function == 123
    driver.launcher_cls.assert_called_once_with(kernel.src, kernel.metadata)
