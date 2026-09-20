import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def source_driver(wafer_modules):
    module = importlib.import_module("_wafer_under_test.driver")
    driver = object.__new__(module.DICPDriver)
    driver.target = "wafer"
    return driver


def test_wafer_benchmark_uses_runtime_and_preserves_other_cache_policies(monkeypatch, wafer_modules):
    driver = source_driver(wafer_modules)
    runtime = SimpleNamespace(Event=Mock(), synchronize=Mock())
    monkeypatch.setattr(wafer_modules[2], "get_runtime", lambda: runtime)
    assert driver.get_device_interface() is runtime
    driver.clear_cache(driver.get_empty_cache_for_benchmark())
    # DICP shares clear_cache across backends; tensor-based policies still flush.
    tensor_cache = SimpleNamespace(zero_=Mock())
    driver.clear_cache(tensor_cache)
    tensor_cache.zero_.assert_called_once_with()


def test_raw_sdk_runtime_cannot_silently_provide_benchmark_timing(monkeypatch, wafer_modules):
    driver = source_driver(wafer_modules)
    monkeypatch.setattr(wafer_modules[2], "get_runtime", lambda: SimpleNamespace(current_device=lambda: 0))
    with pytest.raises(RuntimeError, match="torch_txda Event"):
        driver.get_device_interface()
