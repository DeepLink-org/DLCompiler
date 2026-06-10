from types import MethodType

import pytest
import triton
from triton.runtime.autotuner import Config
from backend.ascend_autotune_runtime.autotuner import AutoTilingTuner
from triton.compiler.errors import CompilationError


@pytest.fixture(autouse=True)
def _patch_mlir_error(monkeypatch):
    """MLIRCompilationError does not exist in stock triton 3.5; alias it."""
    monkeypatch.setattr(
        "triton.compiler.errors.MLIRCompilationError", CompilationError, raising=False
    )


def _make_tuner(do_bench):
    tuner = object.__new__(AutoTilingTuner)
    tuner.compile_parallel = False
    tuner.do_bench = do_bench
    tuner.user_defined_do_bench = True

    def _make_kernel_call(self, *args, config, **meta):
        def kernel_call(warmup):
            return None

        return kernel_call

    tuner._make_kernel_call = MethodType(_make_kernel_call, tuner)
    return tuner


def test_batch_bench_uses_do_bench_npu_with_user_do_bench(monkeypatch):
    calls = {"do_bench_npu": 0}

    def _do_bench(fn, quantiles):
        raise AssertionError(
            "user do_bench should not be used by Ascend autotune runtime"
        )

    def _do_bench_npu(funcs, clear_l2_cache=False):
        calls["do_bench_npu"] += 1
        assert clear_l2_cache is False
        assert len(funcs) == 2
        return [3.0, 4.0]

    tuner = _make_tuner(_do_bench)
    cfg0 = Config({"ID": 0})
    cfg1 = Config({"ID": 1})
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    monkeypatch.setattr("backend.testing.do_bench_npu", _do_bench_npu)

    result = tuner._batch_bench(configs=[cfg0, cfg1])

    assert calls["do_bench_npu"] == 1
    assert result[cfg0] == 3.0
    assert result[cfg1] == 4.0


def test_batch_bench_defaults_to_do_bench_npu_without_user_do_bench(monkeypatch):
    def _do_bench(fn, quantiles):
        raise AssertionError(
            "self.do_bench should not be used when no user do_bench is provided"
        )

    calls = {"do_bench_npu": 0}

    def _do_bench_npu(funcs, clear_l2_cache=False):
        calls["do_bench_npu"] += 1
        assert len(funcs) == 2
        return [1.0, 2.0]

    tuner = _make_tuner(_do_bench)
    tuner.user_defined_do_bench = False
    cfg0 = Config({"ID": 0})
    cfg1 = Config({"ID": 1})
    monkeypatch.delenv("TRITON_BENCH_METHOD", raising=False)
    monkeypatch.setattr("backend.testing.do_bench_npu", _do_bench_npu)

    result = tuner._batch_bench(configs=[cfg0, cfg1])

    assert calls["do_bench_npu"] == 1
    assert result[cfg0] == 1.0
    assert result[cfg1] == 2.0


@pytest.mark.parametrize("method", ["default", "triton", "do_bench"])
def test_batch_bench_ignores_triton_bench_method(monkeypatch, method):
    calls = {"do_bench_npu": 0}

    def _do_bench(fn, quantiles):
        raise AssertionError(
            "TRITON_BENCH_METHOD should not switch Ascend autotune runtime away from do_bench_npu"
        )

    def _do_bench_npu(funcs, clear_l2_cache=False):
        calls["do_bench_npu"] += 1
        assert len(funcs) == 2
        return [4.0, 5.0]

    tuner = _make_tuner(_do_bench)
    tuner.user_defined_do_bench = False
    cfg0 = Config({"ID": 0})
    cfg1 = Config({"ID": 1})
    monkeypatch.setenv("TRITON_BENCH_METHOD", method)
    monkeypatch.setattr("backend.testing.do_bench_npu", _do_bench_npu)

    result = tuner._batch_bench(configs=[cfg0, cfg1])

    assert calls["do_bench_npu"] == 1
    assert result[cfg0] == 4.0
    assert result[cfg1] == 5.0


def test_batch_bench_single_config_uses_do_bench_npu(monkeypatch):
    calls = {"do_bench_npu": 0}

    def _do_bench(fn, quantiles):
        raise AssertionError("single config should still use do_bench_npu")

    def _do_bench_npu(funcs, clear_l2_cache=False):
        calls["do_bench_npu"] += 1
        assert len(funcs) == 1
        return [5.0]

    tuner = _make_tuner(_do_bench)
    tuner.user_defined_do_bench = False
    cfg = Config({"ID": 0})
    monkeypatch.delenv("TRITON_BENCH_METHOD", raising=False)
    monkeypatch.setattr("backend.testing.do_bench_npu", _do_bench_npu)

    result = tuner._batch_bench(configs=[cfg])

    assert calls["do_bench_npu"] == 1
    assert result[cfg] == 5.0


def test_batch_bench_do_bench_npu_timing_count_mismatch(monkeypatch):
    def _do_bench(fn, quantiles):
        raise AssertionError(
            "self.do_bench should not be used when default NPU benchmark is selected"
        )

    def _do_bench_npu(funcs, clear_l2_cache=False):
        assert len(funcs) == 2
        return [1.0]

    tuner = _make_tuner(_do_bench)
    tuner.user_defined_do_bench = False
    cfg0 = Config({"ID": 0})
    cfg1 = Config({"ID": 1})
    monkeypatch.delenv("TRITON_BENCH_METHOD", raising=False)
    monkeypatch.setattr("backend.testing.do_bench_npu", _do_bench_npu)

    with pytest.raises(RuntimeError, match="mismatched timing count"):
        tuner._batch_bench(configs=[cfg0, cfg1])


def test_autotilingtuner_accepts_user_defined_do_bench():
    marker = {"called": False}

    def _do_bench(fn, quantiles):
        marker["called"] = True
        return (0.0, 0.0, 0.0)

    def _dummy_kernel():
        return None

    _dummy_kernel.arg_names = []

    tuner = AutoTilingTuner(
        _dummy_kernel,
        [],
        [Config({})],
        [],
        None,
        None,
        do_bench=_do_bench,
        hints={"compile_options": False},
    )

    assert tuner.user_defined_do_bench is True
    assert marker["called"] is False


def test_ascend_autotune_decorator_accepts_do_bench(monkeypatch):
    import backend.ascend_autotune_runtime.autotuner as ascend_autotuner

    captured = {}

    class DummyAutoTilingTuner:
        def __init__(self, *args, **kwargs):
            captured["do_bench"] = kwargs.get("do_bench")

    monkeypatch.setattr(ascend_autotuner, "AutoTilingTuner", DummyAutoTilingTuner)

    def _dummy_kernel():
        return None

    _dummy_kernel.arg_names = []
    my_do_bench = lambda kernel_call, quantiles: (0.0, 0.0, 0.0)

    ascend_autotuner.autotune(configs=[object()], key=[], do_bench=my_do_bench)(
        _dummy_kernel
    )

    assert captured["do_bench"] is my_do_bench
