from backend.ascend_autotune_runtime.measurement_strategy import (
    NpuProfilerBenchStrategy,
)


def test_npu_bench_strategy_accepts_single_scalar_timing(monkeypatch):
    import backend.testing as testing

    config = object()

    def fake_do_bench_npu(funcs, clear_l2_cache=False):
        assert len(funcs) == 1
        assert clear_l2_cache is False
        return 0.123

    monkeypatch.setattr(testing, "do_bench_npu", fake_do_bench_npu)

    assert NpuProfilerBenchStrategy().bench({config: lambda: None}) == {config: 0.123}


def test_npu_bench_strategy_accepts_multi_config_timing(monkeypatch):
    import backend.testing as testing

    configs = [object(), object()]

    def fake_do_bench_npu(funcs, clear_l2_cache=False):
        assert len(funcs) == 2
        assert clear_l2_cache is False
        return [0.2, 0.1]

    monkeypatch.setattr(testing, "do_bench_npu", fake_do_bench_npu)

    assert NpuProfilerBenchStrategy().bench(
        {configs[0]: lambda: None, configs[1]: lambda: None}
    ) == {
        configs[0]: 0.2,
        configs[1]: 0.1,
    }
