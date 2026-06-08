"""Benchmark strategy for Ascend autotuning."""

from __future__ import annotations

from typing import Mapping


class NpuProfilerBenchStrategy:

    def bench(self, run_fns: Mapping):
        from ..testing import do_bench_npu

        costs = do_bench_npu(list(run_fns.values()), clear_l2_cache=False)
        if not isinstance(costs, (list, tuple)):
            raise RuntimeError(
                "do_bench_npu must return one timing per autotune config "
                f"when benchmarking {len(run_fns)} configs, got {type(costs).__name__}."
            )
        if len(costs) != len(run_fns):
            raise RuntimeError(
                "do_bench_npu returned mismatched timing count: "
                f"expected {len(run_fns)}, got {len(costs)}."
            )
        return {config: cost for config, cost in zip(run_fns.keys(), costs)}


def select_benchmark_strategy(*_unused):
    return NpuProfilerBenchStrategy()
