"""Benchmark strategy for Ascend autotuning."""

from __future__ import annotations

from typing import Mapping, Optional


def ub_bytes_of(compiled_kernel) -> Optional[int]:
    """UB allocation in bytes from compile-time metadata.

    Returns None when the kernel was compiled without ``TRITON_MEMORY_DISPLAY=1``
    / ``--enable-memory-display=true``, because the backend then leaves
    ``required_ub_bits`` at its 0 default. The Ascend profiler does not expose
    static UB size at runtime, so compile metadata is the source of truth.
    """
    bits = (
        getattr(getattr(compiled_kernel, "metadata", None), "required_ub_bits", 0) or 0
    )
    return bits // 8 if bits else None


class NpuProfilerBenchStrategy:
    def bench(self, run_fns: Mapping):
        from ..testing import do_bench_npu

        costs = do_bench_npu(list(run_fns.values()), clear_l2_cache=False)
        if len(run_fns) == 1 and isinstance(costs, (int, float)):
            config = next(iter(run_fns))
            return {config: costs}
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
