import triton
import pytest

from backend.ascend_autotune_runtime.ascend_kernel_autotuner import AutoTilingTuner
from backend.ascend_autotune_runtime.schedule_profiles import (
    COMPILE_MODE_KEY,
    COMPILE_MODE_VECTOR,
    WORKSPACE_CV_AGGRESSIVE_PROBE,
    WORKSPACE_CV_LOW_RESOURCE_PROBE,
    parse_compile_options_hint,
)
from backend.ascend_autotune_runtime.kernel_archetype import (
    OperatorKind,
    make_search_policy,
)
from backend.ascend_autotune_runtime.measurement_cache import SearchMeasureCache
from backend.ascend_autotune_runtime.tile_search_policy import (
    SearchParamsSpec,
    filter_shapes_by_ub_and_timing,
    parse_search_params_hint,
    propose_evolved_shape_configs,
    select_initial_percentile_shapes,
    select_top_shape_entries,
    stage1_children_per_round,
)
from backend.ascend_autotune_runtime.tile_shape_space import (
    expand_search_param_shapes,
    extract_shape,
    shape_key,
)


class _DummyStage1Tuner:
    _timing_sort_key = staticmethod(AutoTilingTuner._timing_sort_key)
    _is_finite_timing = AutoTilingTuner._is_finite_timing
    _apply_fixed_compile_profile = AutoTilingTuner._apply_fixed_compile_profile
    _apply_fixed_compile_profiles = AutoTilingTuner._apply_fixed_compile_profiles
    _format_search_error_log = AutoTilingTuner._format_search_error_log
    _run_stage1_probes_for_shapes = AutoTilingTuner._run_stage1_probes_for_shapes

    def __init__(self, batch_timings=None):
        self.search_params = SearchParamsSpec(
            enabled=True, params=["BLOCK_M", "BLOCK_N"]
        )
        self._search_ub_cache = {}
        self._search_measure_cache = SearchMeasureCache(
            self.search_params.params,
            self._search_ub_cache,
        )
        self.fixed_compile_options = {}
        self.batch_timings = list(batch_timings or [])
        self.batch_calls = []
        self.force_parallel_flags = []
        self.debug_messages = []

    def _search_debug(self, message):
        self.debug_messages.append(message)

    def _bench_search_configs(self, *args, configs, **kwargs):
        self.batch_calls.append(len(configs))
        self.force_parallel_flags.append(kwargs.get("force_parallel", False))
        if self.batch_timings:
            costs = self.batch_timings.pop(0)
            return {config: costs[index] for index, config in enumerate(configs)}, None
        return {config: float("inf") for config in configs}, None

    def _bench_stage1_fast_configs(self, *args, configs, **kwargs):
        self.batch_calls.append(len(configs))
        self.force_parallel_flags.append(True)
        if self.batch_timings:
            costs = self.batch_timings.pop(0)
            timings = {}
            errors = {}
            for index, config in enumerate(configs):
                cost = costs[index]
                if cost == float("inf"):
                    errors[config] = RuntimeError("benchmark returned inf")
                else:
                    timings[config] = cost
                    self._search_ub_cache[config] = 1024
            return timings, errors
        return {}, {
            config: RuntimeError("benchmark returned inf") for config in configs
        }

    def _bench_search_config(self, *args, config, **kwargs):
        raise AssertionError("Stage 1 probes should be batched by probe round")


def _index_shape(config, spec):
    shape = extract_shape(config, spec.params)
    return tuple(spec.values.index(shape[name]) for name in spec.params)


def _run_stage1_sampling_simulation(spec, fail_fn, time_fn, limit=5):
    all_configs = expand_search_param_shapes(
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}),
        spec,
    )
    initial = select_initial_percentile_shapes(all_configs, spec)
    successes = []
    failures = []
    observed = []
    seen = set()

    for config in initial:
        shape = extract_shape(config, spec.params)
        point = _index_shape(config, spec)
        seen.add(shape_key(shape, spec.params))
        observed.append(config)
        if fail_fn(point):
            failures.append(config)
        else:
            successes.append(
                {
                    "shape": shape,
                    "shape_config": config,
                    "time": time_fn(point),
                }
            )

    parents = select_top_shape_entries(
        successes,
        k=spec.shape_final_top_k,
        key_fn=lambda item: item["time"],
    )
    proposals = propose_evolved_shape_configs(
        parents=parents,
        successes=successes,
        failures=failures,
        observed=observed,
        all_configs=all_configs,
        spec=spec,
        seen_keys=seen,
        limit=limit,
    )
    return initial, successes, failures, parents, proposals


def test_search_params_rejects_unit_flag_search_knob():
    with pytest.raises(ValueError, match="enable_unit_flag_search"):
        parse_search_params_hint(
            {
                "params": ["BLOCK_M", "BLOCK_N"],
                "enable_unit_flag_search": False,
            }
        )

    with pytest.raises(ValueError, match="enable_unit_flag_search"):
        parse_search_params_hint(
            {
                "params": ["BLOCK_M", "BLOCK_N"],
                "enable_unit_flag_search": True,
            }
        )


def test_minimal_search_params_uses_current_defaults():
    spec = parse_search_params_hint({"params": ["BLOCK_M", "BLOCK_N"]})

    assert spec.enabled
    assert spec.params == ["BLOCK_M", "BLOCK_N"]
    assert spec.values == [16, 32, 64, 128, 256, 512, 1024, 2048]
    assert spec.shape_initial_percentiles == [0.20, 0.50, 0.80]
    assert spec.shape_refine_rounds == 2
    assert spec.shape_final_top_k == 3
    assert spec.seed_budget == 8
    assert spec.max_compile_trials_per_shape == 16
    assert spec.stage1_bench_warmup == 5
    assert spec.stage1_bench_rep == 10


def test_search_params_uses_operator_policy_for_compile_options():
    tuner = object.__new__(AutoTilingTuner)
    tuner.hints = {"search_params": {"params": ["BLOCK_M", "BLOCK_N"]}}
    tuner.operator_policy = make_search_policy(OperatorKind.VECTOR_AFFINE)

    AutoTilingTuner._infer_compile_options_hint_if_needed(tuner)

    assert tuner.hints["compile_options"] == "vector"


def test_search_params_keeps_runtime_enabled_with_fixed_compile_options():
    tuner = object.__new__(AutoTilingTuner)
    tuner.search_params = SearchParamsSpec(enabled=True, params=["NUM_CHUNKS"])
    tuner.compile_options = parse_compile_options_hint("vector")
    tuner.operator_policy = make_search_policy(OperatorKind.VECTOR_AFFINE)
    tuner.fixed_compile_options = {"num_stages": 3}

    assert AutoTilingTuner._use_search_params_runtime(tuner)

    profiles = AutoTilingTuner._apply_fixed_compile_profiles(
        tuner,
        [
            {
                COMPILE_MODE_KEY: COMPILE_MODE_VECTOR,
                "num_stages": 2,
                "enable_ubuf_saving": True,
            },
            {
                COMPILE_MODE_KEY: COMPILE_MODE_VECTOR,
                "num_stages": 1,
                "enable_ubuf_saving": True,
            },
        ],
    )

    assert profiles == [
        {
            COMPILE_MODE_KEY: COMPILE_MODE_VECTOR,
            "num_stages": 3,
            "enable_ubuf_saving": True,
        }
    ]


def test_failure_aware_sampling_moves_away_from_small_value_failures():
    spec = SearchParamsSpec(enabled=True, params=["BLOCK_M", "BLOCK_N"])

    _, successes, failures, parents, proposals = _run_stage1_sampling_simulation(
        spec,
        fail_fn=lambda point: point[0] <= 1 or point[1] <= 1,
        time_fn=lambda point: 1.0 + 0.1 * abs(point[0] - 4) + 0.1 * abs(point[1] - 4),
        limit=stage1_children_per_round(spec),
    )
    proposal_points = [_index_shape(config, spec) for config in proposals]

    assert successes
    assert failures
    assert parents
    assert proposal_points
    assert len(proposal_points) == 3 ** len(spec.params)
    assert len(set(proposal_points)) == len(proposal_points)
    assert all(point[0] > 1 and point[1] > 1 for point in proposal_points)


def test_resource_failure_sampling_backoffs_when_initial_points_are_too_large():
    spec = SearchParamsSpec(
        enabled=True,
        params=["NUM_CHUNKS"],
        values=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    )
    all_configs = expand_search_param_shapes(
        triton.Config({"NUM_CHUNKS": 64}),
        spec,
    )
    failed_values = [64, 512, 4096]
    failed_configs = [
        next(config for config in all_configs if config.kwargs["NUM_CHUNKS"] == value)
        for value in failed_values
    ]
    failure_observations = [
        {"shape": {"NUM_CHUNKS": 64}, "resource_ratio": 14680064 / 1572864},
        {"shape": {"NUM_CHUNKS": 512}, "resource_ratio": 117440512 / 1572864},
        {"shape": {"NUM_CHUNKS": 4096}, "resource_ratio": 512.0},
    ]
    seen = {
        shape_key(extract_shape(config, spec.params), spec.params)
        for config in failed_configs
    }

    proposals = propose_evolved_shape_configs(
        parents=[],
        successes=[],
        failures=failed_configs,
        failure_observations=failure_observations,
        observed=failed_configs,
        all_configs=all_configs,
        spec=spec,
        seen_keys=seen,
        limit=4,
    )
    proposed_values = [config.kwargs["NUM_CHUNKS"] for config in proposals]

    assert proposed_values
    assert len(proposed_values) == len(set(proposed_values))
    assert not (set(proposed_values) & set(failed_values))
    assert min(proposed_values) <= 4
    assert max(proposed_values) <= 32


def test_stage1_probe_failure_falls_back_until_success():
    # First probe fails, second succeeds, third must not run.
    tuner = _DummyStage1Tuner(
        batch_timings=[
            [float("inf")],
            [0.25],
        ]
    )
    shape_config = triton.Config({"BLOCK_M": 128, "BLOCK_N": 512})

    result = AutoTilingTuner._run_stage1_probe_for_shape(
        tuner, shape_config=shape_config
    )

    assert result is not None
    assert result["time"] == 0.25
    assert result["profile"] == WORKSPACE_CV_LOW_RESOURCE_PROBE
    assert tuner.batch_calls == [1, 1]


def test_stage1_probe_rounds_only_retry_failed_shapes():
    # 2 shapes: A passes first probe, B retries and passes second probe.
    tuner = _DummyStage1Tuner(
        batch_timings=[
            [0.30, float("inf")],
            [0.25],
        ]
    )
    shape_configs = [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 512}),
    ]

    results, failures = AutoTilingTuner._run_stage1_probes_for_shapes(
        tuner, shape_configs=shape_configs
    )

    assert not failures
    assert len(results) == 2
    assert results[0]["shape"] == {"BLOCK_M": 128, "BLOCK_N": 256}
    assert results[0]["time"] == 0.30
    assert results[0]["profile"] == WORKSPACE_CV_AGGRESSIVE_PROBE
    assert results[1]["shape"] == {"BLOCK_M": 128, "BLOCK_N": 512}
    assert results[1]["time"] == 0.25
    assert results[1]["profile"] == WORKSPACE_CV_LOW_RESOURCE_PROBE
    assert tuner.batch_calls == [2, 1]
    assert tuner.force_parallel_flags == [True, True]


def _timing_key(value):
    return float(value)


def test_ub_filter_drops_low_ub_and_slow_entries():
    entries = [
        {"shape": {"BLOCK_M": 128, "BLOCK_N": 128}, "ub": 10000, "time": 0.10},
        {"shape": {"BLOCK_M": 64, "BLOCK_N": 64}, "ub": 4000, "time": 0.20},
        {"shape": {"BLOCK_M": 32, "BLOCK_N": 32}, "ub": 9000, "time": 0.15},
    ]
    kept = filter_shapes_by_ub_and_timing(entries, timing_sort_key=_timing_key)
    shapes = [entry["shape"] for entry in kept]
    # max_ub=10000, ub_floor=5000; min_time=0.10, timing_ceiling=0.14.
    # (64,64) has ub=4000<5000 AND time=0.20>0.14 -> dropped.
    assert {"BLOCK_M": 32, "BLOCK_N": 32} in shapes
    assert {"BLOCK_M": 128, "BLOCK_N": 128} in shapes
    assert {"BLOCK_M": 64, "BLOCK_N": 64} not in shapes
