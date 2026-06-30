from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from triton.runtime.autotuner import Config

from .tile_acquisition_model import ShapeAcquisition
from .tile_shape_space import (
    DiscreteShapeSpace,
    effective_search_value_map,
    extract_shape,
    shape_key,
)


DEFAULT_SEARCH_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048]
DEFAULT_NO_DOT_SEARCH_VALUES = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
]
DEFAULT_INITIAL_PERCENTILES = [0.20, 0.50, 0.80]
SMALL_VALUE_PERCENTILES = [0.30, 0.70]
SMALL_VALUE_COUNT_THRESHOLD = 6
DEFAULT_MAX_STAGE1_CHILDREN_PER_ROUND = 16
DEFAULT_STAGE1_TOTAL_BUDGET_CAP = 18

UB_FLOOR_RATIO = 0.5
TIMING_CEILING_RATIO = 1.4


@dataclass
class SearchParamsSpec:
    enabled: bool = False
    params: List[str] = field(default_factory=list)
    values: List[int] = field(default_factory=lambda: list(DEFAULT_SEARCH_VALUES))
    shape_initial_percentiles: List[float] = field(
        default_factory=lambda: list(DEFAULT_INITIAL_PERCENTILES)
    )
    shape_refine_rounds: int = 2
    shape_final_top_k: int = 3
    seed_budget: int = 8
    max_compile_trials_per_shape: int = 16
    neighbors_per_step: int = 2
    compile_initial_temperature: float = 0.20
    compile_cooling: float = 0.85
    candidate_pool_per_shape: int = 3
    debug: bool = True
    reference_fn: Any = None
    bench_warmup: Optional[int] = None
    bench_active: Optional[int] = None
    stage1_bench_warmup: int = 5
    stage1_bench_rep: int = 10
    max_stage1_children_per_round: int = DEFAULT_MAX_STAGE1_CHILDREN_PER_ROUND
    values_were_default: bool = True


def _as_list(name: str, value: Any) -> List[Any]:
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if not values:
        raise ValueError(f"search_params '{name}' must not be empty")
    return values


def parse_search_params_hint(hint: Any) -> SearchParamsSpec:
    if hint is None or hint is False:
        return SearchParamsSpec(enabled=False)
    if not isinstance(hint, dict):
        raise TypeError("hints['search_params'] must be a dict")

    raw = dict(hint)
    params = _as_list("params", raw.pop("params", []))
    if not all(isinstance(name, str) and name for name in params):
        raise ValueError("search_params params must be non-empty strings")

    values_were_default = "values" not in raw
    values = _as_list("values", raw.pop("values", DEFAULT_SEARCH_VALUES))
    if not all(isinstance(value, int) and value > 0 for value in values):
        raise ValueError("search_params values must be positive integers")

    percentiles = _as_list(
        "shape_initial_percentiles",
        raw.pop("shape_initial_percentiles", DEFAULT_INITIAL_PERCENTILES),
    )
    if not all(
        isinstance(value, (int, float)) and 0 <= value <= 1 for value in percentiles
    ):
        raise ValueError("shape_initial_percentiles must be in [0, 1]")

    spec = SearchParamsSpec(
        enabled=True,
        params=params,
        values=sorted(set(values)),
        shape_initial_percentiles=[float(value) for value in percentiles],
        shape_refine_rounds=int(raw.pop("shape_refine_rounds", 2)),
        shape_final_top_k=int(raw.pop("shape_final_top_k", 3)),
        seed_budget=int(raw.pop("seed_budget", 8)),
        max_compile_trials_per_shape=int(raw.pop("max_compile_trials_per_shape", 16)),
        neighbors_per_step=int(raw.pop("neighbors_per_step", 2)),
        compile_initial_temperature=float(raw.pop("compile_initial_temperature", 0.20)),
        compile_cooling=float(raw.pop("compile_cooling", 0.85)),
        candidate_pool_per_shape=int(raw.pop("candidate_pool_per_shape", 3)),
        debug=bool(raw.pop("debug", True)),
        reference_fn=raw.pop("reference_fn", None),
        bench_warmup=raw.pop("bench_warmup", None),
        bench_active=raw.pop("bench_active", None),
        stage1_bench_warmup=int(raw.pop("stage1_bench_warmup", 5)),
        stage1_bench_rep=int(raw.pop("stage1_bench_rep", 10)),
        max_stage1_children_per_round=int(
            raw.pop(
                "max_stage1_children_per_round",
                DEFAULT_MAX_STAGE1_CHILDREN_PER_ROUND,
            )
        ),
        values_were_default=values_were_default,
    )
    if raw:
        raise ValueError(f"Unknown search_params option(s): {sorted(raw)}")
    if spec.shape_refine_rounds < 0:
        raise ValueError("shape_refine_rounds must be >= 0")
    if spec.shape_final_top_k <= 0:
        raise ValueError("shape_final_top_k must be positive")
    if spec.stage1_bench_warmup < 0 or spec.stage1_bench_rep <= 0:
        raise ValueError(
            "stage1_bench_warmup must be >= 0 and stage1_bench_rep must be > 0"
        )
    if spec.max_stage1_children_per_round <= 0:
        raise ValueError("max_stage1_children_per_round must be positive")
    return spec


def apply_no_dot_search_defaults(spec: SearchParamsSpec) -> None:
    """Expand default search values for no-dot operators.

    Explicit user-provided values are left untouched. The autotuner calls this
    only after operator classification decides the kernel should use the vector
    search policy.
    """
    if not spec.enabled or not spec.values_were_default:
        return
    spec.values = sorted(set(DEFAULT_NO_DOT_SEARCH_VALUES))


def select_initial_percentile_shapes(
    configs: Sequence[Config],
    spec: SearchParamsSpec,
    *,
    limit: Optional[int] = None,
) -> List[Config]:
    if not configs:
        return []

    space = DiscreteShapeSpace(configs, spec)
    selected = []
    seen = set()
    value_map = space.value_map
    percentiles = stage1_initial_percentiles(spec, value_map)
    target_values_by_param = [
        [values[index] for index in _percentile_indices(len(values), percentiles)]
        for values in (value_map[name] for name in spec.params)
    ]

    for value_combo in itertools.product(*target_values_by_param):
        target_shape = dict(zip(spec.params, value_combo))
        config = _nearest_unseen_config(target_shape, configs, spec, seen)
        if config is None:
            continue
        key = shape_key(extract_shape(config, spec.params), spec.params)
        if space.contains_key(key) and key not in seen:
            seen.add(key)
            selected.append(config)

    if limit is None or len(selected) == limit:
        return selected
    if len(selected) > limit:
        return _select_center_axis_configs(configs, spec, limit)

    return _select_diverse_configs(
        list(selected) + [config for config in configs if config not in selected],
        spec,
        limit,
        seed_configs=selected,
    )


def propose_evolved_shape_configs(
    *,
    parents: Sequence[Dict[str, Any]],
    successes: Sequence[Dict[str, Any]],
    failures: Sequence[Config],
    failure_observations: Optional[Sequence[Mapping[str, Any]]] = None,
    observed: Sequence[Config],
    all_configs: Sequence[Config],
    spec: SearchParamsSpec,
    seen_keys: set,
    limit: Optional[int] = None,
) -> List[Config]:
    """Generate Stage-1 children with a bounded acquisition model.

    The search space is discrete. Every unseen point gets a score in log-time
    units from nearby successful timings, nearby failures, predicted UB risk and
    distance from already measured points. Lower score is better. Batch diversity
    is applied greedily so one round does not spend all children around the same
    local basin.
    """

    child_limit = (
        limit if limit is not None else stage1_children_per_round(spec, all_configs)
    )
    if child_limit <= 0:
        return []

    space = DiscreteShapeSpace(all_configs, spec)
    failure_observations = list(failure_observations or [])
    acquisition = ShapeAcquisition.from_history(
        space=space,
        parents=parents,
        successes=successes,
        failures=failures,
        failure_observations=failure_observations,
        observed=observed,
        spec=spec,
    )

    candidates = []
    for key, config in space.by_key.items():
        if key in seen_keys:
            continue
        shape = extract_shape(config, spec.params)
        point = space.point_from_shape(shape)
        score = acquisition.cost(point)
        candidates.append([score, config, point])

    selected = []
    selected_keys = set()
    selected_points = []
    while candidates and len(selected) < child_limit:
        best_index = min(
            range(len(candidates)),
            key=lambda index: (
                acquisition.batch_adjusted_cost(candidates[index][2], selected_points),
                candidates[index][2],
            ),
        )
        _, config, point = candidates.pop(best_index)
        key = shape_key(extract_shape(config, spec.params), spec.params)
        if key in selected_keys:
            continue
        selected.append(config)
        selected_keys.add(key)
        selected_points.append(point)
    return selected


def stage1_initial_percentiles(
    spec: SearchParamsSpec,
    value_map: Optional[Dict[str, List[int]]] = None,
) -> List[float]:
    if not value_map:
        return list(spec.shape_initial_percentiles)
    counts = [len(values) for values in value_map.values()]
    if counts and min(counts) < SMALL_VALUE_COUNT_THRESHOLD:
        return list(SMALL_VALUE_PERCENTILES)
    return list(spec.shape_initial_percentiles)


def stage1_total_budget(
    spec: SearchParamsSpec,
    configs: Optional[Sequence[Config]] = None,
) -> int:
    total = (
        len(configs) if configs is not None else len(spec.values) ** len(spec.params)
    )
    if total <= 0:
        return 0
    budget = 8 + 3 * len(spec.params)
    budget = min(budget, DEFAULT_STAGE1_TOTAL_BUDGET_CAP)
    return min(total, max(spec.shape_final_top_k, budget))


def stage1_initial_budget(
    spec: SearchParamsSpec,
    configs: Optional[Sequence[Config]] = None,
) -> int:
    total_budget = stage1_total_budget(spec, configs)
    if total_budget <= 0:
        return 0
    initial = 2 * len(spec.params) + (4 if len(spec.params) >= 2 else 3)
    return min(total_budget, max(1, initial))


def stage1_children_per_round(
    spec: SearchParamsSpec,
    configs: Optional[Sequence[Config]] = None,
    *,
    remaining_unseen: Optional[int] = None,
    remaining_budget: Optional[int] = None,
    remaining_rounds: Optional[int] = None,
) -> int:
    value_map = (
        effective_search_value_map(configs, spec) if configs is not None else None
    )
    points_per_dim = len(stage1_initial_percentiles(spec, value_map))
    count = min(
        points_per_dim ** len(spec.params),
        spec.max_stage1_children_per_round,
    )
    if remaining_unseen is not None:
        count = min(count, max(0, remaining_unseen))
    if remaining_budget is not None:
        budget = max(0, remaining_budget)
        if remaining_rounds is not None and remaining_rounds > 0:
            budget = int(math.ceil(budget / remaining_rounds))
        count = min(count, budget)
    return count


def select_top_shape_entries(entries: Sequence[Any], *, k: int, key_fn) -> List[Any]:
    return sorted(entries, key=key_fn)[:k]


def filter_shapes_by_ub_and_timing(
    entries: Sequence[Any],
    *,
    ub_floor_ratio: float = UB_FLOOR_RATIO,
    timing_ceiling_ratio: float = TIMING_CEILING_RATIO,
    timing_sort_key,
) -> List[Any]:
    """Drop entries whose UB is below ``ub_floor_ratio * max_ub`` AND whose timing
    is above ``timing_ceiling_ratio * min_timing``. Both conditions must hold for a
    drop; a small-UB shape that is also fast is kept.

    UB=None on any entry disables the filter (returns the input unchanged). This
    keeps backward compatibility when compile-time memory info is unavailable.
    """
    if not entries:
        return list(entries)
    ub_values = [entry.get("ub") for entry in entries]
    if any(value is None for value in ub_values):
        return list(entries)
    max_ub = max(ub_values)
    if max_ub <= 0:
        return list(entries)
    timings = [timing_sort_key(entry["time"]) for entry in entries]
    min_timing = min(timings)
    if min_timing <= 0:
        return list(entries)
    ub_floor = ub_floor_ratio * max_ub
    timing_ceiling = timing_ceiling_ratio * min_timing
    kept = []
    for entry, ub, timing in zip(entries, ub_values, timings):
        if ub < ub_floor and timing > timing_ceiling:
            continue
        kept.append(entry)
    return kept


def ub_timing_weighted_key(entry, *, timing_sort_key):
    """UB-aware rank key after ``filter_shapes_by_ub_and_timing``.

    The hard policy is handled by ``filter_shapes_by_ub_and_timing``:
    low-UB and clearly-slow shapes are removed. Among the remaining shapes,
    timing stays primary and UB is only a tie-breaker. This keeps the search
    performance-oriented while still preferring larger UB when latency is close.

    Returns a tuple that sorts ascending under ``sorted``.
    """
    ub = entry.get("ub")
    time_value = timing_sort_key(entry["time"])
    if not ub or time_value <= 0:
        return (1, time_value, 0)
    return (0, time_value, -ub)


def _percentile_indices(count: int, percentiles: Sequence[float]) -> List[int]:
    if count <= 0:
        return []
    if count == 1:
        return [0]

    indices = []
    for percentile in percentiles:
        raw = percentile * (count - 1)
        if percentile < 0.5:
            index = int(round(raw))
        else:
            index = int(math.ceil(raw))
        if count > len(percentiles) + 1:
            index = min(max(index, 1), count - 2)
        else:
            index = min(max(index, 0), count - 1)
        if index not in indices:
            indices.append(index)

    # Very short domains can collapse nearby percentiles. Fill from the closest
    # remaining high-percentile side first, so [0.30, 0.70] over 3 values becomes
    # middle/high instead of a single middle point.
    if len(indices) < min(len(percentiles), count):
        for index in range(count - 1, -1, -1):
            if index in indices:
                continue
            indices.append(index)
            if len(indices) >= min(len(percentiles), count):
                break
    return indices


def _nearest_unseen_config(
    target_shape: Dict[str, int],
    configs: Sequence[Config],
    spec: SearchParamsSpec,
    seen: set,
) -> Optional[Config]:
    best = None
    best_distance = None
    for config in configs:
        shape = extract_shape(config, spec.params)
        key = shape_key(shape, spec.params)
        if key in seen:
            continue
        distance = _shape_distance(target_shape, shape, spec)
        if best is None or distance < best_distance:
            best = config
            best_distance = distance
    return best


def _select_diverse_configs(
    configs: Sequence[Config],
    spec: SearchParamsSpec,
    limit: int,
    *,
    seed_configs: Sequence[Config] = (),
) -> List[Config]:
    if limit <= 0:
        return []

    space = DiscreteShapeSpace(configs, spec)
    unique = list(space.unique_configs)

    selected = []
    selected_keys = set()
    for config in seed_configs:
        key = space.key(config)
        if key in selected_keys or key not in space.by_key:
            continue
        selected.append(space.by_key[key])
        selected_keys.add(key)
        if len(selected) >= limit:
            return selected

    if not selected and unique:
        center = tuple((len(spec.values) - 1) / 2.0 for _ in spec.params)
        first = min(
            unique,
            key=lambda config: _raw_norm(
                tuple(
                    float(index) - center_dim
                    for index, center_dim in zip(
                        space.point(config),
                        center,
                    )
                )
            ),
        )
        selected.append(first)
        selected_keys.add(space.key(first))

    while len(selected) < limit:
        remaining = [
            config for config in unique if space.key(config) not in selected_keys
        ]
        if not remaining:
            break
        selected_points = [space.point(config) for config in selected]
        best = max(
            remaining,
            key=lambda config: (
                _min_distance(
                    space.point(config),
                    selected_points,
                    spec,
                ),
                _edge_distance(
                    space.point(config),
                    spec,
                ),
            ),
        )
        selected.append(best)
        selected_keys.add(space.key(best))
    return selected


def _select_center_axis_configs(
    configs: Sequence[Config],
    spec: SearchParamsSpec,
    limit: int,
) -> List[Config]:
    """Trim initial sampling without picking expensive corners first.

    The initial budget is small, so the first round should estimate a local
    shape surface around the middle of the discrete domain instead of probing
    high-high corners such as 1024x1024.  This is a discrete central-composite
    design: center hypercube first, then one-axis near-ring probes.
    """
    if limit <= 0:
        return []

    space = DiscreteShapeSpace(configs, spec)
    by_point = space.by_point

    if len(by_point) <= limit:
        return [by_point[point] for point in sorted(by_point)]

    values_by_dim = space.values_by_dim
    center_choices = []
    low_center = []
    for values in values_by_dim:
        if not values:
            center_choices.append([])
            low_center.append(0)
            continue
        upper = len(values) // 2
        lower = max(0, upper - 1) if len(values) % 2 == 0 else upper
        choices = [values[lower]]
        if values[upper] not in choices:
            choices.append(values[upper])
        center_choices.append(choices)
        low_center.append(values[lower])

    selected_points: List[Tuple[int, ...]] = []

    def add_point(point):
        if point in by_point and point not in selected_points:
            selected_points.append(point)

    for point in itertools.product(*center_choices):
        add_point(tuple(point))
        if len(selected_points) >= limit:
            return [by_point[point] for point in selected_points]

    # Add one-axis high-side probes before low-side probes.  For the other
    # dimensions we enumerate the center choices, so 2-D domains cover both
    # (low_center, high_axis) and (high_center, high_axis) within an 8 point
    # budget.  That keeps useful middle-large tiles visible without jumping to
    # high-high corners.
    for dim in range(len(spec.params)):
        values = values_by_dim[dim]
        base_index = values.index(low_center[dim])
        if base_index + 2 < len(values):
            for center_combo in itertools.product(*center_choices):
                point = list(center_combo)
                point[dim] = values[base_index + 2]
                add_point(tuple(point))
                if len(selected_points) >= limit:
                    return [by_point[point] for point in selected_points]

    for dim in range(len(spec.params)):
        values = values_by_dim[dim]
        base_index = values.index(low_center[dim])
        if base_index - 1 >= 0:
            for center_combo in itertools.product(*center_choices):
                point = list(center_combo)
                point[dim] = values[base_index - 1]
                add_point(tuple(point))
                if len(selected_points) >= limit:
                    return [by_point[point] for point in selected_points]

    center = tuple((values[0] + values[-1]) / 2.0 for values in values_by_dim)
    remaining = [point for point in by_point if point not in selected_points]
    remaining.sort(
        key=lambda point: (
            _index_distance_float(point, center, spec),
            -sum(point),
            max(point),
            point,
        )
    )
    for point in remaining:
        selected_points.append(point)
        if len(selected_points) >= limit:
            break
    return [by_point[point] for point in selected_points]


def _shape_distance(
    a: Dict[str, int], b: Dict[str, int], spec: SearchParamsSpec
) -> float:
    denom = max(1, len(spec.values) - 1)
    distance = 0.0
    for name in spec.params:
        ai = spec.values.index(a[name])
        bi = spec.values.index(b[name])
        distance += abs(ai - bi) / denom
    return distance


def _shape_index_vector(
    shape: Dict[str, int], spec: SearchParamsSpec
) -> Tuple[int, ...]:
    return tuple(spec.values.index(shape[name]) for name in spec.params)


def _edge_distance(point: Tuple[int, ...], spec: SearchParamsSpec) -> float:
    max_index = len(spec.values) - 1
    if max_index <= 0:
        return 0.0
    return min(min(index, max_index - index) / max_index for index in point)


def _min_distance(
    point: Tuple[int, ...],
    points: Sequence[Tuple[int, ...]],
    spec: SearchParamsSpec,
) -> float:
    if not points:
        return 1.0
    return min(_index_distance(point, other, spec) for other in points)


def _index_distance(
    a: Tuple[int, ...],
    b: Tuple[int, ...],
    spec: SearchParamsSpec,
) -> float:
    return _index_distance_float(a, b, spec)


def _index_distance_float(
    a: Sequence[float],
    b: Sequence[float],
    spec: SearchParamsSpec,
) -> float:
    denom = max(1, len(spec.values) - 1)
    squared = 0.0
    for ai, bi in zip(a, b):
        squared += ((ai - bi) / denom) ** 2
    return math.sqrt(squared)


def _raw_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))
