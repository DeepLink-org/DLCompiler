from __future__ import annotations

from typing import Any, Mapping, Sequence

from triton.runtime.autotuner import Config

from .schedule_profiles import classify_compile_failure, effective_compile_profile_key
from .tile_acquisition_model import resource_overflow_ratio
from .tile_shape_space import extract_shape, shape_key


class SearchMeasureCache:
    """Exact shape/profile measurement cache for search_params autotune."""

    def __init__(
        self, params: Sequence[str], ub_cache: Mapping[Config, Any] | None = None
    ):
        self.params = list(params)
        self.ub_cache = ub_cache if ub_cache is not None else {}
        self._records = {}

    def __len__(self) -> int:
        return len(self._records)

    def key(self, shape: Mapping[str, Any], profile: Mapping[str, Any]):
        return (
            shape_key(dict(shape), self.params),
            effective_compile_profile_key(profile),
        )

    def get(self, shape: Mapping[str, Any], profile: Mapping[str, Any]):
        return self._records.get(self.key(shape, profile))

    def put(
        self,
        shape: Mapping[str, Any],
        profile: Mapping[str, Any],
        config: Config | None,
        *,
        cost=None,
        error=None,
        failure=None,
        source="stage2",
    ):
        ok = cost is not None and cost != float("inf") and error is None
        failure_kind = None if ok else failure or classify_compile_failure(error)
        ub = None
        if ok and config is not None:
            ub = self.ub_cache.get(config)
        self._records[self.key(shape, profile)] = {
            "ok": ok,
            "shape": dict(shape),
            "profile": dict(profile),
            "config": config,
            "time": cost if ok else None,
            "failure": failure_kind,
            "error": error,
            "resource_ratio": None if ok else resource_overflow_ratio(error),
            "ub": ub,
            "source": source,
        }

    def attempt_count(self, shape: Mapping[str, Any]) -> int:
        shape_id = shape_key(dict(shape), self.params)
        return sum(1 for cached_shape, _ in self._records if cached_shape == shape_id)

    def stage1_failure_observations(self, failed_configs: Sequence[Config]):
        failed_shape_keys = {
            shape_key(extract_shape(config, self.params), self.params)
            for config in failed_configs
        }
        observations = []
        for cached_shape, profile_key in self._records:
            if cached_shape not in failed_shape_keys:
                continue
            cached = self._records[(cached_shape, profile_key)]
            if cached.get("ok") or cached.get("source") != "stage1_fast":
                continue
            observations.append(cached)
        return observations
