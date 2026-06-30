from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from triton.backends.dicp_triton.utils import is_compile_on_910_95

from .schedule_profiles import (
    CompileFailureRegionSet,
    classify_compile_failure,
    compile_profile_resource_not_less,
    compile_profile_to_config,
    effective_compile_profile_key,
    generate_linked_compile_neighbors,
    make_stage2_seed_profiles,
)
from .measurement_cache import SearchMeasureCache


Stage2Candidate = tuple[Any, Mapping[str, Any], Any]


@dataclass
class _Stage2State:
    base_profile: Mapping[str, Any]
    rng: random.Random
    timing_sort_key: Callable[[Any], float]
    candidates: list[Stage2Candidate] = field(default_factory=list)
    failed_regions: CompileFailureRegionSet = field(
        default_factory=CompileFailureRegionSet
    )
    seen_profiles: set = field(default_factory=set)
    current_profile: Mapping[str, Any] | None = None
    current_time: Any = float("inf")
    best_profile: Mapping[str, Any] | None = None
    best_time: Any = float("inf")

    def __post_init__(self):
        self.current_profile = self.base_profile
        self.best_profile = self.base_profile
        self.seen_profiles.add(effective_compile_profile_key(self.base_profile))

    def accept_annealed(self, new_cost, temperature: float) -> bool:
        old_value = self.timing_sort_key(self.current_time)
        new_value = self.timing_sort_key(new_cost)
        if new_value <= old_value:
            return True
        if temperature <= 0:
            return False
        return self.rng.random() < math.exp((old_value - new_value) / temperature)

    def add_success(self, config, profile, cost, temperature: float):
        self.candidates.append((config, profile, cost))
        if self.timing_sort_key(cost) < self.timing_sort_key(self.best_time):
            self.best_profile = profile
            self.best_time = cost
        if self.accept_annealed(cost, temperature):
            self.current_profile = profile
            self.current_time = cost


class Stage2CompileSearcher:
    """Branch-aware compile option search for one selected Stage-1 shape."""

    def __init__(
        self,
        *,
        search_params,
        operator_policy,
        cache: SearchMeasureCache,
        apply_fixed_profile: Callable[[Mapping[str, Any]], dict],
        bench_configs: Callable[[Sequence[Any]], tuple[dict, dict]],
        timing_sort_key: Callable[[Any], float],
        is_finite_timing: Callable[[Any], bool],
        debug: Callable[[str], None],
        format_error_log: Callable[[Any], str],
    ):
        self.search_params = search_params
        self.operator_policy = operator_policy
        self.cache = cache
        self.apply_fixed_profile = apply_fixed_profile
        self.bench_configs = bench_configs
        self.timing_sort_key = timing_sort_key
        self.is_finite_timing = is_finite_timing
        self.debug = debug
        self.format_error_log = format_error_log

    def search(self, *, stage1_entry, stage1_rank: int, stage1_top1_time):
        shape = stage1_entry["shape"]
        shape_config = stage1_entry["shape_config"]
        stage1_ub = stage1_entry.get("ub")
        rng = random.Random(self.search_params.shape_final_top_k + len(shape))
        state = _Stage2State(
            base_profile=stage1_entry["profile"],
            rng=rng,
            timing_sort_key=self.timing_sort_key,
        )
        trials = max(1, self.cache.attempt_count(shape))

        self.debug(
            "Stage 2 start "
            f"shape={shape}, stage1_base_time={stage1_entry['time']}, "
            f"stage1_ub={stage1_ub}, base={stage1_entry['profile']}"
        )
        if not getattr(self.operator_policy, "stage2_enabled", True):
            self.debug(
                "Stage 2 skipped "
                f"shape={shape}, operator_kind={self.operator_policy.operator_kind.value}"
            )
            return []

        batch_size = max(1, self.search_params.neighbors_per_step)
        main_trial_budget = max(1, self.search_params.max_compile_trials_per_shape - 1)
        prefer_resource_relax = stage1_ub is not None and stage1_ub >= 160 * 1024

        base_result = self._cached_or_bench_profile(
            shape=shape,
            shape_config=shape_config,
            profile=stage1_entry["profile"],
            item_index=0,
        )
        self._apply_result(
            state,
            base_result,
            "base cache" if base_result["source"] == "cache" else "base fast",
            self.search_params.compile_initial_temperature,
            shape,
        )
        base_cost = (
            base_result["cost"] if base_result["error"] is None else float("inf")
        )
        if not self.is_finite_timing(base_cost):
            self.debug(
                f"Stage 2 gate stop shape={shape}, rank={stage1_rank}, "
                "reason=base_failed"
            )
            return state.candidates

        stage1_top1_value = self.timing_sort_key(stage1_top1_time)
        base_ratio = self.timing_sort_key(base_cost) / stage1_top1_value
        if base_ratio > 3.0:
            self.debug(
                f"Stage 2 gate stop shape={shape}, rank={stage1_rank}, "
                f"base_ratio={base_ratio:.3f} > 3.000"
            )
            return state.candidates

        if base_ratio >= 1.5 and self.operator_policy.mixcv_quick_gate:
            trials += self._run_quick_gate(
                state,
                shape=shape,
                shape_config=shape_config,
                base_profile=stage1_entry["profile"],
                base_ratio=base_ratio,
                stage1_rank=stage1_rank,
            )
            quick_ratio = self.timing_sort_key(state.best_time) / stage1_top1_value
            if quick_ratio > 1.5:
                self.debug(
                    f"Stage 2 gate stop shape={shape}, rank={stage1_rank}, "
                    f"quick_ratio={quick_ratio:.3f} > 1.500"
                )
                return self._top_candidates(state)
            self.debug(
                f"Stage 2 gate pass shape={shape}, rank={stage1_rank}, "
                f"quick_ratio={quick_ratio:.3f} <= 1.500"
            )

        trials = self._run_seed_phase(
            state,
            shape=shape,
            shape_config=shape_config,
            stage1_ub=stage1_ub,
            trials=trials,
            main_trial_budget=main_trial_budget,
            batch_size=batch_size,
        )
        trials = self._run_anneal_phase(
            state,
            shape=shape,
            shape_config=shape_config,
            trials=trials,
            main_trial_budget=main_trial_budget,
            batch_size=batch_size,
            prefer_resource_relax=prefer_resource_relax,
        )
        self._run_final_unit_flag_trial(
            state,
            shape=shape,
            shape_config=shape_config,
            trials=trials,
        )
        self.debug(
            f"Stage 2 done shape={shape}, trials={trials}, "
            f"best_time={state.best_time}, best_profile={state.best_profile}"
        )
        return self._top_candidates(state)

    def _run_quick_gate(
        self,
        state: _Stage2State,
        *,
        shape,
        shape_config,
        base_profile,
        base_ratio,
        stage1_rank,
    ) -> int:
        quick_items = []
        for quick_index, (workspace, cube, vector) in enumerate(
            ((4, 2, 2), (2, 1, 1)), 1
        ):
            profile = dict(
                base_profile, set_workspace_multibuffer=workspace, unit_flag=False
            )
            if not is_compile_on_910_95:
                profile["tile_mix_cube_loop"] = cube
                profile["tile_mix_vector_loop"] = vector
            profile = self.apply_fixed_profile(profile)
            if not self._mark_runnable(state, profile, shape, f"quick[{quick_index}]"):
                continue
            quick_items.append((quick_index, profile))

        if not quick_items:
            return 0
        self.debug(
            f"Stage 2 quick gate shape={shape}, rank={stage1_rank}, "
            f"base_ratio={base_ratio:.3f}, size={len(quick_items)}"
        )
        for result in self._bench_profile_batch(
            shape=shape,
            shape_config=shape_config,
            profile_items=quick_items,
        ):
            self._apply_result(state, result, f"quick[{result['index']}]", 0.20, shape)
        return len(quick_items)

    def _run_seed_phase(
        self,
        state: _Stage2State,
        *,
        shape,
        shape_config,
        stage1_ub,
        trials,
        main_trial_budget,
        batch_size,
    ) -> int:
        seeds = make_stage2_seed_profiles(
            state.base_profile,
            shape_kwargs=shape,
            seed_budget=self.search_params.seed_budget,
            allow_unit_flag=False,
            stage1_ub_bytes=stage1_ub,
        )
        seed_batch = []

        def flush_seed_batch():
            nonlocal trials, seed_batch
            if not seed_batch:
                return
            self.debug(f"Stage 2 seed batch shape={shape}, size={len(seed_batch)}")
            results = self._bench_profile_batch(
                shape=shape,
                shape_config=shape_config,
                profile_items=seed_batch,
            )
            trials += len(seed_batch)
            for result in results:
                self._apply_result(
                    state, result, f"seed[{result['index']}]", 0.20, shape
                )
            seed_batch = []

        for seed_index, profile in enumerate(seeds, 1):
            profile = self.apply_fixed_profile(profile)
            if not self._mark_runnable(state, profile, shape, f"seed[{seed_index}]"):
                continue
            cached = self.cache.get(shape, profile)
            if cached is not None:
                self._apply_cached_result(
                    state, cached, profile, f"seed[{seed_index}]", shape, 0.20
                )
                continue
            if trials >= main_trial_budget:
                break
            if seed_batch and self._has_resource_dependency(profile, seed_batch):
                flush_seed_batch()
                if state.failed_regions.is_forbidden(profile):
                    self.debug(f"Stage 2 seed[{seed_index}] forbidden shape={shape}")
                    continue
            self.debug(
                f"Stage 2 seed[{seed_index}] queued shape={shape}, profile={profile}"
            )
            seed_batch.append((seed_index, profile))
            if (
                len(seed_batch) >= batch_size
                or trials + len(seed_batch) >= main_trial_budget
            ):
                flush_seed_batch()

        flush_seed_batch()
        return trials

    def _run_anneal_phase(
        self,
        state: _Stage2State,
        *,
        shape,
        shape_config,
        trials,
        main_trial_budget,
        batch_size,
        prefer_resource_relax,
    ) -> int:
        temperature = self.search_params.compile_initial_temperature
        while trials < main_trial_budget:
            neighbors = [
                self.apply_fixed_profile(profile)
                for profile in generate_linked_compile_neighbors(
                    state.current_profile,
                    shape_kwargs=shape,
                    limit=self.search_params.neighbors_per_step,
                    allow_unit_flag=False,
                    prefer_resource_relax=prefer_resource_relax,
                )
            ]
            runnable = [
                profile for profile in neighbors if self._is_runnable(state, profile)
            ]
            if not runnable:
                break

            anneal_batch = []
            while (
                runnable
                and len(anneal_batch) < batch_size
                and trials + len(anneal_batch) < main_trial_budget
            ):
                profile = state.rng.choice(runnable)
                runnable.remove(profile)
                if anneal_batch and self._has_resource_dependency(
                    profile, anneal_batch
                ):
                    runnable.append(profile)
                    break
                state.seen_profiles.add(effective_compile_profile_key(profile))
                cached = self.cache.get(shape, profile)
                if cached is None:
                    anneal_batch.append((trials + len(anneal_batch) + 1, profile))
                    continue
                self._apply_cached_result(
                    state, cached, profile, "anneal", shape, temperature
                )
                temperature *= self.search_params.compile_cooling

            if not anneal_batch:
                continue
            self.debug(f"Stage 2 anneal batch shape={shape}, size={len(anneal_batch)}")
            results = self._bench_profile_batch(
                shape=shape,
                shape_config=shape_config,
                profile_items=anneal_batch,
            )
            trials += len(anneal_batch)
            for result in results:
                self.debug(
                    "Stage 2 anneal trial="
                    f"{result['index']}, shape={shape}, profile={result['profile']}"
                )
                self._apply_result(state, result, "anneal", temperature, shape)
                temperature *= self.search_params.compile_cooling
        return trials

    def _run_final_unit_flag_trial(self, state, *, shape, shape_config, trials):
        if (
            not self.operator_policy.allow_final_unit_flag
            or state.best_profile.get("unit_flag", False)
            or trials >= self.search_params.max_compile_trials_per_shape
        ):
            return
        trial_profile = self.apply_fixed_profile(
            dict(state.best_profile, unit_flag=True)
        )
        trial_key = effective_compile_profile_key(trial_profile)
        if trial_key in state.seen_profiles:
            return
        state.seen_profiles.add(trial_key)
        self.debug(f"Stage 2 final unit_flag trial shape={shape}")
        for result in self._bench_profile_batch(
            shape=shape,
            shape_config=shape_config,
            profile_items=[(trials + 1, trial_profile)],
        ):
            if result["error"] is None:
                state.candidates.append(
                    (result["config"], trial_profile, result["cost"])
                )

    def _cached_or_bench_profile(self, *, shape, shape_config, profile, item_index):
        cached = self.cache.get(shape, profile)
        if cached is not None:
            return self._cache_result(profile, cached, item_index)
        return self._bench_profile_batch(
            shape=shape,
            shape_config=shape_config,
            profile_items=[(item_index, profile)],
        )[0]

    def _bench_profile_batch(self, *, shape, shape_config, profile_items):
        profile_items = [
            (item_index, self.apply_fixed_profile(profile))
            for item_index, profile in profile_items
        ]
        configs = [
            compile_profile_to_config(
                profile, shape_kwargs=shape, base_config=shape_config
            )
            for _, profile in profile_items
        ]
        timings, errors = self.bench_configs(configs)
        results = []
        for (item_index, profile), config in zip(profile_items, configs):
            cost = timings.get(config, float("inf"))
            if self.is_finite_timing(cost):
                self.cache.put(shape, profile, config, cost=cost, source="stage2_fast")
                results.append(
                    self._result(item_index, profile, config, cost, None, None, "bench")
                )
                continue
            error = errors.get(config) or RuntimeError("benchmark returned inf")
            failure = classify_compile_failure(error)
            self.cache.put(
                shape,
                profile,
                config,
                error=error,
                failure=failure,
                source="stage2_fast",
            )
            results.append(
                self._result(item_index, profile, config, None, error, failure, "bench")
            )
        return results

    def _apply_result(self, state, result, stage_name, temperature, shape):
        profile = result["profile"]
        if result["error"] is not None:
            failure = state.failed_regions.add(profile, result["failure"])
            self.debug(
                f"Stage 2 {stage_name} fail shape={shape}, "
                f"classified_as={failure}, "
                f"error_log:\n{self.format_error_log(result['error'])}"
            )
            return
        state.add_success(result["config"], profile, result["cost"], temperature)
        self.debug(f"Stage 2 {stage_name} success shape={shape}, cost={result['cost']}")

    def _apply_cached_result(
        self, state, cached, profile, stage_name, shape, temperature
    ):
        result = self._cache_result(profile, cached, 0)
        self._apply_result(state, result, f"{stage_name} cache", temperature, shape)

    def _cache_result(self, profile, cached, item_index):
        if cached["ok"]:
            return self._result(
                item_index,
                profile,
                cached["config"],
                cached["time"],
                None,
                None,
                "cache",
            )
        return self._result(
            item_index,
            profile,
            cached.get("config"),
            None,
            cached.get("error"),
            cached.get("failure"),
            "cache",
        )

    @staticmethod
    def _result(index, profile, config, cost, error, failure, source):
        return {
            "index": index,
            "profile": profile,
            "config": config,
            "cost": cost,
            "error": error,
            "failure": failure,
            "source": source,
        }

    def _mark_runnable(self, state, profile, shape, label, *, quiet=False):
        profile_key = effective_compile_profile_key(profile)
        if profile_key in state.seen_profiles:
            if label and not quiet:
                self.debug(f"Stage 2 {label} duplicate shape={shape}")
            return False
        if state.failed_regions.is_forbidden(profile):
            if label and not quiet:
                self.debug(f"Stage 2 {label} forbidden shape={shape}")
            return False
        state.seen_profiles.add(profile_key)
        return True

    @staticmethod
    def _is_runnable(state, profile):
        profile_key = effective_compile_profile_key(profile)
        if profile_key in state.seen_profiles:
            return False
        if state.failed_regions.is_forbidden(profile):
            return False
        return True

    @staticmethod
    def _has_resource_dependency(profile, profile_batch):
        return any(
            compile_profile_resource_not_less(profile, pending_profile)
            or compile_profile_resource_not_less(pending_profile, profile)
            for _, pending_profile in profile_batch
        )

    def _top_candidates(self, state):
        return sorted(
            state.candidates,
            key=lambda item: self.timing_sort_key(item[2]),
        )[: self.search_params.candidate_pool_per_shape]
