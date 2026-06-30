from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from regression_config import (
    CASE_DIR,
    case_cmd,
    case_env,
    load_regression_specs,
    prepare_clean_cache,
)


MAX_ATTEMPTS = 3


@dataclass
class CaseRun:
    name: str
    attempt: int
    rc: int
    log_path: Path
    operator_kinds: list[str]
    compile_options: list[str]
    precise_costs: list[float]
    triton_ms: list[float]
    search_times: list[float]
    selected_configs: list[str]
    autotune_enabled_count: int
    stage1_initial_count: int
    stage1_selected_count: int
    stage3_final_pick_count: int
    fatal_log_markers: list[str]


def _as_float_list(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(value)]


def _extend_from_payload(target: list, value):
    if value is None:
        return
    if isinstance(value, list):
        target.extend(value)
    else:
        target.insert(0, value)


def _has_npu_runtime() -> bool:
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return bool(hasattr(torch, "npu") and torch.npu.is_available())


def _load_goldens():
    return load_regression_specs()


def _artifact_root(tmp_path: Path) -> Path:
    configured = os.getenv("DLC_AUTOTUNE_REGRESSION_LOG_DIR")
    root = Path(configured) if configured else tmp_path / "autotune_regression_logs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _parse_run(name: str, attempt: int, rc: int, log_path: Path) -> CaseRun:
    text = log_path.read_text(errors="ignore")
    payloads = [
        json.loads(match)
        for match in re.findall(r"DLC_AUTOTUNE_REGRESSION_RESULT=(\{.*\})", text)
    ]
    precise_costs = [
        float(value)
        for value in re.findall(
            r"Stage 3 final pick .*?precise_cost=([0-9.eE+-]+)", text
        )
    ]
    triton_ms = [
        float(value) for value in re.findall(r"triton_ms: ([0-9.eE+-]+)", text)
    ]
    search_times = [
        float(value)
        for value in re.findall(
            r"finished after ([0-9.eE+-]+)s; best config selected", text
        )
    ]
    selected_configs = re.findall(r"best config selected: (.*)", text)
    if payloads:
        payload = payloads[-1]
        _extend_from_payload(triton_ms, payload.get("triton_ms"))
        if not search_times:
            _extend_from_payload(search_times, payload.get("search_time"))
        if not selected_configs:
            _extend_from_payload(selected_configs, payload.get("best_config"))
    fatal_markers = [
        marker
        for marker in (
            "Traceback",
            "TIMEOUT after",
            "Segmentation fault",
            "Aborted",
            "core dumped",
        )
        if marker in text
    ]
    return CaseRun(
        name=name,
        attempt=attempt,
        rc=rc,
        log_path=log_path,
        operator_kinds=re.findall(r"operator_kind=([A-Z_]+)", text),
        compile_options=re.findall(r"compile_options=([a-z0-9_]+)", text),
        precise_costs=precise_costs,
        triton_ms=triton_ms,
        search_times=search_times,
        selected_configs=selected_configs,
        autotune_enabled_count=text.count("Search params autotuning: enabled"),
        stage1_initial_count=text.count("Stage 1 initial"),
        stage1_selected_count=text.count("Stage 1 selected shapes"),
        stage3_final_pick_count=text.count("Stage 3 final pick"),
        fatal_log_markers=fatal_markers,
    )


def _run_once(name: str, spec: dict, attempt: int, artifact_root: Path) -> CaseRun:
    script = CASE_DIR / spec["script"]
    assert script.is_file(), f"missing regression case script: {script}"

    case_root = artifact_root / name / f"attempt_{attempt}"
    cache_root = case_root / "triton_cache"
    case_root.mkdir(parents=True, exist_ok=True)
    prepare_clean_cache(cache_root)
    log_path = case_root / "run.log"

    env = case_env(os.environ, cache_root, name)
    env["DLC_AUTOTUNE_CASE_LIMIT"] = env.get("DLC_AUTOTUNE_CASE_LIMIT", "0")

    cmd = case_cmd(script)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(CASE_DIR),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(spec.get("timeout_s", 600)),
            check=False,
        )
        log_path.write_text(proc.stdout)
        return _parse_run(name, attempt, proc.returncode, log_path)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="ignore")
        log_path.write_text(output + f"\nTIMEOUT after {spec.get('timeout_s', 600)}s\n")
        return _parse_run(name, attempt, 124, log_path)


def _performance_failures(run: CaseRun, spec: dict) -> list[str]:
    failures = []
    golden_precise = _as_float_list(spec.get("golden_precise_cost"))
    if golden_precise:
        if len(run.precise_costs) < len(golden_precise):
            failures.append(
                f"precise_cost count {len(run.precise_costs)} < golden count {len(golden_precise)}"
            )
        for index, golden in enumerate(golden_precise):
            if index >= len(run.precise_costs):
                break
            limit = golden * spec.get("max_precise_cost_ratio", 1.5)
            if run.precise_costs[index] > limit:
                failures.append(
                    f"precise_cost[{index}] {run.precise_costs[index]:.6g} > {limit:.6g}"
                )
    golden_ms = _as_float_list(spec.get("golden_triton_ms"))
    if golden_ms:
        if len(run.triton_ms) < len(golden_ms):
            failures.append(
                f"triton_ms count {len(run.triton_ms)} < golden count {len(golden_ms)}"
            )
        for index, golden in enumerate(golden_ms):
            if index >= len(run.triton_ms):
                break
            limit = golden * spec.get("max_triton_ms_ratio", 2.0)
            if run.triton_ms[index] > limit:
                failures.append(
                    f"triton_ms[{index}] {run.triton_ms[index]:.6g} > {limit:.6g}"
                )
    return failures


def _performance_score(run: CaseRun, spec: dict) -> float:
    score = 0.0
    golden_precise = _as_float_list(spec.get("golden_precise_cost"))
    for index, golden in enumerate(golden_precise):
        if index < len(run.precise_costs) and golden > 0:
            score = max(score, run.precise_costs[index] / golden)
    golden_ms = _as_float_list(spec.get("golden_triton_ms"))
    for index, golden in enumerate(golden_ms):
        if index < len(run.triton_ms) and golden > 0:
            score = max(score, run.triton_ms[index] / golden)
    return score if score > 0.0 else float("inf")


def _max_search_time_failures(run: CaseRun, spec: dict) -> list[str]:
    limit_hint = spec.get("max_search_time_s")
    if limit_hint is None:
        return []

    failures = []
    limits = _as_float_list(limit_hint)
    if len(limits) == 1:
        limits = limits * max(1, len(run.search_times))
    if len(run.search_times) < len(limits):
        failures.append(
            f"search_time count {len(run.search_times)} < expected count {len(limits)}"
        )
    for index, limit in enumerate(limits):
        if index >= len(run.search_times):
            break
        if run.search_times[index] > limit:
            failures.append(
                f"search_time[{index}] {run.search_times[index]:.2f}s > {limit:.2f}s"
            )
    return failures


def _log_failures(run: CaseRun, spec: dict) -> list[str]:
    failures = []
    if run.fatal_log_markers:
        failures.append(
            f"fatal log markers found: {sorted(set(run.fatal_log_markers))}"
        )

    expected_runs = int(spec.get("expected_autotune_runs", 1))
    log_counts = {
        "autotune enabled": run.autotune_enabled_count,
        "Stage 1 initial": run.stage1_initial_count,
        "Stage 1 selected shapes": run.stage1_selected_count,
        "Stage 3 final pick": run.stage3_final_pick_count,
        "best config selected": len(run.selected_configs),
    }
    for label, observed in log_counts.items():
        if observed < expected_runs:
            failures.append(f"{label} count {observed} < expected {expected_runs}")

    failures.extend(_max_search_time_failures(run, spec))
    return failures


def _semantic_failures(run: CaseRun, spec: dict) -> list[str]:
    failures = []
    expected_kind = spec.get("expected_operator_kind")
    if expected_kind and expected_kind not in run.operator_kinds:
        failures.append(
            f"operator_kind {expected_kind!r} not found; observed={run.operator_kinds}"
        )
    expected_compile = spec.get("expected_compile_options")
    if expected_compile and expected_compile not in run.compile_options:
        failures.append(
            f"compile_options {expected_compile!r} not found; observed={run.compile_options}"
        )
    return failures


def _format_attempts(attempts: list[CaseRun]) -> str:
    lines = []
    for run in attempts:
        lines.append(
            "attempt={attempt}, rc={rc}, precise={precise}, triton_ms={triton}, "
            "search_times={search}, selected_configs={selected}, log={log}".format(
                attempt=run.attempt,
                rc=run.rc,
                precise=run.precise_costs,
                triton=run.triton_ms,
                search=run.search_times[:3],
                selected=len(run.selected_configs),
                log=run.log_path,
            )
        )
    return "\n".join(lines)


def _case_names():
    return list(_load_goldens())


@pytest.mark.parametrize("case_name", _case_names())
def test_autotune_regression_case(case_name, tmp_path):
    if not _has_npu_runtime():
        pytest.skip("Ascend NPU runtime is not available")

    goldens = _load_goldens()
    spec = goldens[case_name]
    artifact_root = _artifact_root(tmp_path)
    attempts = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        run = _run_once(case_name, spec, attempt, artifact_root)
        attempts.append(run)
        if run.rc != 0:
            break
        if _semantic_failures(run, spec):
            break
        if _log_failures(run, spec):
            break
        if not _performance_failures(run, spec):
            break

    successful_runs = [run for run in attempts if run.rc == 0]
    passing_runs = [
        run
        for run in successful_runs
        if not _semantic_failures(run, spec)
        and not _log_failures(run, spec)
        and not _performance_failures(run, spec)
    ]
    best = (
        passing_runs[0]
        if passing_runs
        else min(
            successful_runs,
            key=lambda run: _performance_score(run, spec),
            default=attempts[-1],
        )
    )

    failures = []
    if best.rc != 0:
        failures.append(f"process failed with rc={best.rc}")
    failures.extend(_semantic_failures(best, spec))
    failures.extend(_log_failures(best, spec))
    failures.extend(_performance_failures(best, spec))
    assert not failures, "\n".join(failures + ["", _format_attempts(attempts)])
