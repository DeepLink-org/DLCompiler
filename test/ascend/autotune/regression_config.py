from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
CASE_DIR = THIS_DIR / "regression_cases"
GOLDEN_PATH = THIS_DIR / "autotune_regression_goldens.json"

DEFAULT_BENCH_WARMUP = 1
DEFAULT_BENCH_REPEAT = 3
FA2_SELECTED_CASES = "1,4"

PARALLEL_DEVICES = {
    "fa2": 1,
    "fla_chunk_delta_hupdate": 2,
    "fla_cumsum": 3,
    "gdn_chunk_meta": 6,
    "batch_invariant_mean": 4,
}


def load_regression_specs() -> dict:
    return json.loads(GOLDEN_PATH.read_text())


def case_env(
    base_env: dict[str, str],
    cache_root: Path,
    name: str,
    device: int | None = None,
) -> dict[str, str]:
    env = dict(base_env)
    env.update(
        {
            "PYTHONPATH": f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}",
            "PYTHONUNBUFFERED": "1",
            "TRITON_PRINT_AUTOTUNING": "1",
            "TRITON_PRINT_AUTOTUNING_TIMINGS": "1",
            "TRITON_CACHE_DIR": str(cache_root),
            "DLC_AUTOTUNE_BENCH_WARMUP": str(DEFAULT_BENCH_WARMUP),
            "DLC_AUTOTUNE_BENCH_ACTIVE": str(DEFAULT_BENCH_REPEAT),
        }
    )
    if device is not None:
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
    if name == "fa2":
        env["TEST_FA2_CASES"] = FA2_SELECTED_CASES
    return env


def prepare_clean_cache(cache_root: Path) -> None:
    shutil.rmtree(cache_root, ignore_errors=True)
    cache_root.mkdir(parents=True, exist_ok=True)


def case_cmd(script: Path) -> list[str]:
    return [
        sys.executable,
        str(script),
        "--warmup",
        str(DEFAULT_BENCH_WARMUP),
        "--repeat",
        str(DEFAULT_BENCH_REPEAT),
    ]
