#!/usr/bin/env python3
"""Audit compiled Gluon layout variants and measured runtime resources.

The input may be a v4/v5 layout manifest, a Triton kernel metadata JSON file, a
layout-export directory, or a recursive Triton cache directory.  The report is
explicit about evidence levels: operation/call counts are static textual sites,
while allocated registers and local memory are accepted only from a matching
post-load runtime measurement record. The tool never infers those resources
from TTGIR or LLVM IR.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterable


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_FAILURE_DETAIL_BYTES = 1024
_V8_REPEAT_LAUNCHES = 5
_V8_MAX_REMEASUREMENT_ROUND = 2
_V8_MEASUREMENT_PROTOCOL = {
    "version": 8,
    "timer": "device-event-isolated-v1",
    "executor": "forkserver-parallel-compile-persistent-serial-measure-v1",
    "cache_policy": "restore-scratch-then-clear-before-each-warmup-and-sample",
    "quantiles": [0.5, 0.2, 0.8],
    "warmup_launches": 5,
    "repeat_launches": 5,
    "worker_setup_timeout_seconds": 60,
    "candidate_compile_timeout_seconds": 300,
    "candidate_timeout_seconds": 60,
}
DOMAIN_ATTR_RE = re.compile(
    r'"?ttg\.gluon\.layout-domain-digest"?\s*=\s*"([0-9a-f]{64})"'
)
VARIANT_ATTR_RE = re.compile(
    r'"?ttg\.gluon\.layout-variant-digest"?\s*=\s*"([0-9a-f]{64})"'
)
LAYOUT_START_RE = re.compile(r"#ttg\.([A-Za-z0-9_]+)<")

TTGIR_OPS = {
    "dot": r"(?<![A-Za-z0-9_.])tt\.dot(?=\s)",
    "local_load": r"(?<![A-Za-z0-9_.])ttg\.local_load(?=\s)",
    "local_store": r"(?<![A-Za-z0-9_.])ttg\.local_store(?=\s)",
    "bsm_perm": r"(?<![A-Za-z0-9_.])ttg\.bsm_perm(?=[\s\"(])",
    "gvm_arrive": r"(?<![A-Za-z0-9_.])ttg\.gvm_arrive(?=\s)",
    "barrier_shared": r"(?<![A-Za-z0-9_.])ttg\.barrier_shared(?=\s)",
    "convert_layout": r"(?<![A-Za-z0-9_.])ttg\.convert_layout(?=\s)",
}
ASYNC_OP_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(ttg\.[A-Za-z0-9_.]*async_copy[A-Za-z0-9_.]*)(?=\s)"
)
GVM_NUM_RE = re.compile(r"ttg\.gvm_arrive\s*\{[^}\n]*\bnum\s*=\s*([0-9]+)")

LLVM_VECTOR_LOAD_RE = re.compile(
    r"\bload\s+(?:atomic\s+|volatile\s+)*(<\s*[0-9]+\s+x\s+[^>]+>)\s*,"
)
LLVM_VECTOR_STORE_RE = re.compile(
    r"\bstore\s+(?:atomic\s+|volatile\s+)*(<\s*[0-9]+\s+x\s+[^>]+>)\s+"
)
VECTOR_TYPE_RE = re.compile(r"<\s*([0-9]+)\s+x\s+([A-Za-z0-9]+)\s*>")
MXC_VECTOR_MEMORY_RE = re.compile(
    r"@llvm\.mxc\.(ldg|stg)\.[A-Za-z0-9_.]*?v([0-9]+)(bf16|f16|f32|f64|i[0-9]+)\b"
)
MXC_CALL_RE = re.compile(r"\b(?:call|invoke)\b[^\n]*@llvm\.mxc\.([A-Za-z0-9_.]+)")

ELEMENT_BITS = {"bf16": 16, "f16": 16, "half": 16,
                "f32": 32, "float": 32, "f64": 64, "double": 64}


@dataclass(frozen=True)
class ManifestSource:
    base: Path
    manifest: dict[str, Any]
    source: Path


@dataclass(frozen=True)
class VariantArtifact:
    domain: str | None
    digest: str
    ttgir: Path
    manifest_ttgir: Path | None
    llir: Path | None
    binary: Path | None
    expected_sha256: str | None
    fallback: bool
    manifest_source: Path | None


@dataclass(frozen=True)
class RuntimeMeasurement:
    domain: str
    digest: str
    mlir_sha256: str
    source: Path
    workload_digest: str | None
    winner: bool
    timing_ms: tuple[float, float, float] | None
    failure: str | None
    failure_detail: str | None
    compile_metrics: dict[str, Any] | None
    binary_sha256: str | None
    shared_bytes: int | None
    shared_residency_tier_kib: int | None
    allocated_registers_per_thread: int | None
    local_bytes_per_thread: int | None
    measurement_rounds: tuple[dict[str, Any], ...] = ()
    selection: dict[str, Any] | None = None

    def to_json(self, actual_binary_sha256: str | None = None) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "workload_digest": self.workload_digest,
            "winner": self.winner,
            "timing_ms": list(self.timing_ms) if self.timing_ms else None,
            "failure": self.failure,
            "failure_detail": self.failure_detail,
            "compile_metrics": self.compile_metrics,
            "measurement_rounds": list(self.measurement_rounds),
            "selection": self.selection,
            "binary_sha256": self.binary_sha256,
            "binary_sha256_match": (
                None
                if self.binary_sha256 is None or actual_binary_sha256 is None
                else self.binary_sha256 == actual_binary_sha256
            ),
            "resources": {
                "shared_bytes": self.shared_bytes,
                "shared_residency_tier_kib": self.shared_residency_tier_kib,
                "allocated_registers_per_thread": (
                    self.allocated_registers_per_thread
                ),
                "local_bytes_per_thread": self.local_bytes_per_thread,
            },
        }


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _validate_layout_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("manifest is not an object")
    version = manifest.get("version")
    if type(version) is not int or version not in {4, 5}:
        raise ValueError("unsupported manifest version")
    if set(manifest) != {
        "version", "digest", "fallback", "fallback_only", "variants"
    }:
        raise ValueError("manifest has unknown or missing fields")

    domain = manifest["digest"]
    fallback = manifest["fallback"]
    fallback_only = manifest["fallback_only"]
    variants = manifest["variants"]
    if (
        not _is_digest(domain)
        or not _is_digest(fallback)
        or not isinstance(fallback_only, bool)
        or not isinstance(variants, list)
        or not variants
    ):
        raise ValueError("manifest has invalid identities")

    expected_variant_fields = {
        "digest", "mlir_file", "mlir_sha256"
    }
    if version == 5:
        expected_variant_fields.add("stage")
    seen_digests = set()
    seen_files = set()
    for index, variant in enumerate(variants):
        if (
            not isinstance(variant, dict)
            or set(variant) != expected_variant_fields
        ):
            raise ValueError("manifest variant has unknown or missing fields")
        digest = variant["digest"]
        filename = variant["mlir_file"]
        if (
            not _is_digest(digest)
            or not _is_digest(variant["mlir_sha256"])
            or filename != f"{digest}.ttgir"
            or digest in seen_digests
            or filename in seen_files
        ):
            raise ValueError("manifest variant has an invalid identity")
        if version == 5:
            if variant["stage"] != "final-ttgir":
                raise ValueError("manifest has inconsistent finalization stages")
        seen_digests.add(digest)
        seen_files.add(filename)

    if variants[0]["digest"] != fallback:
        raise ValueError("the first manifest variant must be the fallback")
    if fallback_only != (len(variants) == 1):
        raise ValueError("manifest fallback_only is inconsistent")
    return manifest


def _extract_manifest(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    candidates = [payload]
    if "gluon_layout_manifest" in payload:
        candidates.insert(0, payload["gluon_layout_manifest"])
    for manifest in candidates:
        try:
            return _validate_layout_manifest(manifest)
        except ValueError:
            continue
    return None


def _manifest_payload(payload: Any) -> Any | None:
    if not isinstance(payload, dict):
        return None
    if "gluon_layout_manifest" in payload:
        return payload["gluon_layout_manifest"]
    if payload.get("version") in {4, 5} and "variants" in payload:
        return payload
    return None


def _manifest_score(source: ManifestSource) -> int:
    return sum(
        (source.base / variant.get("mlir_file", "")).is_file()
        for variant in source.manifest.get("variants", [])
        if isinstance(variant, dict)
    )


def _discover_manifests(
    path: Path,
) -> tuple[list[ManifestSource], list[str]]:
    candidates: Iterable[Path]
    if path.is_file():
        candidates = [path]
    else:
        candidates = path.rglob("*.json")

    by_domain: dict[str, ManifestSource] = {}
    warnings = []
    for candidate in candidates:
        raw_manifest = _manifest_payload(_read_json(candidate))
        if raw_manifest is None:
            continue
        try:
            manifest = _validate_layout_manifest(raw_manifest)
        except ValueError as error:
            warnings.append(
                f"{candidate}: ignored malformed layout manifest: {error}"
            )
            continue
        domain = manifest.get("digest")
        source = ManifestSource(candidate.parent, manifest, candidate)
        previous = by_domain.get(domain)
        if previous is None or (_manifest_score(source), str(source.base)) > (
            _manifest_score(previous), str(previous.base)
        ):
            by_domain[domain] = source
    return [by_domain[key] for key in sorted(by_domain)], warnings


def _optional_nonnegative_int(mapping: dict[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} is not a non-negative integer")
    return value


def _require_nonnegative_int(mapping: dict[str, Any], key: str) -> int:
    value = _optional_nonnegative_int(mapping, key)
    if value is None:
        raise ValueError(f"{key} is required")
    return value


def _validate_compile_metrics(
    metrics: Any, *, fallback: bool
) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        raise ValueError("compile_metrics is not an object")
    mode = metrics.get("mode")
    outcome = metrics.get("outcome")
    if fallback:
        if (
            set(metrics) != {"mode", "outcome", "wall_us"}
            or mode != "parent-fallback"
            or outcome != "success"
        ):
            raise ValueError("fallback compile_metrics has an invalid schema")
        _require_nonnegative_int(metrics, "wall_us")
        return metrics

    if mode != "isolated-candidate":
        raise ValueError("candidate compile_metrics has an invalid mode")
    if outcome == "failure":
        if set(metrics) != {"mode", "outcome", "isolated_wall_us"}:
            raise ValueError("failed candidate compile_metrics has an invalid schema")
        _require_nonnegative_int(metrics, "isolated_wall_us")
        return metrics
    if outcome != "success" or set(metrics) != {
        "mode", "outcome", "cache_hit", "ir_initialization_us",
        "lowering_stages_us", "store_results_us", "total_us",
        "isolated_wall_us",
    }:
        raise ValueError("successful candidate compile_metrics has an invalid schema")
    if not isinstance(metrics["cache_hit"], bool):
        raise ValueError("compile_metrics cache_hit is not a boolean")
    ir_us = _require_nonnegative_int(metrics, "ir_initialization_us")
    store_us = _require_nonnegative_int(metrics, "store_results_us")
    total_us = _require_nonnegative_int(metrics, "total_us")
    _require_nonnegative_int(metrics, "isolated_wall_us")
    stages = metrics["lowering_stages_us"]
    if not isinstance(stages, list):
        raise ValueError("compile_metrics lowering_stages_us is not a list")
    stage_names = set()
    stage_total = 0
    for stage in stages:
        if (
            not isinstance(stage, list)
            or len(stage) != 2
            or not isinstance(stage[0], str)
            or not stage[0]
            or stage[0] in stage_names
        ):
            raise ValueError("compile_metrics has an invalid or duplicate stage")
        duration = stage[1]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, int)
            or duration < 0
        ):
            raise ValueError("compile_metrics stage duration is invalid")
        stage_names.add(stage[0])
        stage_total += duration
    if total_us != ir_us + stage_total + store_us:
        raise ValueError("compile_metrics total_us is inconsistent")
    return metrics


def _validate_executable_identity(
    executable: Any, digest: str, mlir_sha256: str
) -> dict[str, Any]:
    if not isinstance(executable, dict) or set(executable) != {
        "digest", "mlir_sha256", "device_signature", "binary",
        "launch_resources",
    }:
        raise ValueError("executable identity has an invalid schema")
    if (
        executable["digest"] != digest
        or executable["mlir_sha256"] != mlir_sha256
        or not isinstance(executable["device_signature"], list)
        or any(
            not isinstance(value, str)
            for value in executable["device_signature"]
        )
    ):
        raise ValueError("executable identity is inconsistent")
    binary = executable["binary"]
    if (
        not isinstance(binary, dict)
        or set(binary) != {"binary_sha256"}
        or not _is_digest(binary["binary_sha256"])
    ):
        raise ValueError("executable binary identity is invalid")
    launch_resources = executable["launch_resources"]
    allowed_launch_resources = {
        "shared", "num_warps", "num_ctas", "num_stages", "cluster_dims"
    }
    if (
        not isinstance(launch_resources, dict)
        or not set(launch_resources).issubset(allowed_launch_resources)
    ):
        raise ValueError("executable launch_resources is not an object")
    _require_nonnegative_int(launch_resources, "shared")
    for name in ("num_warps", "num_ctas", "num_stages"):
        if name in launch_resources:
            _require_nonnegative_int(launch_resources, name)
    cluster_dims = launch_resources.get("cluster_dims")
    if cluster_dims is not None and (
        not isinstance(cluster_dims, list)
        or not cluster_dims
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in cluster_dims
        )
    ):
        raise ValueError("executable cluster_dims is invalid")
    return executable


def _validate_resource_record(resources: Any) -> dict[str, Any]:
    if not isinstance(resources, dict) or set(resources) != {
        "shared", "shared_residency_tier_kib", "registers",
        "private_words32",
    }:
        raise ValueError("runtime resources have an invalid schema")
    for name in (
        "shared", "shared_residency_tier_kib", "registers",
        "private_words32",
    ):
        _optional_nonnegative_int(resources, name)
    return resources


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def _normalize_timing(value: Any, description: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            or item < 0
            for item in value
        )
        or not value[1] <= value[0] <= value[2]
    ):
        raise ValueError(f"{description} is not an ordered p50/p20/p80 triple")
    return tuple(float(item) for item in value)


def _summarize_v8_samples(samples: list[float]) -> tuple[float, float, float]:
    ordered = sorted(samples)

    def quantile(q: float) -> float:
        point = q * (len(ordered) - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        fraction = point - lower
        return ((1.0 - fraction) * ordered[lower]
                + fraction * ordered[upper])

    return tuple(quantile(q) for q in _V8_MEASUREMENT_PROTOCOL["quantiles"])


def _same_timing(lhs: Iterable[float], rhs: Iterable[float]) -> bool:
    return all(
        math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
        for a, b in zip(lhs, rhs)
    )


def _validate_v8_measurement_rounds(
    records: Any, final_timing: tuple[float, float, float] | None,
) -> list[dict[str, Any]]:
    expected_fields = {
        "round_index", "order_index", "timing_ms", "device_samples_ms",
        "setup_wall_us", "execution_wall_us",
    }
    if not isinstance(records, list):
        raise ValueError("v8 measurement_rounds is not a list")

    normalized = []
    for expected_round, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ValueError("v8 measurement round has an invalid schema")
        round_index = record["round_index"]
        order_index = record["order_index"]
        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or round_index != expected_round
            or round_index > _V8_MAX_REMEASUREMENT_ROUND
            or isinstance(order_index, bool)
            or not isinstance(order_index, int)
            or order_index < 0
        ):
            raise ValueError("v8 measurement rounds are not consecutive")
        timing = _normalize_timing(
            record["timing_ms"], f"v8 measurement round {round_index} timing"
        )
        samples = record["device_samples_ms"]
        walls = (record["setup_wall_us"], record["execution_wall_us"])
        if samples is None:
            if any(wall is not None for wall in walls):
                raise ValueError("v8 unavailable telemetry has partial wall time")
        else:
            if (
                not isinstance(samples, list)
                or len(samples) != _V8_REPEAT_LAUNCHES
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(item)
                    or item < 0
                    for item in samples
                )
                or any(
                    isinstance(wall, bool)
                    or not isinstance(wall, int)
                    or wall < 0
                    for wall in walls
                )
            ):
                raise ValueError("v8 measurement telemetry is invalid")
            if not _same_timing(timing, _summarize_v8_samples(samples)):
                raise ValueError("v8 timing does not summarize raw samples")
        normalized.append(record)

    if final_timing is not None:
        if not normalized:
            raise ValueError("timed v8 variant has no measurement rounds")
        aggregate = tuple(
            statistics.median(record["timing_ms"][index]
                              for record in normalized)
            for index in range(3)
        )
        if not _same_timing(final_timing, aggregate):
            raise ValueError("v8 final timing does not aggregate its rounds")
    return normalized


def _validate_v8_selection(
    selection: Any, variants: list[dict[str, Any]], fallback_digest: str,
    winner_digest: str,
) -> dict[str, Any]:
    expected_fields = {
        "initial_winner", "close", "stable", "selected",
        "fallback_forced",
    }
    if not isinstance(selection, dict) or set(selection) != expected_fields:
        raise ValueError("v8 selection has an invalid schema")
    known = {variant["digest"] for variant in variants}
    timed = {
        variant["digest"] for variant in variants
        if variant["timing_ms"] is not None
    }
    initial = selection["initial_winner"]
    selected = selection["selected"]
    close = selection["close"]
    stable = selection["stable"]
    forced = selection["fallback_forced"]
    if (
        initial not in timed
        or selected != winner_digest
        or selected not in timed
        or not isinstance(close, list)
        or not isinstance(stable, list)
        or any(not isinstance(digest, str) for digest in (*close, *stable))
        or not set(close).issubset(timed)
        or not set(stable).issubset(timed)
        or len(set(close)) != len(close)
        or len(set(stable)) != len(stable)
        or fallback_digest not in stable
        or selected not in stable
        or not isinstance(forced, bool)
        or (forced and selected != fallback_digest)
        or not known.issuperset({initial, selected, *close, *stable})
    ):
        raise ValueError("v8 selection is inconsistent")
    return selection


def _validate_v6_runtime_record(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "version", "manifest_digest", "workload_digest", "workload",
        "effect_digest", "bundle_identity", "winner", "variants",
    }:
        raise ValueError("v6 runtime record has unknown or missing fields")
    if (
        type(payload["version"]) is not int
        or payload["version"] != 6
        or not _is_digest(payload["manifest_digest"])
        or not _is_digest(payload["workload_digest"])
        or not _is_digest(payload["effect_digest"])
        or not isinstance(payload["workload"], dict)
    ):
        raise ValueError("v6 runtime record has invalid identities")

    bundle = payload["bundle_identity"]
    variants = payload["variants"]
    if (
        not isinstance(bundle, list)
        or not bundle
        or not isinstance(variants, list)
        or len(variants) != len(bundle)
    ):
        raise ValueError("v6 runtime record has an invalid closed bundle")
    expected_bundle_fields = {
        "digest", "mlir_file", "mlir_sha256", "stage"
    }
    seen_digests = set()
    seen_files = set()
    for index, identity in enumerate(bundle):
        if (
            not isinstance(identity, dict)
            or set(identity) != expected_bundle_fields
        ):
            raise ValueError("v6 bundle identity has an invalid schema")
        digest = identity["digest"]
        filename = identity["mlir_file"]
        if (
            not _is_digest(digest)
            or filename != f"{digest}.ttgir"
            or not _is_digest(identity["mlir_sha256"])
            or identity["stage"] != "final-ttgir"
            or digest in seen_digests
            or filename in seen_files
        ):
            raise ValueError("v6 bundle identity is inconsistent")
        seen_digests.add(digest)
        seen_files.add(filename)

    expected_variant_fields = {
        "digest", "mlir_file", "mlir_sha256", "timing_ms", "failure",
        "failure_detail", "compile_metrics", "executable", "resources",
    }
    for index, (variant, identity) in enumerate(zip(variants, bundle)):
        if (
            not isinstance(variant, dict)
            or set(variant) != expected_variant_fields
            or any(
                variant[name] != identity[name]
                for name in ("digest", "mlir_file", "mlir_sha256")
            )
        ):
            raise ValueError("v6 runtime variant does not match its bundle identity")
        _validate_compile_metrics(
            variant["compile_metrics"], fallback=index == 0
        )
        executable = variant["executable"]
        resources = variant["resources"]
        failure = variant["failure"]
        failure_detail = variant["failure_detail"]
        if (
            failure is not None and not isinstance(failure, str)
        ) or (
            failure_detail is not None
            and not isinstance(failure_detail, str)
        ):
            raise ValueError("v6 runtime variant has an invalid failure")
        has_failure = failure is not None
        has_failure_detail = failure_detail is not None
        if has_failure != has_failure_detail:
            raise ValueError(
                "v6 runtime failure and failure_detail must be present together"
            )
        if has_failure and (
            not failure
            or not failure_detail
            or "\n" in failure_detail
            or "\r" in failure_detail
            or len(failure_detail.encode("utf-8")) > _MAX_FAILURE_DETAIL_BYTES
        ):
            raise ValueError(
                "v6 runtime failure detail is empty, multiline, or oversized"
            )
        if executable is None:
            if resources is not None or failure is None:
                raise ValueError(
                    "uncompiled v6 variant must carry a failure and no resources"
                )
            if variant["compile_metrics"]["outcome"] != "failure":
                raise ValueError(
                    "uncompiled v6 variant must carry failed compile metrics"
                )
        else:
            _validate_executable_identity(
                executable, variant["digest"], variant["mlir_sha256"]
            )
            _validate_resource_record(resources)
            if variant["compile_metrics"]["outcome"] != "success":
                raise ValueError(
                    "compiled v6 variant must carry successful compile metrics"
                )
        timing = variant["timing_ms"]
        if timing is not None and (
            not isinstance(timing, list)
            or len(timing) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in timing
            )
            or not timing[1] <= timing[0] <= timing[2]
        ):
            raise ValueError("v6 runtime variant has invalid timing")
        if executable is None and timing is not None:
            raise ValueError("uncompiled v6 variant cannot carry timing")
        if (timing is not None) == has_failure:
            raise ValueError(
                "v6 runtime variant must carry exactly one of timing or failure"
            )

    winner = payload["winner"]
    if not isinstance(winner, dict) or set(winner) != {
        "digest", "mlir_sha256", "executable"
    }:
        raise ValueError("v6 winner has an invalid schema")
    winner_variant = next(
        (
            variant for variant in variants
            if variant["digest"] == winner["digest"]
            and variant["mlir_sha256"] == winner["mlir_sha256"]
        ),
        None,
    )
    if winner_variant is None:
        raise ValueError("v6 winner is outside its closed bundle")
    if (
        winner_variant["timing_ms"] is None
        or winner_variant["failure"] is not None
    ):
        raise ValueError("v6 winner has no successful timing")
    executable = _validate_executable_identity(
        winner["executable"], winner["digest"], winner["mlir_sha256"]
    )
    if winner_variant["executable"] != executable:
        raise ValueError("v6 winner executable differs from its variant record")
    return payload


def _validate_v8_runtime_record(payload: Any) -> dict[str, Any]:
    expected_top_level = {
        "version", "manifest_digest", "workload_digest", "workload",
        "effect_digest", "bundle_identity", "winner", "selection",
        "variants",
    }
    if not isinstance(payload, dict) or set(payload) != expected_top_level:
        raise ValueError("v8 runtime record has unknown or missing fields")
    workload = payload["workload"]
    if (
        type(payload["version"]) is not int
        or payload["version"] != 8
        or not _is_digest(payload["manifest_digest"])
        or not _is_digest(payload["workload_digest"])
        or not _is_digest(payload["effect_digest"])
        or not isinstance(workload, dict)
        or workload.get("measurement") != _V8_MEASUREMENT_PROTOCOL
        or hashlib.sha256(_canonical_json(workload).encode("utf-8")).hexdigest()
        != payload["workload_digest"]
    ):
        raise ValueError("v8 runtime record has invalid identities")

    bundle = payload["bundle_identity"]
    variants = payload["variants"]
    if (
        not isinstance(bundle, list)
        or not bundle
        or not isinstance(variants, list)
        or len(variants) != len(bundle)
    ):
        raise ValueError("v8 runtime record has an invalid closed bundle")
    expected_bundle_fields = {
        "digest", "mlir_file", "mlir_sha256", "stage"
    }
    seen_digests = set()
    seen_files = set()
    for index, identity in enumerate(bundle):
        if (
            not isinstance(identity, dict)
            or set(identity) != expected_bundle_fields
            or not _is_digest(identity.get("digest"))
            or identity.get("mlir_file") != f"{identity.get('digest')}.ttgir"
            or not _is_digest(identity.get("mlir_sha256"))
            or identity.get("stage") != "final-ttgir"
            or identity["digest"] in seen_digests
            or identity["mlir_file"] in seen_files
        ):
            raise ValueError("v8 bundle identity is inconsistent")
        seen_digests.add(identity["digest"])
        seen_files.add(identity["mlir_file"])

    expected_variant_fields = {
        "digest", "mlir_file", "mlir_sha256", "timing_ms", "failure",
        "failure_detail", "compile_metrics", "measurement_rounds",
        "executable", "resources",
    }
    occupied_positions = set()
    for index, (variant, identity) in enumerate(zip(variants, bundle)):
        if (
            not isinstance(variant, dict)
            or set(variant) != expected_variant_fields
            or any(
                variant.get(name) != identity[name]
                for name in ("digest", "mlir_file", "mlir_sha256")
            )
        ):
            raise ValueError("v8 runtime variant does not match its bundle")
        metrics = _validate_compile_metrics(
            variant["compile_metrics"], fallback=index == 0
        )
        failure = variant["failure"]
        failure_detail = variant["failure_detail"]
        if (
            (failure is not None and not isinstance(failure, str))
            or (failure_detail is not None and not isinstance(failure_detail, str))
        ):
            raise ValueError("v8 runtime variant has an invalid failure")
        has_failure = failure is not None
        if has_failure != (failure_detail is not None):
            raise ValueError("v8 failure and failure_detail must be paired")
        if has_failure and (
            not failure
            or not failure_detail
            or "\n" in failure_detail
            or "\r" in failure_detail
            or len(failure_detail.encode("utf-8")) > _MAX_FAILURE_DETAIL_BYTES
        ):
            raise ValueError("v8 failure detail is empty, multiline, or oversized")

        timing = (
            _normalize_timing(variant["timing_ms"], "v8 final timing")
            if variant["timing_ms"] is not None else None
        )
        rounds = _validate_v8_measurement_rounds(
            variant["measurement_rounds"], timing
        )
        for round_record in rounds:
            position = (
                round_record["round_index"], round_record["order_index"]
            )
            if position in occupied_positions:
                raise ValueError("v8 measurement order position is duplicated")
            occupied_positions.add(position)

        executable = variant["executable"]
        resources = variant["resources"]
        if executable is None:
            if (
                resources is not None
                or not has_failure
                or timing is not None
                or rounds
                or metrics["outcome"] != "failure"
            ):
                raise ValueError("uncompiled v8 variant has an invalid outcome")
        else:
            _validate_executable_identity(
                executable, variant["digest"], variant["mlir_sha256"]
            )
            _validate_resource_record(resources)
            if metrics["outcome"] != "success":
                raise ValueError("compiled v8 variant has failed compile metrics")
            if (timing is not None) == has_failure:
                raise ValueError(
                    "compiled v8 variant must carry exactly one of timing or failure"
                )

    winner = payload["winner"]
    if not isinstance(winner, dict) or set(winner) != {
        "digest", "mlir_sha256", "executable"
    }:
        raise ValueError("v8 winner has an invalid schema")
    winner_variant = next(
        (
            variant for variant in variants
            if variant["digest"] == winner["digest"]
            and variant["mlir_sha256"] == winner["mlir_sha256"]
        ),
        None,
    )
    if (
        winner_variant is None
        or winner_variant["timing_ms"] is None
        or winner_variant["failure"] is not None
    ):
        raise ValueError("v8 winner is not a successfully measured variant")
    executable = _validate_executable_identity(
        winner["executable"], winner["digest"], winner["mlir_sha256"]
    )
    if winner_variant["executable"] != executable:
        raise ValueError("v8 winner executable differs from its variant record")
    _validate_v8_selection(
        payload["selection"], variants, bundle[0]["digest"], winner["digest"]
    )
    return payload


def _discover_runtime_measurements(
    path: Path,
) -> tuple[list[RuntimeMeasurement], list[str]]:
    root = path if path.is_dir() else path.parent
    candidates = sorted(root.rglob("gluon-layout-autotune.json"))
    measurements = []
    warnings = []
    for candidate in candidates:
        payload = _read_json(candidate)
        if not isinstance(payload, dict) or payload.get("version") not in {4, 5, 6, 8}:
            continue
        if payload["version"] == 6:
            try:
                payload = _validate_v6_runtime_record(payload)
            except ValueError as error:
                warnings.append(
                    f"{candidate}: ignored malformed v6 runtime record: {error}"
                )
                continue
        if payload["version"] == 8:
            try:
                payload = _validate_v8_runtime_record(payload)
            except ValueError as error:
                warnings.append(
                    f"{candidate}: ignored malformed v8 runtime record: {error}"
                )
                continue
        domain = payload.get("manifest_digest")
        variants = payload.get("variants")
        winner = payload.get("winner")
        if (
            not isinstance(domain, str)
            or not SHA256_RE.fullmatch(domain)
            or not isinstance(variants, list)
            or not isinstance(winner, dict)
        ):
            warnings.append(f"{candidate}: ignored malformed runtime record")
            continue
        winner_digest = winner.get("digest")
        winner_mlir = winner.get("mlir_sha256")
        workload_digest = payload.get("workload_digest")
        if not isinstance(workload_digest, str):
            workload_digest = None

        for variant in variants:
            try:
                if not isinstance(variant, dict):
                    raise ValueError("variant entry is not an object")
                digest = variant.get("digest")
                mlir_sha256 = variant.get("mlir_sha256")
                executable = variant.get("executable")
                resources = variant.get("resources")
                failure = variant.get("failure")
                failure_detail = variant.get("failure_detail")
                if (
                    not isinstance(digest, str)
                    or not SHA256_RE.fullmatch(digest)
                    or not isinstance(mlir_sha256, str)
                    or not SHA256_RE.fullmatch(mlir_sha256)
                    or (failure is not None and not isinstance(failure, str))
                    or (
                        failure_detail is not None
                        and not isinstance(failure_detail, str)
                    )
                ):
                    raise ValueError("variant identity or resources are malformed")
                compiled = isinstance(executable, dict) and isinstance(
                    resources, dict
                )
                if compiled:
                    if (
                        executable.get("digest") != digest
                        or executable.get("mlir_sha256") != mlir_sha256
                    ):
                        raise ValueError("variant executable identity is malformed")
                elif executable is not None or resources is not None or failure is None:
                    raise ValueError(
                        "uncompiled variant must carry a failure and no executable"
                    )

                timing = variant.get("timing_ms")
                timing_ms = None
                if not compiled and timing is not None:
                    raise ValueError("uncompiled variant cannot carry timing")
                if timing is not None:
                    if (
                        not isinstance(timing, list)
                        or len(timing) != 3
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or value < 0
                            for value in timing
                        )
                        or not timing[1] <= timing[0] <= timing[2]
                    ):
                        raise ValueError(
                            "timing_ms is not an ordered median/lower/upper triple"
                        )
                    timing_ms = tuple(float(value) for value in timing)

                binary = executable.get("binary") if compiled else None
                binary_sha256 = (
                    binary.get("binary_sha256")
                    if isinstance(binary, dict)
                    and isinstance(binary.get("binary_sha256"), str)
                    and SHA256_RE.fullmatch(binary["binary_sha256"])
                    else None
                )
                private_words32 = (
                    _optional_nonnegative_int(resources, "private_words32")
                    if compiled
                    else None
                )
                compile_metrics = variant.get("compile_metrics")
                if compile_metrics is not None and not isinstance(
                    compile_metrics, dict
                ):
                    raise ValueError("compile_metrics is not an object")
                measurements.append(
                    RuntimeMeasurement(
                        domain=domain,
                        digest=digest,
                        mlir_sha256=mlir_sha256,
                        source=candidate,
                        workload_digest=workload_digest,
                        winner=(
                            digest == winner_digest
                            and mlir_sha256 == winner_mlir
                        ),
                        timing_ms=timing_ms,
                        failure=failure,
                        failure_detail=failure_detail,
                        compile_metrics=compile_metrics,
                        binary_sha256=binary_sha256,
                        shared_bytes=_optional_nonnegative_int(
                            resources, "shared"
                        ) if compiled else None,
                        shared_residency_tier_kib=(
                            _optional_nonnegative_int(
                                resources, "shared_residency_tier_kib"
                            ) if compiled else None
                        ),
                        allocated_registers_per_thread=(
                            _optional_nonnegative_int(resources, "registers")
                            if compiled else None
                        ),
                        local_bytes_per_thread=(
                            private_words32 * 4
                            if private_words32 is not None
                            else None
                        ),
                        measurement_rounds=tuple(
                            variant.get("measurement_rounds", ())
                        ),
                        selection=(
                            payload.get("selection")
                            if payload.get("version") == 8 else None
                        ),
                    )
                )
            except ValueError as error:
                warnings.append(f"{candidate}: ignored runtime variant: {error}")
    measurements.sort(
        key=lambda item: (
            item.domain,
            item.digest,
            item.mlir_sha256,
            item.workload_digest or "",
            str(item.source),
        )
    )
    return measurements, warnings


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="strict")


def _extract_identity(text: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def _index_artifacts(root: Path) -> tuple[list[Path], list[Path]]:
    if root.is_file():
        root = root.parent
    ttgir = {
        *root.rglob("*.ttgir"),
        *root.rglob("*.gluon_ttgir"),
    }
    return sorted(ttgir), sorted(root.rglob("*.llir"))


def _choose_variant_file(
    source: ManifestSource,
    variant: dict[str, Any],
) -> Path | None:
    filename = variant.get("mlir_file")
    digest = variant.get("digest")
    stage = variant.get("stage")
    version = source.manifest.get("version")
    if isinstance(version, int) and version >= 5 and stage != "final-ttgir":
        return None
    if isinstance(filename, str):
        direct = source.base / filename
        if direct.is_file():
            return direct
    return None


def _choose_manifest_ttgir(
    source: ManifestSource, variant: dict[str, Any], ttgir_files: list[Path]
) -> Path | None:
    filename = variant.get("mlir_file")
    if isinstance(filename, str):
        direct = source.base / filename
        if direct.is_file():
            return direct
        matches = [path for path in ttgir_files if path.name == filename]
        if matches:
            return min(matches, key=lambda item: (len(item.parts), str(item)))
    return None


def _llir_metadata_identity(path: Path) -> tuple[str | None, str | None]:
    payload = _read_json(path.with_suffix(".json"))
    if not isinstance(payload, dict):
        return None, None
    domain = payload.get("gluon_layout_domain_digest")
    variant = payload.get("gluon_layout_variant_digest")
    return (
        domain if isinstance(domain, str) else None,
        variant if isinstance(variant, str) else None,
    )


def _choose_llir(
    ttgir: Path, domain: str | None, digest: str, llir_files: list[Path]
) -> Path | None:
    direct = ttgir.with_suffix(".llir")
    if direct.is_file():
        return direct
    matches = [path for path in llir_files if digest in path.stem]
    if not matches:
        return None
    identified = []
    unidentified = []
    for path in matches:
        candidate_domain, candidate_variant = _llir_metadata_identity(path)
        if candidate_variant == digest and (
            domain is None or candidate_domain == domain
        ):
            identified.append(path)
        elif candidate_domain is None and candidate_variant is None:
            unidentified.append(path)
    matches = identified or unidentified
    if not matches:
        return None
    return min(
        matches,
        key=lambda item: (
            item.parent != ttgir.parent,
            len(item.parts),
            str(item),
        ),
    )


def _choose_binary(ttgir: Path, llir: Path | None) -> Path | None:
    candidates = (
        ttgir.with_suffix(".mcfatbin"),
        llir.with_suffix(".mcfatbin") if llir else None,
    )
    for path in candidates:
        if path and path.is_file():
            return path
    return None


def discover_variants(path: Path) -> tuple[list[VariantArtifact], list[str]]:
    root = path if path.is_dir() else path.parent
    ttgir_files, llir_files = _index_artifacts(root)
    warnings = []
    manifests, manifest_warnings = _discover_manifests(path)
    warnings.extend(manifest_warnings)
    artifacts = []

    for source in manifests:
        fallback = source.manifest.get("fallback")
        domain = source.manifest.get("digest")
        for variant in source.manifest.get("variants", []):
            if not isinstance(variant, dict):
                warnings.append(f"{source.source}: ignored malformed variant entry")
                continue
            digest = variant.get("digest")
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                warnings.append(f"{source.source}: ignored variant with invalid digest")
                continue
            ttgir = _choose_variant_file(source, variant)
            if ttgir is None:
                warnings.append(f"{source.source}: missing TTGIR for variant {digest}")
                continue
            llir = _choose_llir(ttgir, domain, digest, llir_files)
            manifest_ttgir = _choose_manifest_ttgir(
                source, variant, ttgir_files
            )
            artifacts.append(
                VariantArtifact(
                    domain=domain,
                    digest=digest,
                    ttgir=ttgir,
                    manifest_ttgir=manifest_ttgir,
                    llir=llir,
                    binary=_choose_binary(ttgir, llir),
                    expected_sha256=variant.get("mlir_sha256"),
                    fallback=digest == fallback,
                    manifest_source=source.source,
                )
            )

    if not artifacts:
        for ttgir in ttgir_files:
            try:
                text = _read_text(ttgir)
            except (OSError, UnicodeDecodeError) as error:
                warnings.append(f"{ttgir}: cannot read TTGIR: {error}")
                continue
            digest = _extract_identity(text, VARIANT_ATTR_RE)
            if digest is None:
                stem_digest = next(
                    (token for token in re.findall(r"[0-9a-f]{64}", ttgir.stem)),
                    None,
                )
                digest = stem_digest or hashlib.sha256(text.encode()).hexdigest()
            domain = _extract_identity(text, DOMAIN_ATTR_RE)
            llir = _choose_llir(ttgir, domain, digest, llir_files)
            artifacts.append(
                VariantArtifact(
                    domain=domain,
                    digest=digest,
                    ttgir=ttgir,
                    manifest_ttgir=None,
                    llir=llir,
                    binary=_choose_binary(ttgir, llir),
                    expected_sha256=None,
                    fallback=False,
                    manifest_source=None,
                )
            )

    unique = {}
    for artifact in artifacts:
        content_sha = hashlib.sha256(artifact.ttgir.read_bytes()).hexdigest()
        key = (artifact.domain, artifact.digest, content_sha)
        previous = unique.get(key)
        if previous is None or (previous.llir is None and artifact.llir is not None):
            unique[key] = artifact
    return sorted(unique.values(), key=lambda item: (
        item.domain or "", not item.fallback, item.digest, str(item.ttgir)
    )), warnings


def _strip_mlir_comments(text: str) -> str:
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def _layout_signatures(text: str) -> dict[str, list[str]]:
    result = defaultdict(set)
    # Layouts may be named at module scope or printed inline in a tensor type
    # (notably dot_op and slice). Scan balanced angle brackets so a nested
    # concrete parent encoding is captured without truncating the child.
    for match in LAYOUT_START_RE.finditer(text):
        depth = 0
        end = None
        for index in range(match.end() - 1, len(text)):
            if text[index] == "<":
                depth += 1
            elif text[index] == ">":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
            elif text[index] == "\n" and depth == 1:
                break
        if end is None:
            continue
        signature = " ".join(text[match.start():end].split())
        result[match.group(1)].add(signature)
    return {kind: sorted(signatures) for kind, signatures in sorted(result.items())}


def audit_ttgir(text: str) -> dict[str, Any]:
    code = _strip_mlir_comments(text)
    operation_counts = {
        name: len(re.findall(pattern, code)) for name, pattern in TTGIR_OPS.items()
    }
    async_counts = Counter(ASYNC_OP_RE.findall(code))
    operation_counts["async_copy"] = sum(async_counts.values())
    gvm_values = Counter(int(value) for value in GVM_NUM_RE.findall(code))
    layouts = _layout_signatures(code)
    return {
        "blocked_signatures": layouts.get("blocked", []),
        "layout_signatures": layouts,
        "operation_counts": operation_counts,
        "async_copy_ops": dict(sorted(async_counts.items())),
        "gvm_arrive_num_histogram": {
            str(value): count for value, count in sorted(gvm_values.items())
        },
    }


def _element_bits(name: str) -> int | None:
    if name in ELEMENT_BITS:
        return ELEMENT_BITS[name]
    if name.startswith("i") and name[1:].isdigit():
        return int(name[1:])
    return None


def _vector_bits(type_text: str) -> int | None:
    match = VECTOR_TYPE_RE.fullmatch(type_text.strip())
    if not match:
        return None
    bits = _element_bits(match.group(2))
    return int(match.group(1)) * bits if bits else None


def _width_histogram(types: Iterable[str]) -> Counter[int]:
    result = Counter()
    for type_text in types:
        bits = _vector_bits(type_text)
        if bits:
            result[bits] += 1
    return result


def _counter_json(counter: Counter[int]) -> dict[str, int]:
    return {str(width): count for width, count in sorted(counter.items())}


def audit_llir(text: str) -> dict[str, Any]:
    code = "\n".join(line.split(";", 1)[0] for line in text.splitlines())
    llvm_loads = _width_histogram(LLVM_VECTOR_LOAD_RE.findall(code))
    llvm_stores = _width_histogram(LLVM_VECTOR_STORE_RE.findall(code))
    mxc_loads = Counter()
    mxc_stores = Counter()
    for line in code.splitlines():
        if not re.search(r"\b(?:call|invoke)\b", line):
            continue
        for kind, lanes, element in MXC_VECTOR_MEMORY_RE.findall(line):
            bits = _element_bits(element)
            if bits:
                (mxc_loads if kind == "ldg" else mxc_stores)[
                    int(lanes) * bits
                ] += 1

    calls = Counter(MXC_CALL_RE.findall(code))
    mma = sum(count for name, count in calls.items() if name.startswith("mma."))
    bsm = sum(count for name, count in calls.items() if name.startswith("bsm."))
    arrive = sum(count for name, count in calls.items() if name == "arrive")
    shared_barrier = sum(
        count for name, count in calls.items() if name == "barrier.shared"
    )
    other_barrier = sum(
        count
        for name, count in calls.items()
        if name.startswith("barrier") and name != "barrier.shared"
    )
    return {
        "llvm_vector_load_width_bits": _counter_json(llvm_loads),
        "llvm_vector_store_width_bits": _counter_json(llvm_stores),
        "mxc_vector_global_load_width_bits": _counter_json(mxc_loads),
        "mxc_vector_global_store_width_bits": _counter_json(mxc_stores),
        "call_counts": {
            "mma": mma,
            "bsm": bsm,
            "arrive": arrive,
            "barrier_shared": shared_barrier,
            "barrier_other": other_barrier,
            "barrier_total": shared_barrier + other_barrier,
        },
    }


def audit_variant(
    artifact: VariantArtifact,
    runtime_measurements: Iterable[RuntimeMeasurement] = (),
) -> dict[str, Any]:
    ttgir_bytes = artifact.ttgir.read_bytes()
    ttgir_sha = hashlib.sha256(ttgir_bytes).hexdigest()
    manifest_ttgir_sha = (
        hashlib.sha256(artifact.manifest_ttgir.read_bytes()).hexdigest()
        if artifact.manifest_ttgir
        else None
    )
    binary_sha = (
        hashlib.sha256(artifact.binary.read_bytes()).hexdigest()
        if artifact.binary
        else None
    )
    report = {
        "domain_digest": artifact.domain,
        "variant_digest": artifact.digest,
        "fallback": artifact.fallback,
        "ttgir": str(artifact.ttgir),
        "ttgir_sha256": ttgir_sha,
        "manifest_ttgir": (
            str(artifact.manifest_ttgir) if artifact.manifest_ttgir else None
        ),
        "manifest_ttgir_sha256": manifest_ttgir_sha,
        "manifest_sha256_match": (
            None
            if artifact.expected_sha256 is None or manifest_ttgir_sha is None
            else artifact.expected_sha256 == manifest_ttgir_sha
        ),
        "manifest_source": (
            str(artifact.manifest_source) if artifact.manifest_source else None
        ),
        "ttgir_audit": audit_ttgir(ttgir_bytes.decode("utf-8")),
        "llir": str(artifact.llir) if artifact.llir else None,
        "llir_audit": None,
        "binary": str(artifact.binary) if artifact.binary else None,
        "binary_sha256": binary_sha,
        "runtime_measurements": [
            measurement.to_json(binary_sha)
            for measurement in runtime_measurements
        ],
    }
    if artifact.llir:
        report["llir_audit"] = audit_llir(_read_text(artifact.llir))
    return report


def _compare_runtime_resources(
    winner: RuntimeMeasurement, fallback: RuntimeMeasurement
) -> str:
    winner_values = (
        winner.allocated_registers_per_thread,
        winner.local_bytes_per_thread,
        winner.shared_bytes,
    )
    fallback_values = (
        fallback.allocated_registers_per_thread,
        fallback.local_bytes_per_thread,
        fallback.shared_bytes,
    )
    if any(value is None for value in (*winner_values, *fallback_values)):
        return "unknown"
    winner_no_worse = all(
        lhs <= rhs for lhs, rhs in zip(winner_values, fallback_values)
    )
    fallback_no_worse = all(
        rhs <= lhs for lhs, rhs in zip(winner_values, fallback_values)
    )
    if winner_no_worse and winner_values != fallback_values:
        return "winner-dominates-fallback"
    if fallback_no_worse and winner_values != fallback_values:
        return "fallback-dominates-winner"
    if winner_values == fallback_values:
        return "equal"
    return "tradeoff"


def _compare_runtime_timings(
    winner: RuntimeMeasurement, fallback: RuntimeMeasurement
) -> tuple[str, float | None]:
    if winner.timing_ms is None or fallback.timing_ms is None:
        return "unknown", None
    winner_median, winner_lower, winner_upper = winner.timing_ms
    fallback_median, fallback_lower, fallback_upper = fallback.timing_ms
    if winner_upper < fallback_lower:
        relation = "winner-strictly-faster"
    elif fallback_upper < winner_lower:
        relation = "fallback-strictly-faster"
    else:
        relation = "intervals-overlap"
    speedup = (
        (fallback_median / winner_median - 1.0) * 100.0
        if winner_median > 0.0
        else None
    )
    return relation, speedup


def _build_runtime_comparisons(
    artifacts: Iterable[VariantArtifact],
    measurements: Iterable[RuntimeMeasurement],
) -> list[dict[str, Any]]:
    fallback_by_domain = {
        artifact.domain: artifact.digest
        for artifact in artifacts
        if artifact.domain and artifact.fallback
    }
    grouped = defaultdict(list)
    for measurement in measurements:
        grouped[
            (
                str(measurement.source),
                measurement.domain,
                measurement.workload_digest,
            )
        ].append(measurement)

    comparisons = []
    for (source, domain, workload_digest), group in sorted(grouped.items()):
        fallback_digest = fallback_by_domain.get(domain)
        winners = [measurement for measurement in group if measurement.winner]
        fallbacks = [
            measurement
            for measurement in group
            if measurement.digest == fallback_digest
        ]
        if len(winners) != 1 or len(fallbacks) != 1:
            continue
        winner = winners[0]
        fallback = fallbacks[0]
        timing_relation, median_speedup_percent = _compare_runtime_timings(
            winner, fallback
        )
        timed = [item for item in group if item.timing_ms is not None]
        best_median = (
            min(timed, key=lambda item: item.timing_ms[0])
            if timed
            else winner
        )
        best_timing_relation, best_median_speedup_percent = (
            _compare_runtime_timings(best_median, fallback)
        )
        comparisons.append({
            "source": source,
            "domain_digest": domain,
            "workload_digest": workload_digest,
            "winner_digest": winner.digest,
            "fallback_digest": fallback.digest,
            "timing_relation": timing_relation,
            "median_speedup_percent": median_speedup_percent,
            "resource_relation": _compare_runtime_resources(
                winner, fallback
            ),
            "best_median_digest": best_median.digest,
            "best_median_timing_relation": best_timing_relation,
            "best_median_speedup_percent": best_median_speedup_percent,
            "best_median_resource_relation": _compare_runtime_resources(
                best_median, fallback
            ),
        })
    return comparisons


def build_report(path: Path) -> dict[str, Any]:
    variants, warnings = discover_variants(path)
    if not variants:
        raise RuntimeError(f"no Gluon layout TTGIR variants found under {path}")
    measurements, measurement_warnings = _discover_runtime_measurements(path)
    warnings.extend(measurement_warnings)
    measurements_by_identity = defaultdict(list)
    for measurement in measurements:
        measurements_by_identity[
            (measurement.domain, measurement.digest, measurement.mlir_sha256)
        ].append(measurement)
    return {
        "schema_version": 4,
        "input": str(path),
        "variant_count": len(variants),
        "warnings": warnings,
        "limitations": [
            "Counts are static textual operation or call sites, not dynamic executions.",
            "TTGIR and LLVM IR do not provide the real allocated register count; only identity-matched runtime records may report it.",
            "MetaX local bytes are per-thread private allocation, not a count of spill/fill instructions.",
            "Vector-width histograms describe emitted LLVM load/store forms and MXC global-memory call forms only.",
            "Timing dominance requires non-overlapping measured lower/upper intervals; a median delta alone is not strict dominance.",
        ],
        "runtime_comparisons": _build_runtime_comparisons(
            variants, measurements
        ),
        "variants": [
            audit_variant(
                artifact,
                measurements_by_identity.get(
                    (
                        artifact.domain,
                        artifact.digest,
                        artifact.expected_sha256
                        or hashlib.sha256(artifact.ttgir.read_bytes()).hexdigest(),
                    ),
                    (),
                ),
            )
            for artifact in variants
        ],
    }


def _format_text(report: dict[str, Any]) -> str:
    lines = [
        f"input: {report['input']}",
        f"variants: {report['variant_count']}",
        "register_count: unavailable (not inferred from TTGIR/LLVM IR)",
    ]
    for warning in report["warnings"]:
        lines.append(f"warning: {warning}")
    for comparison in report["runtime_comparisons"]:
        lines.append(
            "runtime-comparison: "
            + json.dumps(comparison, sort_keys=True)
        )
    for variant in report["variants"]:
        ttgir = variant["ttgir_audit"]
        lines.extend([
            "",
            f"variant {variant['variant_digest']}"
            + (" [fallback]" if variant["fallback"] else ""),
            f"  domain: {variant['domain_digest'] or 'unknown'}",
            f"  ttgir: {variant['ttgir']}",
            "  layout-kinds: "
            + json.dumps(
                {
                    kind: len(signatures)
                    for kind, signatures in ttgir["layout_signatures"].items()
                },
                sort_keys=True,
            ),
            f"  ops: {json.dumps(ttgir['operation_counts'], sort_keys=True)}",
            f"  gvm-arrive-num: {json.dumps(ttgir['gvm_arrive_num_histogram'], sort_keys=True)}",
        ])
        for kind, signatures in ttgir["layout_signatures"].items():
            for signature in signatures:
                lines.append(f"  layout[{kind}]: {signature}")
        if variant["llir_audit"]:
            llir = variant["llir_audit"]
            lines.extend([
                f"  llir: {variant['llir']}",
                f"  llvm-vector-load-bits: {json.dumps(llir['llvm_vector_load_width_bits'], sort_keys=True)}",
                f"  llvm-vector-store-bits: {json.dumps(llir['llvm_vector_store_width_bits'], sort_keys=True)}",
                f"  mxc-vector-global-load-bits: {json.dumps(llir['mxc_vector_global_load_width_bits'], sort_keys=True)}",
                f"  mxc-vector-global-store-bits: {json.dumps(llir['mxc_vector_global_store_width_bits'], sort_keys=True)}",
                f"  calls: {json.dumps(llir['call_counts'], sort_keys=True)}",
            ])
        else:
            lines.append("  llir: unavailable")
        for measurement in variant["runtime_measurements"]:
            lines.append(
                "  runtime-measurement: "
                + json.dumps(measurement, sort_keys=True)
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit final TTGIR and optional LLIR for every Gluon layout variant."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="v4/v5 manifest, kernel metadata JSON, layout bundle, or cache directory",
    )
    parser.add_argument(
        "--format", choices=("json", "text"), default="text", help="report format"
    )
    parser.add_argument("-o", "--output", type=Path, help="write report to this file")
    args = parser.parse_args(argv)

    try:
        report = build_report(args.input.resolve())
    except (OSError, UnicodeDecodeError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    output = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else _format_text(report)
    )
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
