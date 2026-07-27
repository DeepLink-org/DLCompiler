import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def _load_auditor():
    source = (
        Path(__file__).parents[4]
        / "test/MetaX/Inputs/audit_gluon_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_gluon_pipeline_unit_test", source
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_runtime_writer():
    source = (
        Path(__file__).parents[3]
        / "triton/experimental/gluon/_layout_autotune.py"
    )
    spec = importlib.util.spec_from_file_location(
        "gluon_layout_autotune_audit_contract_test", source
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ttgir(domain, digest):
    return f'''#blocked = #ttg.blocked<{{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}}>
#mma = #ttg.maca_mma<{{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], elementsMNK = [1, 2, 8], colMajor = 1, isATrans = false, isBTrans = false, elementsStride = [1, 1]}}>
#dot_a = #ttg.dot_op<{{opIdx = 0, parent = #mma}}>
#dot_b = #ttg.dot_op<{{opIdx = 1, parent = #mma}}>
module attributes {{"ttg.gluon.layout-domain-digest" = "{domain}", "ttg.gluon.layout-variant-digest" = "{digest}"}} {{
  tt.func public @kernel() {{
    ttg.local_load %arg0 : !ttg.memdesc<32xf16, #shared, #smem, mutable> -> tensor<32xf16, #blocked>
    ttg.local_store %arg1, %arg0 : tensor<32xf16, #blocked> -> !ttg.memdesc<32xf16, #shared, #smem, mutable>
    tt.dot %a, %b, %c : tensor<32x32xf16, #dot_a> * tensor<32x32xf16, #dot_b> -> tensor<32x32xf32, #mma>
    "ttg.bsm_perm"(%b) : (tensor<32x32xf16, #dot_b>) -> tensor<32x32xf16, #dot_b>
    ttg.async_copy_global_to_local %src, %arg0 : tensor<32x!tt.ptr<f16>, #blocked> -> !ttg.memdesc<32xf16, #shared, #smem, mutable>
    ttg.gvm_arrive {{num = 4 : i32}}
    ttg.barrier_shared
    ttg.convert_layout %arg1 : tensor<32xf16, #blocked> -> tensor<32xf16, #other>
    tt.return
  }}
}}
'''


def _executable_identity(digest, mlir_sha256, ordinal):
    return {
        "digest": digest,
        "mlir_sha256": mlir_sha256,
        "device_signature": ["*fp16", "i32"],
        "binary": {"binary_sha256": str(ordinal + 1) * 64},
        "launch_resources": {"shared": 32768},
    }


def _v6_runtime_record(domain, variants, winner_digest):
    runtime_variants = []
    for index, variant in enumerate(variants):
        executable = _executable_identity(
            variant["digest"], variant["mlir_sha256"], index
        )
        compile_metrics = (
            {
                "mode": "parent-fallback",
                "outcome": "success",
                "wall_us": 80,
            }
            if index == 0
            else {
                "mode": "isolated-candidate",
                "outcome": "success",
                "cache_hit": False,
                "ir_initialization_us": 10,
                "lowering_stages_us": [["gluon_ttgir", 45]],
                "store_results_us": 5,
                "total_us": 60,
                "isolated_wall_us": 123,
            }
        )
        runtime_variants.append({
            "digest": variant["digest"],
            "mlir_file": variant["mlir_file"],
            "mlir_sha256": variant["mlir_sha256"],
            "timing_ms": [0.1 + index, 0.09 + index, 0.11 + index],
            "failure": None,
            "failure_detail": None,
            "compile_metrics": compile_metrics,
            "executable": executable,
            "resources": {
                "shared": 32768,
                "shared_residency_tier_kib": 32,
                "registers": 48,
                "private_words32": 8,
            },
        })
    winner = next(
        variant
        for variant in runtime_variants
        if variant["digest"] == winner_digest
    )
    return {
        "version": 6,
        "manifest_digest": domain,
        "workload_digest": "c" * 64,
        "workload": {"grid": [1, 1, 1]},
        "effect_digest": "e" * 64,
        "bundle_identity": [dict(variant) for variant in variants],
        "winner": {
            "digest": winner["digest"],
            "mlir_sha256": winner["mlir_sha256"],
            "executable": winner["executable"],
        },
        "variants": runtime_variants,
    }


def _v8_runtime_record(domain, variants, winner_digest):
    runtime_variants = []
    for index, variant in enumerate(variants):
        executable = _executable_identity(
            variant["digest"], variant["mlir_sha256"], index
        )
        compile_metrics = (
            {
                "mode": "parent-fallback",
                "outcome": "success",
                "wall_us": 80,
            }
            if index == 0
            else {
                "mode": "isolated-candidate",
                "outcome": "success",
                "cache_hit": False,
                "ir_initialization_us": 10,
                "lowering_stages_us": [["gluon_ttgir", 45]],
                "store_results_us": 5,
                "total_us": 60,
                "isolated_wall_us": 123,
            }
        )
        samples = [
            0.09 + index,
            0.09 + index,
            0.10 + index,
            0.11 + index,
            0.11 + index,
        ]
        runtime_variants.append({
            "digest": variant["digest"],
            "mlir_file": variant["mlir_file"],
            "mlir_sha256": variant["mlir_sha256"],
            "timing_ms": [0.10 + index, 0.09 + index, 0.11 + index],
            "failure": None,
            "failure_detail": None,
            "compile_metrics": compile_metrics,
            "measurement_rounds": [{
                "round_index": 0,
                "order_index": index,
                "timing_ms": [0.10 + index, 0.09 + index, 0.11 + index],
                "device_samples_ms": samples,
                "setup_wall_us": 1000 + index,
                "execution_wall_us": 2000 + index,
            }],
            "executable": executable,
            "resources": {
                "shared": 32768,
                "shared_residency_tier_kib": 32,
                "registers": 48,
                "private_words32": 8,
            },
        })
    winner = next(
        variant
        for variant in runtime_variants
        if variant["digest"] == winner_digest
    )
    measurement = {
        "version": 8,
        "timer": "device-event-isolated-v1",
        "executor": (
            "forkserver-parallel-compile-persistent-serial-measure-v1"
        ),
        "cache_policy": "restore-scratch-then-clear-before-each-warmup-and-sample",
        "quantiles": [0.5, 0.2, 0.8],
        "warmup_launches": 5,
        "repeat_launches": 5,
        "worker_setup_timeout_seconds": 60,
        "candidate_compile_timeout_seconds": 300,
        "candidate_timeout_seconds": 60,
    }
    workload = {"grid": [1, 1, 1], "measurement": measurement}
    workload_digest = hashlib.sha256(
        json.dumps(
            workload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "version": 8,
        "manifest_digest": domain,
        "workload_digest": workload_digest,
        "workload": workload,
        "effect_digest": "e" * 64,
        "bundle_identity": [dict(variant) for variant in variants],
        "winner": {
            "digest": winner["digest"],
            "mlir_sha256": winner["mlir_sha256"],
            "executable": winner["executable"],
        },
        "selection": {
            "initial_winner": winner["digest"],
            "close": [winner["digest"]],
            "stable": [variants[0]["digest"], winner["digest"]]
            if winner["digest"] != variants[0]["digest"]
            else [winner["digest"]],
            "selected": winner["digest"],
            "fallback_forced": False,
        },
        "variants": runtime_variants,
    }


def test_manifest_bundle_audit_reports_static_ttgir_and_llir_sites(tmp_path):
    auditor = _load_auditor()
    domain = "d" * 64
    digests = ["a" * 64, "b" * 64]
    variants = []
    for digest in digests:
        path = tmp_path / f"{digest}.ttgir"
        path.write_text(_ttgir(domain, digest))
        variants.append({
            "digest": digest,
            "mlir_file": path.name,
            "mlir_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    (tmp_path / "manifest.json").write_text(json.dumps({
        "version": 4,
        "digest": domain,
        "fallback": digests[0],
        "fallback_only": False,
        "variants": variants,
    }))
    runtime_variants = []
    for index, variant in enumerate(variants):
        digest = variant["digest"]
        runtime_variants.append({
            **variant,
            "timing_ms": [0.10 + index, 0.09 + index, 0.11 + index],
            "executable": {
                "digest": digest,
                "mlir_sha256": variant["mlir_sha256"],
                "binary": {"binary_sha256": str(index + 1) * 64},
            },
            "resources": {
                "shared": 32768,
                "shared_residency_tier_kib": 32,
                "registers": 48 + index,
                "private_words32": 12 + index,
            },
        })
    (tmp_path / "gluon-layout-autotune.json").write_text(json.dumps({
        "version": 4,
        "manifest_digest": domain,
        "workload_digest": "c" * 64,
        "winner": {
            "digest": digests[1],
            "mlir_sha256": variants[1]["mlir_sha256"],
        },
        "variants": runtime_variants,
    }))
    (tmp_path / f"{digests[0]}.llir").write_text('''
define void @kernel() {
  %v = load <4 x i32>, ptr addrspace(3) %p, align 16
  store <2 x half> %x, ptr addrspace(1) %q, align 4
  %g = call <8 x half> @llvm.mxc.ldg.predicator.v8f16(ptr %p)
  call void @llvm.mxc.stg.predicator.v2f16(ptr %q, <2 x half> %x)
  %m = call <4 x float> @llvm.mxc.mma.f32.16x16x16f16(<4 x half> %a, <4 x half> %b, <4 x float> %c)
  %r = call <8 x half> @llvm.mxc.bsm.bpermute(<8 x half> %b)
  call void @llvm.mxc.arrive(i32 0)
  call void @llvm.mxc.barrier.shared()
  ret void
}
declare void @llvm.mxc.arrive(i32)
''')
    binary_path = tmp_path / f"{digests[0]}.mcfatbin"
    binary_path.write_bytes(b"compiled-c500-binary")
    binary_sha = hashlib.sha256(binary_path.read_bytes()).hexdigest()
    runtime_variants[0]["executable"]["binary"][
        "binary_sha256"
    ] = binary_sha
    (tmp_path / "gluon-layout-autotune.json").write_text(json.dumps({
        "version": 4,
        "manifest_digest": domain,
        "workload_digest": "c" * 64,
        "winner": {
            "digest": digests[1],
            "mlir_sha256": variants[1]["mlir_sha256"],
        },
        "variants": runtime_variants,
    }))

    report = auditor.build_report(tmp_path)

    assert report["schema_version"] == 4
    assert report["variant_count"] == 2
    assert "identity-matched runtime records" in report["limitations"][1]
    fallback = report["variants"][0]
    assert fallback["fallback"]
    assert fallback["manifest_sha256_match"] is True
    assert fallback["ttgir_audit"]["operation_counts"] == {
        "dot": 1,
        "local_load": 1,
        "local_store": 1,
        "bsm_perm": 1,
        "gvm_arrive": 1,
        "barrier_shared": 1,
        "convert_layout": 1,
        "async_copy": 1,
    }
    assert fallback["ttgir_audit"]["gvm_arrive_num_histogram"] == {"4": 1}
    assert len(fallback["ttgir_audit"]["blocked_signatures"]) == 1
    assert len(fallback["ttgir_audit"]["layout_signatures"]["dot_op"]) == 2
    assert fallback["llir_audit"] == {
        "llvm_vector_load_width_bits": {"128": 1},
        "llvm_vector_store_width_bits": {"32": 1},
        "mxc_vector_global_load_width_bits": {"128": 1},
        "mxc_vector_global_store_width_bits": {"32": 1},
        "call_counts": {
            "mma": 1,
            "bsm": 1,
            "arrive": 1,
            "barrier_shared": 1,
            "barrier_other": 0,
            "barrier_total": 1,
        },
    }
    assert fallback["runtime_measurements"] == [{
        "source": str(tmp_path / "gluon-layout-autotune.json"),
        "workload_digest": "c" * 64,
        "winner": False,
        "timing_ms": [0.1, 0.09, 0.11],
        "failure": None,
        "failure_detail": None,
        "compile_metrics": None,
        "measurement_rounds": [],
        "selection": None,
        "binary_sha256": binary_sha,
        "binary_sha256_match": True,
        "resources": {
            "shared_bytes": 32768,
            "shared_residency_tier_kib": 32,
            "allocated_registers_per_thread": 48,
            "local_bytes_per_thread": 48,
        },
    }]
    assert fallback["binary"] == str(binary_path)
    assert fallback["binary_sha256"] == binary_sha
    assert report["variants"][1]["llir_audit"] is None
    assert report["variants"][1]["runtime_measurements"][0]["winner"] is True
    assert report["variants"][1]["runtime_measurements"][0][
        "binary_sha256_match"
    ] is None
    assert report["variants"][1]["runtime_measurements"][0]["resources"] == {
        "shared_bytes": 32768,
        "shared_residency_tier_kib": 32,
        "allocated_registers_per_thread": 49,
        "local_bytes_per_thread": 52,
    }
    comparison = report["runtime_comparisons"][0]
    assert comparison == {
        "source": str(tmp_path / "gluon-layout-autotune.json"),
        "domain_digest": domain,
        "workload_digest": "c" * 64,
        "winner_digest": digests[1],
        "fallback_digest": digests[0],
        "timing_relation": "fallback-strictly-faster",
        "median_speedup_percent": comparison["median_speedup_percent"],
        "resource_relation": "fallback-dominates-winner",
        "best_median_digest": digests[0],
        "best_median_timing_relation": "intervals-overlap",
        "best_median_speedup_percent": 0.0,
        "best_median_resource_relation": "equal",
    }
    assert round(comparison["median_speedup_percent"], 6) == -90.909091


def test_v5_source_manifest_audits_final_experimental_stage_and_v6_measurement(
    tmp_path,
):
    auditor = _load_auditor()
    domain = "d" * 64
    fallback_digest = "a" * 64
    experiment_digest = "b" * 64
    variants = []
    for index, digest in enumerate((fallback_digest, experiment_digest)):
        source = tmp_path / f"{digest}.ttgir"
        source.write_text(
            _ttgir(domain, digest)
            if index == 0
            else _ttgir(domain, digest).replace("#blocked", "#gluon.auto_encoding")
        )
        variants.append({
            "digest": digest,
            "mlir_file": source.name,
            "mlir_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "stage": "final-ttgir",
        })
    (tmp_path / "manifest.json").write_text(json.dumps({
        "version": 5,
        "digest": domain,
        "fallback": fallback_digest,
        "fallback_only": False,
        "variants": variants,
    }))
    (tmp_path / "gluon-layout-autotune.json").write_text(json.dumps(
        _v6_runtime_record(domain, variants, experiment_digest)
    ))

    report = auditor.build_report(tmp_path)

    experiment = next(
        variant
        for variant in report["variants"]
        if variant["variant_digest"] == experiment_digest
    )
    assert experiment["ttgir"] == str(
        tmp_path / f"{experiment_digest}.ttgir"
    )
    assert experiment["manifest_ttgir"] == str(
        tmp_path / f"{experiment_digest}.ttgir"
    )
    assert experiment["manifest_sha256_match"] is True
    assert "auto_encoding" not in json.dumps(experiment["ttgir_audit"])
    assert experiment["runtime_measurements"][0]["winner"] is True
    assert experiment["runtime_measurements"][0]["compile_metrics"] == {
        "mode": "isolated-candidate",
        "outcome": "success",
        "cache_hit": False,
        "ir_initialization_us": 10,
        "isolated_wall_us": 123,
        "lowering_stages_us": [["gluon_ttgir", 45]],
        "store_results_us": 5,
        "total_us": 60,
    }


def test_v5_finalized_artifacts_are_indexed_by_identity_across_cache_dirs(
    tmp_path,
):
    auditor = _load_auditor()
    domain = "d" * 64
    digests = ["a" * 64, "b" * 64, "c" * 64]
    variants = []
    for index, digest in enumerate(digests):
        source = tmp_path / f"{digest}.ttgir"
        source.write_text(_ttgir(domain, digest))
        variants.append({
            "digest": digest,
            "mlir_file": source.name,
            "mlir_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "stage": (
                "final-ttgir"
            ),
        })
    for cache_name, digest in zip(("cache-b", "cache-c"), digests[1:]):
        finalized = tmp_path / cache_name / "@kernel.gluon_ttgir"
        finalized.parent.mkdir()
        finalized.write_text(_ttgir(domain, digest))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "version": 5,
        "digest": domain,
        "fallback": digests[0],
        "fallback_only": False,
        "variants": variants,
    }))

    report = auditor.build_report(tmp_path)

    assert report["variant_count"] == 3
    by_digest = {
        variant["variant_digest"]: variant for variant in report["variants"]
    }
    assert by_digest[digests[0]]["ttgir"] == str(
        tmp_path / f"{digests[0]}.ttgir"
    )
    for digest in digests[1:]:
        assert by_digest[digest]["ttgir"] == str(
            tmp_path / f"{digest}.ttgir"
        )
        assert by_digest[digest]["manifest_ttgir"] == str(
            tmp_path / f"{digest}.ttgir"
        )


@pytest.mark.parametrize(
    "case",
    [
        "fallback-not-first",
        "unknown-top-level-field",
        "invalid-experiment-stage",
        "inconsistent-fallback-only",
        "noninteger-version",
    ],
)
def test_v5_manifest_schema_is_strict_and_fallback_first(case):
    auditor = _load_auditor()
    variants = [
        {
            "digest": "a" * 64,
            "mlir_file": f"{'a' * 64}.ttgir",
            "mlir_sha256": "1" * 64,
            "stage": "final-ttgir",
        },
        {
            "digest": "b" * 64,
            "mlir_file": f"{'b' * 64}.ttgir",
            "mlir_sha256": "2" * 64,
            "stage": "final-ttgir",
        },
    ]
    manifest = {
        "version": 5,
        "digest": "d" * 64,
        "fallback": "a" * 64,
        "fallback_only": False,
        "variants": variants,
    }
    if case == "fallback-not-first":
        manifest["variants"] = list(reversed(variants))
    elif case == "unknown-top-level-field":
        manifest["unexpected"] = True
    elif case == "invalid-experiment-stage":
        manifest["variants"][1]["stage"] = "pre-finalization-ttgir"
    elif case == "inconsistent-fallback-only":
        manifest["fallback_only"] = True
    else:
        manifest["version"] = 5.0

    assert auditor._extract_manifest(manifest) is None


@pytest.mark.parametrize(
    "case",
    [
        "missing-workload",
        "bundle-order",
        "variant-extra-field",
        "winner-executable",
        "compile-metrics-stage",
        "noninteger-version",
        "failure-detail-missing",
        "timing-and-failure",
        "compiled-without-outcome",
        "empty-failure-detail",
        "multiline-failure-detail",
        "oversized-failure-detail",
    ],
)
def test_v6_runtime_schema_rejects_malformed_records(tmp_path, case):
    auditor = _load_auditor()
    domain = "d" * 64
    variants = [
        {
            "digest": "a" * 64,
            "mlir_file": f"{'a' * 64}.ttgir",
            "mlir_sha256": "1" * 64,
            "stage": "final-ttgir",
        },
        {
            "digest": "b" * 64,
            "mlir_file": f"{'b' * 64}.ttgir",
            "mlir_sha256": "2" * 64,
            "stage": "final-ttgir",
        },
    ]
    record = _v6_runtime_record(domain, variants, "b" * 64)
    if case == "missing-workload":
        del record["workload"]
    elif case == "bundle-order":
        record["bundle_identity"] = list(reversed(record["bundle_identity"]))
    elif case == "variant-extra-field":
        record["variants"][1]["unexpected"] = True
    elif case == "winner-executable":
        del record["winner"]["executable"]["device_signature"]
    elif case == "compile-metrics-stage":
        record["variants"][1]["compile_metrics"]["lowering_stages_us"].append(
            ["gluon_ttgir", 0]
        )
    elif case == "noninteger-version":
        record["version"] = 6.0
    else:
        rejected = record["variants"][1]
        if case == "failure-detail-missing":
            rejected.update(
                timing_ms=None,
                failure="execution-error",
                failure_detail=None,
            )
        elif case == "timing-and-failure":
            rejected.update(
                failure="execution-error",
                failure_detail="device failure",
            )
        elif case == "compiled-without-outcome":
            rejected.update(
                timing_ms=None,
                failure=None,
                failure_detail=None,
            )
        elif case == "empty-failure-detail":
            rejected.update(
                timing_ms=None,
                failure="execution-error",
                failure_detail="",
            )
        elif case == "multiline-failure-detail":
            rejected.update(
                timing_ms=None,
                failure="execution-error",
                failure_detail="first line\nsecond line",
            )
        else:
            rejected.update(
                timing_ms=None,
                failure="execution-error",
                failure_detail="x" * 1200,
            )
    runtime_path = tmp_path / "gluon-layout-autotune.json"
    runtime_path.write_text(json.dumps(record))

    measurements, warnings = auditor._discover_runtime_measurements(tmp_path)

    assert measurements == []
    assert len(warnings) == 1
    assert "ignored malformed v6 runtime record" in warnings[0]


def test_v6_runtime_schema_accepts_bounded_failed_candidate_metrics(tmp_path):
    auditor = _load_auditor()
    domain = "d" * 64
    variants = [
        {
            "digest": "a" * 64,
            "mlir_file": f"{'a' * 64}.ttgir",
            "mlir_sha256": "1" * 64,
            "stage": "final-ttgir",
        },
        {
            "digest": "b" * 64,
            "mlir_file": f"{'b' * 64}.ttgir",
            "mlir_sha256": "2" * 64,
            "stage": "final-ttgir",
        },
    ]
    record = _v6_runtime_record(domain, variants, "a" * 64)
    failed = record["variants"][1]
    failed.update({
        "timing_ms": None,
        "failure": "compile-error",
        "failure_detail": "gluon_ttgir: rejected layout candidate",
        "compile_metrics": {
            "mode": "isolated-candidate",
            "outcome": "failure",
            "isolated_wall_us": 321,
        },
        "executable": None,
        "resources": None,
    })
    runtime_path = tmp_path / "gluon-layout-autotune.json"
    runtime_path.write_text(json.dumps(record))

    measurements, warnings = auditor._discover_runtime_measurements(tmp_path)

    assert warnings == []
    assert len(measurements) == 2
    rejected = next(
        measurement
        for measurement in measurements
        if measurement.digest == "b" * 64
    )
    assert rejected.failure_detail == "gluon_ttgir: rejected layout candidate"
    assert rejected.compile_metrics == {
        "mode": "isolated-candidate",
        "outcome": "failure",
        "isolated_wall_us": 321,
    }


def test_v8_runtime_schema_reports_raw_rounds_and_selection(tmp_path):
    auditor = _load_auditor()
    domain = "d" * 64
    variants = [
        {
            "digest": "a" * 64,
            "mlir_file": f"{'a' * 64}.ttgir",
            "mlir_sha256": "1" * 64,
            "stage": "final-ttgir",
        },
        {
            "digest": "b" * 64,
            "mlir_file": f"{'b' * 64}.ttgir",
            "mlir_sha256": "2" * 64,
            "stage": "final-ttgir",
        },
    ]
    record = _v8_runtime_record(domain, variants, "a" * 64)
    (tmp_path / "gluon-layout-autotune.json").write_text(json.dumps(record))

    measurements, warnings = auditor._discover_runtime_measurements(tmp_path)

    assert warnings == []
    assert len(measurements) == 2
    fallback = next(item for item in measurements if item.digest == "a" * 64)
    assert fallback.measurement_rounds[0]["device_samples_ms"] == [
        0.09, 0.09, 0.10, 0.11, 0.11
    ]
    assert fallback.measurement_rounds[0]["setup_wall_us"] == 1000
    assert fallback.selection == record["selection"]


@pytest.mark.parametrize(
    "case",
    [
        "workload-protocol",
        "workload-digest",
        "raw-count",
        "raw-summary",
        "partial-wall",
        "duplicate-order",
        "selection",
    ],
)
def test_v8_runtime_schema_rejects_malformed_telemetry(tmp_path, case):
    auditor = _load_auditor()
    domain = "d" * 64
    variants = [
        {
            "digest": "a" * 64,
            "mlir_file": f"{'a' * 64}.ttgir",
            "mlir_sha256": "1" * 64,
            "stage": "final-ttgir",
        },
        {
            "digest": "b" * 64,
            "mlir_file": f"{'b' * 64}.ttgir",
            "mlir_sha256": "2" * 64,
            "stage": "final-ttgir",
        },
    ]
    record = _v8_runtime_record(domain, variants, "a" * 64)
    if case == "workload-protocol":
        record["workload"]["measurement"]["repeat_launches"] = 4
    elif case == "workload-digest":
        record["workload_digest"] = "f" * 64
    elif case == "raw-count":
        record["variants"][1]["measurement_rounds"][0][
            "device_samples_ms"
        ].pop()
    elif case == "raw-summary":
        record["variants"][1]["measurement_rounds"][0][
            "device_samples_ms"
        ][2] += 0.5
    elif case == "partial-wall":
        record["variants"][1]["measurement_rounds"][0][
            "setup_wall_us"
        ] = None
    elif case == "duplicate-order":
        record["variants"][1]["measurement_rounds"][0]["order_index"] = 0
    else:
        record["selection"]["stable"] = ["b" * 64]
    (tmp_path / "gluon-layout-autotune.json").write_text(json.dumps(record))

    measurements, warnings = auditor._discover_runtime_measurements(tmp_path)

    assert measurements == []
    assert len(warnings) == 1
    assert "ignored malformed v8 runtime record" in warnings[0]


def test_runtime_writer_v8_record_is_consumed_by_audit_strict_parser(
    tmp_path,
):
    auditor = _load_auditor()
    runtime = _load_runtime_writer()

    domain = "d" * 64
    variants = [
        {
            "digest": "a" * 64,
            "mlir_file": f"{'a' * 64}.ttgir",
            "mlir_sha256": "1" * 64,
            "stage": "final-ttgir",
        },
        {
            "digest": "b" * 64,
            "mlir_file": f"{'b' * 64}.ttgir",
            "mlir_sha256": "2" * 64,
            "stage": "final-ttgir",
        },
        {
            "digest": "c" * 64,
            "mlir_file": f"{'c' * 64}.ttgir",
            "mlir_sha256": "3" * 64,
            "stage": "final-ttgir",
        },
    ]
    manifest_raw = {
        "version": 5,
        "digest": domain,
        "fallback": variants[0]["digest"],
        "fallback_only": False,
        "variants": variants,
    }
    manifest = runtime.parse_layout_manifest(manifest_raw)

    class Kernel:

        def __init__(self, binary):
            self.kernel = binary
            self.metadata = SimpleNamespace(shared=32768)
            self.n_regs = 48
            self.n_spills = 2

    compiled = {}
    for specification, binary in zip(
        (manifest.variants[0], manifest.variants[2]),
        (b"fallback-binary", b"execution-failure-binary"),
    ):
        compiled[specification.digest] = runtime.GluonLayoutCompiledVariant(
            specification.digest,
            specification.mlir_file,
            specification.mlir_sha256,
            ("*fp16", "i32"),
            Kernel(binary),
        )
    compile_metrics = {
        manifest.variants[0].digest: {
            "mode": "parent-fallback",
            "outcome": "success",
            "wall_us": 80,
        },
        manifest.variants[1].digest: {
            "mode": "isolated-candidate",
            "outcome": "failure",
            "isolated_wall_us": 123,
        },
        manifest.variants[2].digest: {
            "mode": "isolated-candidate",
            "outcome": "success",
            "cache_hit": False,
            "ir_initialization_us": 10,
            "lowering_stages_us": [["gluon_ttgir", 20], ["mlir", 30]],
            "store_results_us": 5,
            "total_us": 65,
            "isolated_wall_us": 100,
        },
    }
    bundle = runtime.GluonLayoutVariantBundle(
        manifest_raw,
        compiled,
        compile_failures={
            manifest.variants[1].digest: runtime._CandidateFailureRecord(
                "compile-error", "gluon_ttgir: rejected layout candidate"
            ),
        },
        compile_metrics=compile_metrics,
    )

    class Cache:

        @staticmethod
        def put(data, filename, binary=False):
            assert not binary
            path = tmp_path / filename
            path.write_text(data)
            return str(path)

    fallback = manifest.variants[0]
    execution_failure = manifest.variants[2]
    workload = {
        "grid": [1, 1, 1],
        "measurement": dict(runtime._MEASUREMENT_PROTOCOL),
    }
    workload_digest = runtime._sha256_json(workload)
    measurement_rounds = {
        fallback: [
            runtime._MeasurementRound(
                0, 0, (0.10, 0.09, 0.11), None, None, None
            )
        ]
    }
    selection = {
        "initial_winner": fallback.digest,
        "close": [fallback.digest],
        "stable": [fallback.digest],
        "selected": fallback.digest,
        "fallback_forced": False,
    }
    runtime.CompilerLayoutAutotuner._write_cache(
        Cache(),
        manifest,
        workload_digest,
        workload,
        fallback,
        {fallback: (0.10, 0.09, 0.11)},
        {},
        {execution_failure: "execution-error"},
        {execution_failure: "shared memory allocation failed"},
        bundle,
        "e" * 64,
        [dict(variant) for variant in variants],
        measurement_rounds,
        selection,
    )

    measurements, warnings = auditor._discover_runtime_measurements(tmp_path)

    assert warnings == []
    assert len(measurements) == 3
    by_digest = {
        measurement.digest: measurement for measurement in measurements
    }
    assert by_digest[fallback.digest].timing_ms == (0.10, 0.09, 0.11)
    assert by_digest[fallback.digest].measurement_rounds == ({
        "round_index": 0,
        "order_index": 0,
        "timing_ms": [0.10, 0.09, 0.11],
        "device_samples_ms": None,
        "setup_wall_us": None,
        "execution_wall_us": None,
    },)
    assert by_digest[fallback.digest].selection == selection
    compile_failure = by_digest[manifest.variants[1].digest]
    assert compile_failure.failure == "compile-error"
    assert compile_failure.failure_detail == (
        "gluon_ttgir: rejected layout candidate"
    )
    assert compile_failure.compile_metrics["outcome"] == "failure"
    rejected = by_digest[execution_failure.digest]
    assert rejected.failure == "execution-error"
    assert rejected.failure_detail == "shared memory allocation failed"
    assert rejected.compile_metrics["outcome"] == "success"


def test_unpublished_cached_artifacts_do_not_override_final_manifest_sources(
    tmp_path,
):
    auditor = _load_auditor()
    domain = "d" * 64
    digests = ["a" * 64, "b" * 64]
    variants = []
    for index, digest in enumerate(digests):
        source = tmp_path / f"{digest}.ttgir"
        source.write_text(_ttgir(domain, digest))
        variants.append({
            "digest": digest,
            "mlir_file": source.name,
            "mlir_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "stage": (
                "final-ttgir"
            ),
        })
    for cache_name, suffix in (("cache-1", ""), ("cache-2", "\n// changed")):
        finalized = tmp_path / cache_name / "@kernel.gluon_ttgir"
        finalized.parent.mkdir()
        finalized.write_text(_ttgir(domain, digests[1]) + suffix)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "version": 5,
        "digest": domain,
        "fallback": digests[0],
        "fallback_only": False,
        "variants": variants,
    }))

    artifacts, warnings = auditor.discover_variants(tmp_path)

    assert [artifact.digest for artifact in artifacts] == digests
    assert warnings == []
    assert [artifact.ttgir for artifact in artifacts] == [
        tmp_path / f"{digest}.ttgir" for digest in digests
    ]
