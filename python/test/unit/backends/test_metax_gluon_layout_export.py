import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


def _load_compiler_source(monkeypatch):
    triton = ModuleType("triton")
    triton.knobs = SimpleNamespace()
    backends = ModuleType("triton.backends")
    backend_compiler = ModuleType("triton.backends.compiler")
    backend_compiler.BaseBackend = type("BaseBackend", (), {})
    backend_compiler.GPUTarget = type("GPUTarget", (), {})
    backend_compiler.Language = SimpleNamespace(TRITON="triton", GLUON="gluon")
    libtriton = ModuleType("triton._C.libtriton")
    libtriton.ir = SimpleNamespace()
    libtriton.passes = SimpleNamespace()
    libtriton.llvm = SimpleNamespace()
    libtriton.metax = SimpleNamespace()

    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.backends", backends)
    monkeypatch.setitem(sys.modules, "triton.backends.compiler", backend_compiler)
    monkeypatch.setitem(sys.modules, "triton._C", ModuleType("triton._C"))
    monkeypatch.setitem(sys.modules, "triton._C.libtriton", libtriton)

    backend_dir = Path(__file__).parents[4] / "third_party/metax/backend"
    package_name = "metax_backend_layout_export_source_test"
    package = ModuleType(package_name)
    package.__path__ = [str(backend_dir)]
    monkeypatch.setitem(sys.modules, package_name, package)
    source = backend_dir / "compiler.py"
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.compiler", source
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class _FakeCache:

    def __init__(self):
        self.files = {}

    def put(self, data, filename, binary=True):
        assert binary
        self.files[filename] = data
        return f"/cache/{filename}"


def _runtime_contract():
    return {
        "version": 2,
        "argument_index_space": "lowered-tt-func-runtime-abi",
        "replayable": True,
        "deterministic": True,
        "tensor_args": [0, 1],
        "read_args": [0],
        "written_args": [1],
        "write_only_args": [1],
        "atomic_args": [],
        "required_noalias": [[0, 1]],
    }


def _module_with_attrs(str_attrs=None, int_attrs=None, **extra):
    str_attrs = str_attrs or {}
    int_attrs = int_attrs or {}
    operation = SimpleNamespace(
        get_str_attr=lambda name: str_attrs.get(name)
    )
    return SimpleNamespace(
        get_operation=lambda: operation,
        get_int_attr=lambda name: int_attrs.get(name),
        **extra,
    )


def _write_export(directory, *, domain=None, variants=None, fallback_only=False):
    domain = domain or "d" * 64
    variants = variants or ["a" * 64, "b" * 64]
    records = []
    contract_literal = json.dumps(json.dumps(_runtime_contract(), separators=(",", ":")))
    for index, digest in enumerate(variants):
        filename = f"{digest}.ttgir"
        finalization_attr = ', "ttg.gluon.gvm-finalized" = 1 : i32'
        content = (
            "module attributes {"
            f'"ttg.gluon.layout-domain-digest" = "{domain}", '
            f'"ttg.gluon.layout-variant-digest" = "{digest}", '
            f'"ttg.gluon.layout-runtime-contract" = {contract_literal}'
            f"{finalization_attr}}} {{}}\n"
        ).encode()
        (directory / filename).write_bytes(content)
        records.append({
            "digest": digest,
            "mlir_file": filename,
            "mlir_sha256": hashlib.sha256(content).hexdigest(),
            "stage": "final-ttgir",
        })
    manifest = {
        "version": 5,
        "digest": domain,
        "fallback": variants[0],
        "fallback_only": fallback_only,
        "variants": records,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def _write_sources(directory, *, domain=None, variants=None):
    domain = domain or "d" * 64
    variants = variants or ["a" * 64, "b" * 64]
    records = []
    contract_literal = json.dumps(
        json.dumps(_runtime_contract(), separators=(",", ":"))
    )
    for digest in variants:
        filename = f"{digest}.mlir"
        content = (
            "module attributes {"
            f'"ttg.gluon.layout-domain-digest" = "{domain}", '
            f'"ttg.gluon.layout-variant-digest" = "{digest}", '
            f'"ttg.gluon.layout-runtime-contract" = {contract_literal}'
            "} {}\n"
        ).encode()
        (directory / filename).write_bytes(content)
        records.append({
            "digest": digest,
            "mlir_file": filename,
            "mlir_sha256": hashlib.sha256(content).hexdigest(),
        })
    manifest = {
        "version": 1,
        "digest": domain,
        "fallback": variants[0],
        "fallback_only": len(variants) == 1,
        "variants": records,
    }
    (directory / "sources.json").write_text(json.dumps(manifest))
    return manifest


def test_finalized_bundle_is_published_and_validated_as_v5(
    monkeypatch, tmp_path
):
    compiler = _load_compiler_source(monkeypatch)
    final_dir = tmp_path / "final"
    domain = "d" * 64
    digests = ["a" * 64, "b" * 64]
    source_manifest = {
        "version": 1,
        "digest": domain,
        "fallback": digests[0],
        "fallback_only": False,
        "variants": [{"digest": digest} for digest in digests],
    }
    contract = json.dumps(
        json.dumps(_runtime_contract(), separators=(",", ":"))
    )
    published = [
        (
            digest,
            (
                "module attributes {"
                f'"ttg.gluon.layout-domain-digest" = "{domain}", '
                f'"ttg.gluon.layout-variant-digest" = "{digest}", '
                f'"ttg.gluon.layout-runtime-contract" = {contract}, '
                '"ttg.gluon.gvm-finalized" = 1 : i32} {}\n'
            ).encode(),
            "final-ttgir",
        )
        for digest in digests
    ]
    manifest = compiler.gluon_layout.publish_export(
        str(final_dir), source_manifest, published
    )

    assert manifest["version"] == 5
    assert manifest["fallback"] == source_manifest["fallback"]
    assert manifest["digest"] == source_manifest["digest"]
    export = compiler.gluon_layout.read_export(
        str(final_dir), _runtime_contract()
    )
    assert export["manifest"] == manifest
    assert export["variants"] == manifest["variants"]


def test_final_export_consumes_variant_payloads_incrementally(monkeypatch, tmp_path):
    compiler = _load_compiler_source(monkeypatch)
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source_manifest = _write_sources(
        source_dir,
        variants=["a" * 64, "b" * 64],
    )
    final_dir = tmp_path / "final"
    first_content = b"first candidate"
    second_content = b"second candidate"

    def finalized_variants():
        yield "a" * 64, first_content, "final-ttgir"
        assert (final_dir / f"{'a' * 64}.ttgir.tmp").read_bytes() == first_content
        yield "b" * 64, second_content, "final-ttgir"

    manifest = compiler.gluon_layout.publish_export(
        str(final_dir), source_manifest, finalized_variants()
    )

    assert [variant["digest"] for variant in manifest["variants"]] == [
        "a" * 64,
        "b" * 64,
    ]
    assert not list(final_dir.glob("*.tmp"))


def test_final_export_cleans_streamed_files_when_generation_fails(
    monkeypatch, tmp_path
):
    compiler = _load_compiler_source(monkeypatch)
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source_manifest = _write_sources(
        source_dir, variants=["a" * 64, "b" * 64]
    )
    final_dir = tmp_path / "final"

    def finalized_variants():
        yield "a" * 64, b"fallback", "final-ttgir"
        raise RuntimeError("candidate finalization failed")

    with pytest.raises(RuntimeError, match="candidate finalization failed"):
        compiler.gluon_layout.publish_export(
            str(final_dir), source_manifest, finalized_variants()
        )

    assert not list(final_dir.iterdir())


def test_export_validation_caches_exact_bytes_without_temporary_paths(
    monkeypatch, tmp_path
):
    compiler = _load_compiler_source(monkeypatch)
    manifest = _write_export(tmp_path)
    cache = _FakeCache()

    runtime = ModuleType("triton.runtime")
    cache_module = ModuleType("triton.runtime.cache")
    cache_module.get_cache_manager = lambda key: cache
    monkeypatch.setitem(sys.modules, "triton.runtime", runtime)
    monkeypatch.setitem(sys.modules, "triton.runtime.cache", cache_module)

    export = compiler.gluon_layout.cache_export(
        str(tmp_path), "c" * 64, _runtime_contract()
    )

    assert export["manifest"] == manifest
    assert export["runtime_contract"] == _runtime_contract()
    assert set(cache.files) == {
        variant["mlir_file"] for variant in manifest["variants"]
    }
    for variant in manifest["variants"]:
        content = cache.files[variant["mlir_file"]]
        assert hashlib.sha256(content).hexdigest() == variant["mlir_sha256"]
    assert str(tmp_path) not in json.dumps(export["manifest"])


def test_export_cache_waits_for_complete_manifest_validation(monkeypatch, tmp_path):
    compiler = _load_compiler_source(monkeypatch)
    _write_export(tmp_path)
    (tmp_path / "unexpected").write_text("not in manifest")
    cache = _FakeCache()

    runtime = ModuleType("triton.runtime")
    cache_module = ModuleType("triton.runtime.cache")
    cache_module.get_cache_manager = lambda key: cache
    monkeypatch.setitem(sys.modules, "triton.runtime", runtime)
    monkeypatch.setitem(sys.modules, "triton.runtime.cache", cache_module)

    with pytest.raises(RuntimeError, match="outside its v5 manifest"):
        compiler.gluon_layout.cache_export(
            str(tmp_path), "c" * 64, _runtime_contract()
        )

    assert cache.files == {}


def test_standalone_candidate_stage_accepts_only_cpp_finalized_sources(
    monkeypatch,
):
    compiler = _load_compiler_source(monkeypatch)
    backend = object.__new__(compiler.MACABackend)
    finalized = _module_with_attrs(
        {"ttg.gluon.layout-variant-digest": "a" * 64},
        {"ttg.gluon.gvm-finalized": 1},
    )
    assert (
        backend._finalize_standalone_gluon_ttgir(
            finalized, {}, object(), 80
        )
        is finalized
    )
    stale = _module_with_attrs(
        {"ttg.gluon.layout-variant-digest": "a" * 64},
    )
    with pytest.raises(RuntimeError, match="must be finalized by the C.."):
        backend._finalize_standalone_gluon_ttgir(
            stale, {}, object(), 80
        )


@pytest.mark.parametrize(
    ("candidate", "reproducer", "expected_debug"),
    ((True, False, 0), (True, True, 1), (False, False, 1)),
)
def test_downstream_mlir_reproducer_gate(
    monkeypatch, candidate, reproducer, expected_debug
):
    compiler = _load_compiler_source(monkeypatch)
    if reproducer:
        monkeypatch.setenv("TRITON_METAX_GLUON_LAYOUT_REPRODUCER", "1")
    else:
        monkeypatch.delenv(
            "TRITON_METAX_GLUON_LAYOUT_REPRODUCER", raising=False
        )
    debug_calls = []

    class StopAfterGate(RuntimeError):
        pass

    class PassManager:

        @staticmethod
        def enable_debug():
            debug_calls.append(True)

    compiler.ir.pass_manager = lambda _context: PassManager()
    compiler.passes.ttgpuir = SimpleNamespace(
        add_combine_tensor_select_and_if=lambda _pm: None
    )
    compiler.passes.convert = SimpleNamespace(
        add_scf_to_cf=lambda _pm: (_ for _ in ()).throw(StopAfterGate())
    )
    contract = json.dumps(_runtime_contract(), separators=(",", ":"))
    attributes = (
        {
            "ttg.gluon.layout-domain-digest": "d" * 64,
            "ttg.gluon.layout-variant-digest": "a" * 64,
            "ttg.gluon.layout-runtime-contract": contract,
        }
        if candidate
        else {}
    )
    source = _module_with_attrs(attributes, context=object())

    with pytest.raises(StopAfterGate):
        compiler.MACABackend.make_mlir(
            source, {"num_warps": 4}, object(), 80
        )

    assert len(debug_calls) == expected_debug


@pytest.mark.parametrize(
    "mutate, error",
    [
        (
            lambda manifest, directory: manifest["variants"][0].update(
                mlir_file="nested/variant.ttgir"
            ),
            "invalid identity or filename",
        ),
        (
            lambda manifest, directory: manifest["variants"][0].update(
                mlir_sha256="0" * 64
            ),
            "manifest SHA256",
        ),
        (
            lambda manifest, directory: directory.joinpath(
                manifest["variants"][0]["mlir_file"]
            ).write_text(
                'module attributes {"ttg.gluon.layout-domain-digest" = "'
                + "e" * 64
                + '", "ttg.gluon.layout-variant-digest" = "'
                + manifest["variants"][0]["digest"]
                + '", "ttg.gluon.gvm-finalized" = 1 : i32} {}\n'
            ),
            "manifest SHA256",
        ),
    ],
)
def test_export_validation_rejects_untrusted_files(
    monkeypatch, tmp_path, mutate, error
):
    compiler = _load_compiler_source(monkeypatch)
    manifest = _write_export(tmp_path)
    mutate(manifest, tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match=error):
        compiler.gluon_layout.read_export(
            str(tmp_path), _runtime_contract()
        )


def test_export_validation_treats_finalized_ttgir_as_opaque_bytes(
    monkeypatch, tmp_path
):
    compiler = _load_compiler_source(monkeypatch)
    manifest = _write_export(tmp_path)
    variant = manifest["variants"][0]
    path = tmp_path / variant["mlir_file"]
    content = path.read_text().replace("d" * 64, "e" * 64).encode()
    path.write_bytes(content)
    variant["mlir_sha256"] = hashlib.sha256(content).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    export = compiler.gluon_layout.read_export(
        str(tmp_path), _runtime_contract()
    )
    assert export["manifest"] == manifest
    assert export["runtime_contract"] == _runtime_contract()


def test_export_validation_rejects_files_outside_manifest(monkeypatch, tmp_path):
    compiler = _load_compiler_source(monkeypatch)
    _write_export(tmp_path)
    (tmp_path / "unpublished.tmp").write_text("not part of the closed domain")

    with pytest.raises(RuntimeError, match="outside its v5 manifest"):
        compiler.gluon_layout.read_export(
            str(tmp_path), _runtime_contract()
        )


def test_layout_candidate_kernel_name_remains_single_entry(monkeypatch):
    compiler = _load_compiler_source(monkeypatch)
    llir = "define metaxgpu_kernel void @fallback() {}\n"

    assert compiler.maca_get_kernel_name(llir) == "fallback"
    assert not hasattr(compiler, "maca_get_kernel_names")
    assert compiler.gluon_layout.get_single_kernel_name(llir) == "fallback"
    with pytest.raises(RuntimeError, match="exactly one"):
        compiler.gluon_layout.get_single_kernel_name(
            llir + "define metaxgpu_kernel void @unexpected() {}\n"
        )


def test_independent_ttgir_records_only_proved_module_identity(monkeypatch):
    compiler = _load_compiler_source(monkeypatch)
    domain = "d" * 64
    variant = "a" * 64
    src = _module_with_attrs(
        {
            "ttg.gluon.layout-domain-digest": domain,
            "ttg.gluon.layout-variant-digest": variant,
            "ttg.gluon.layout-runtime-contract": json.dumps(_runtime_contract()),
        }
    )
    metadata = {}

    compiler.gluon_layout.record_variant_identity(src, metadata)

    assert metadata == {
        "gluon_layout_domain_digest": domain,
        "gluon_layout_variant_digest": variant,
        "gluon_layout_runtime_contract": _runtime_contract(),
    }


def test_in_memory_module_identity_is_not_parsed_from_mlir_text(monkeypatch):
    compiler = _load_compiler_source(monkeypatch)

    class TextOnlyModule:
        def __str__(self):
            raise AssertionError("compiler.py must not parse candidate MLIR")

    metadata = {}
    compiler.gluon_layout.record_variant_identity(TextOnlyModule(), metadata)

    assert metadata == {}


def test_independent_ttgir_rejects_partial_identity(monkeypatch):
    compiler = _load_compiler_source(monkeypatch)
    src = _module_with_attrs(
        {"ttg.gluon.layout-domain-digest": "d" * 64}
    )

    with pytest.raises(RuntimeError, match="both domain and variant"):
        compiler.gluon_layout.record_variant_identity(src, {})


def test_independent_ttgir_rejects_invalid_runtime_contract(monkeypatch):
    compiler = _load_compiler_source(monkeypatch)
    src = _module_with_attrs(
        {
            "ttg.gluon.layout-domain-digest": "d" * 64,
            "ttg.gluon.layout-variant-digest": "a" * 64,
            "ttg.gluon.layout-runtime-contract": "{}",
        }
    )

    with pytest.raises(RuntimeError, match="invalid schema"):
        compiler.gluon_layout.record_variant_identity(src, {})


def test_backend_contract_requires_atomic_read_write_consistency(monkeypatch):
    compiler = _load_compiler_source(monkeypatch)
    contract = _runtime_contract()
    contract["atomic_args"] = [1]
    contract["deterministic"] = False
    with pytest.raises(RuntimeError, match="inconsistent effects"):
        compiler.gluon_layout.parse_runtime_contract(json.dumps(contract))

    contract["read_args"] = [0, 1]
    contract["write_only_args"] = []
    assert (
        compiler.gluon_layout.parse_runtime_contract(json.dumps(contract))
        == contract
    )


def test_removed_manual_layout_mode_does_not_split_candidate_cache(monkeypatch):
    compiler = _load_compiler_source(monkeypatch)
    monkeypatch.setenv("MACA_PATH", "/unused")
    monkeypatch.delenv("TRITON_METAX_GLUON_MANUAL_LAYOUTS", raising=False)
    options = compiler.MACAOptions(extern_libs={"libdevice": "/unused/libdevice.bc"})
    baseline = options.hash()
    monkeypatch.setenv("TRITON_METAX_GLUON_MANUAL_LAYOUTS", "1")
    assert options.hash() == baseline
