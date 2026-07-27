import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest


# Load the checkout directly so an older installed Triton cannot satisfy these
# tests with stale runtime code.
SOURCE_ROOT = Path(__file__).parents[3]
AUTOTUNE_SOURCE = SOURCE_ROOT / "triton/experimental/gluon/_layout_autotune.py"
spec = importlib.util.spec_from_file_location("gluon_layout_autotune_source_test", AUTOTUNE_SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

CompilerLayoutAutotuner = module.CompilerLayoutAutotuner
FallbackVariantExecutionError = module.FallbackVariantExecutionError
GluonLayoutCompiledVariant = module.GluonLayoutCompiledVariant
GluonLayoutVariantBundle = module.GluonLayoutVariantBundle
LayoutAutotuneError = module.LayoutAutotuneError
_check_required_noalias = module._check_required_noalias
_device_event_benchmarker = module._device_event_benchmarker
_kernel_resources = module._kernel_resources
_measure_variant = module._measure_variant
bind_runtime_arguments = module.bind_runtime_arguments
build_workload_key = module.build_workload_key
parse_layout_manifest = module.parse_layout_manifest
parse_runtime_contract = module.parse_runtime_contract
select_measured_variant = module.select_measured_variant


def _load_runtime_source(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton.experimental.gluon._layout_autotune", module)
    _load_parallel_source(monkeypatch)
    mixin_source = (
        SOURCE_ROOT
        / "triton/experimental/gluon/_layout_autotune_runtime.py"
    )
    mixin_spec = importlib.util.spec_from_file_location(
        "triton.experimental.gluon._layout_autotune_runtime",
        mixin_source,
    )
    mixin_module = importlib.util.module_from_spec(mixin_spec)
    monkeypatch.setitem(sys.modules, mixin_spec.name, mixin_module)
    mixin_spec.loader.exec_module(mixin_module)
    source = SOURCE_ROOT / "triton/experimental/gluon/_runtime.py"
    runtime_spec = importlib.util.spec_from_file_location(
        "triton.experimental.gluon._runtime_source_test", source,
    )
    runtime_module = importlib.util.module_from_spec(runtime_spec)
    monkeypatch.setitem(sys.modules, runtime_spec.name, runtime_module)
    runtime_spec.loader.exec_module(runtime_module)
    runtime_module.layout_autotune_runtime = mixin_module
    return runtime_module


def _load_parallel_source(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "triton.experimental.gluon._layout_autotune", module
    )
    factory_source = (
        SOURCE_ROOT
        / "triton/experimental/gluon/_layout_autotune_factory.py"
    )
    factory_spec = importlib.util.spec_from_file_location(
        "triton.experimental.gluon._layout_autotune_factory",
        factory_source,
    )
    factory_module = importlib.util.module_from_spec(factory_spec)
    monkeypatch.setitem(sys.modules, factory_spec.name, factory_module)
    factory_spec.loader.exec_module(factory_module)

    parallel_source = (
        SOURCE_ROOT
        / "triton/experimental/gluon/_parallel_compile_autotuner.py"
    )
    parallel_spec = importlib.util.spec_from_file_location(
        "triton.experimental.gluon._parallel_compile_autotuner",
        parallel_source,
    )
    parallel_module = importlib.util.module_from_spec(parallel_spec)
    monkeypatch.setitem(sys.modules, parallel_spec.name, parallel_module)
    parallel_spec.loader.exec_module(parallel_module)
    return factory_module, parallel_module


def _digest(char):
    return char * 64


def _isolated_compile_metrics(wall_us=7):
    return {
        "mode": "isolated-candidate",
        "outcome": "success",
        "cache_hit": False,
        "ir_initialization_us": 1,
        "lowering_stages_us": [["gluon_ttgir", 2], ["mlir", 3]],
        "store_results_us": 4,
        "total_us": 10,
        "isolated_wall_us": wall_us,
    }


class FakeTensor:

    _next_scratch_pointer = 0x100000

    def __init__(self, pointer, *, shape=(8, 16), stride=(16, 1)):
        self._pointer = pointer
        self.shape = shape
        self._stride = stride
        self.dtype = "float16"
        self.device = "maca:0"
        self.copy_sources = []

    def data_ptr(self):
        return self._pointer

    def stride(self):
        return self._stride

    def element_size(self):
        return 2

    def new_empty_strided(self, shape, stride):
        pointer = FakeTensor._next_scratch_pointer
        FakeTensor._next_scratch_pointer += 0x10000
        return FakeTensor(pointer, shape=tuple(shape), stride=tuple(stride))

    def copy_(self, source):
        self.copy_sources.append(source)
        return self


class FakeSource:

    def __init__(self, *, arg_names=("out", "M"), signature=None, constants=None):
        self.signature = signature or {"out": "*fp16", "M": "i32"}
        self.fn = SimpleNamespace(
            arg_names=list(arg_names),
            constexprs=[
                index
                for index, name in enumerate(arg_names)
                if self.signature[name] == "constexpr"
            ],
        )
        self.constants = constants or {}

    def hash(self):
        return _digest("f")


class FakeKernel:

    def __init__(self, name, timing, *, binary=None, source=None):
        self.name = name
        self.timing = timing
        self.hash = f"cache-{name}"
        self.kernel = binary if binary is not None else f"binary-{name}".encode()
        self.src = source or FakeSource()
        self.metadata = SimpleNamespace(
            shared=1024,
            target=SimpleNamespace(backend="maca", arch="c500", warp_size=64),
        )


class FakeCache:

    def __init__(self, root, events=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events = events

    def get_file(self, filename):
        if self.events is not None:
            self.events.append(("read", filename))
        path = self.root / filename
        return str(path) if path.is_file() else None

    def put(self, data, filename, binary=True):
        path = self.root / filename
        path.write_bytes(data) if binary else path.write_text(str(data))
        return str(path)


@pytest.fixture(autouse=True)
def _use_fake_device_identity(monkeypatch):
    monkeypatch.setattr(
        module,
        "_device_identity",
        lambda: {
            "driver": "test.FakeDriver",
            "device_index": 0,
            "target": {"backend": "maca", "arch": "c500", "warp_size": 64},
        },
    )


def _manifest(entries=None):
    if entries is None:
        digests = [_digest(char) for char in ("a", "b", "c", "d")]
        entries = [
            (digest, f"{digest}.ttgir", _digest(str(ordinal)))
            for ordinal, digest in enumerate(digests, start=1)
        ]
    return {
        "version": 5,
        "digest": _digest("e"),
        "fallback": entries[0][0],
        "fallback_only": len(entries) == 1,
        "variants": [
            {
                "digest": digest,
                "mlir_file": mlir_file,
                "mlir_sha256": mlir_sha256,
                "stage": "final-ttgir",
            }
            for ordinal, (digest, mlir_file, mlir_sha256) in enumerate(entries)
        ],
    }


def _contract(*, noalias=None):
    noalias = noalias or []
    tensor_args = sorted({index for pair in noalias for index in pair} | {0})
    return {
        "version": 2,
        "argument_index_space": "lowered-tt-func-runtime-abi",
        "replayable": True,
        "tensor_args": tensor_args,
        "read_args": [],
        "written_args": [0],
        "write_only_args": [0],
        "atomic_args": [],
        "deterministic": True,
        "required_noalias": noalias,
    }


def _make_bundle(kernels, *, manifest=None, contract=None):
    manifest = manifest or _manifest()
    contract = contract or _contract()
    mapping = {}
    for entry, kernel in zip(manifest["variants"], kernels):
        kernel.metadata.gluon_layout_manifest = manifest
        kernel.metadata.gluon_layout_runtime_contract = contract
        kernel.metadata.gluon_layout_domain_digest = manifest["digest"]
        kernel.metadata.gluon_layout_variant_digest = entry["digest"]
        mapping[entry["digest"]] = GluonLayoutCompiledVariant(
            entry["digest"], entry["mlir_file"], entry["mlir_sha256"], ("*fp16", "i32"), kernel,
        )
    assert len(mapping) == len(manifest["variants"])
    compile_metrics = {
        entry["digest"]: (
            {"mode": "parent-fallback", "outcome": "success", "wall_us": 7}
            if index == 0
            else _isolated_compile_metrics()
        )
        for index, entry in enumerate(manifest["variants"])
    }
    return GluonLayoutVariantBundle(
        manifest, mapping, contract, compile_metrics=compile_metrics
    )


def _kernels():
    return [
        FakeKernel("fallback", (1.0, 0.99, 1.01)),
        FakeKernel("winner", (0.70, 0.69, 0.71)),
        FakeKernel("other", (0.90, 0.89, 0.91)),
        FakeKernel("slow", (1.20, 1.19, 1.21)),
    ]


def _cache_factory(tmp_path):
    caches = {}
    return caches, lambda key: caches.setdefault(key, FakeCache(tmp_path / key))


def test_manifest_v5_is_closed_ordered_and_file_identified():
    manifest = parse_layout_manifest(_manifest())
    assert manifest.fallback == manifest.variants[0]
    assert manifest.variants[1].mlir_file == f'{_digest("b")}.ttgir'
    assert manifest.variants[1].mlir_sha256 == _digest("2")

    invalid = _manifest()
    invalid["variants"][1]["mlir_file"] = "../outside.ttgir"
    with pytest.raises(LayoutAutotuneError, match="invalid digest, MLIR file"):
        parse_layout_manifest(invalid)

    extra_top_level = {**_manifest(), "future": True}
    with pytest.raises(LayoutAutotuneError, match="top-level fields"):
        parse_layout_manifest(extra_top_level)
    extra_variant = _manifest()
    extra_variant["variants"][0]["ordinal"] = 0
    with pytest.raises(LayoutAutotuneError, match="variant has unknown"):
        parse_layout_manifest(extra_variant)

    out_of_order = _manifest()
    out_of_order["variants"][0], out_of_order["variants"][1] = (
        out_of_order["variants"][1],
        out_of_order["variants"][0],
    )
    with pytest.raises(LayoutAutotuneError, match="first .* variant .* fallback"):
        parse_layout_manifest(out_of_order)

    floating_version = _manifest()
    floating_version["version"] = 5.0
    with pytest.raises(LayoutAutotuneError, match="manifest version"):
        parse_layout_manifest(floating_version)

    empty = _manifest()
    empty["variants"] = []
    with pytest.raises(LayoutAutotuneError, match="malformed"):
        parse_layout_manifest(empty)


def test_runtime_contract_is_replay_permission_not_a_comparator_contract():
    contract = _contract()
    contract["read_args"] = [0]
    contract["write_only_args"] = []
    parsed = parse_runtime_contract(contract, 1)
    assert parsed.read_args == (0,)
    assert parsed.written_args == (0,)

    contract["write_only_args"] = [0]
    contract["written_args"] = []
    with pytest.raises(LayoutAutotuneError, match="must equal written"):
        parse_runtime_contract(contract, 1)


def test_runtime_uses_effect_driven_scratch_without_a_correctness_comparator():
    assert hasattr(module, "_prepare_scratch_replay")
    assert hasattr(module, "_allocate_scratch_argument")
    assert not hasattr(module, "_validate_variant")
    assert "validator" not in inspect.signature(CompilerLayoutAutotuner.__init__).parameters


def test_workload_key_ignores_pointer_identity_but_tracks_alias_topology():
    identities = {
        "source_identity": {"source_hash": _digest("1")},
        "target_identity": {"backend": "maca", "arch": "c500"},
        "device_identity": {"device_index": 0},
        "domain_digest": _digest("2"),
        "effect_digest": _digest("3"),
    }
    same_a, _ = build_workload_key((4,), {"x": FakeTensor(0x1000)}, **identities)
    same_b, _ = build_workload_key((4, 1, 1), {"x": FakeTensor(0x2000)}, **identities)
    alias, alias_payload = build_workload_key(
        (4,), {"x": FakeTensor(0x1000), "y": FakeTensor(0x1080)}, **identities,
    )
    disjoint, _ = build_workload_key(
        (4,), {"x": FakeTensor(0x1000), "y": FakeTensor(0x4000)}, **identities,
    )
    assert same_a == same_b
    assert alias != disjoint
    assert alias_payload["alias_classes"] == [{"arguments": [0, 1], "overlaps": [[0, 1]]}]


def test_runtime_argument_mapping_skips_constexpr_but_launch_keeps_full_signature():
    source = FakeSource(
        arg_names=("BLOCK", "out", "M"),
        signature={"BLOCK": "constexpr", "out": "*fp16", "M": "i32"},
        constants={(0,): 256},
    )
    kernel = FakeKernel("fallback", (1.0, 0.9, 1.1), source=source)
    out = FakeTensor(0x1000)
    assert bind_runtime_arguments(kernel, {"BLOCK": 256, "out": out, "M": 128}) == (("out", out), ("M", 128))


def test_measurement_times_user_arguments_directly():
    calls = []

    class LaunchKernel:

        function = None
        packed_metadata = ("packed",)

        @property
        def run(self):
            self.function = "initialized-function"

            def launch(*args):
                calls.append(args)

            return launch

        def __getitem__(self, grid):
            raise AssertionError("layout benchmark called CompiledKernel.__getitem__")

    user_arg = object()

    def benchmarker(fn, **kwargs):
        fn()
        assert kwargs == {"quantiles": [0.5, 0.2, 0.8], "warmup": 5, "rep": 5}
        return (1.0, 0.9, 1.1)

    measurement = _measure_variant(
        LaunchKernel(), (1, 1, 1), "stream", [user_arg], benchmarker,
    )
    assert measurement == module._TimingResult((1.0, 0.9, 1.1), None)
    assert calls == [(
        1,
        1,
        1,
        "stream",
        "initialized-function",
        ("packed",),
        None,
        None,
        None,
        user_arg,
    )]


def test_device_event_benchmarker_restores_before_timed_interval(monkeypatch):
    import importlib
    import triton.testing as testing

    driver_module = importlib.import_module("triton.runtime.driver")

    trace = []
    elapsed_pairs = []

    class Event:

        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            trace.append(("record", self))

        def elapsed_time(self, end):
            elapsed_pairs.append((self, end))
            return 1.0

    interface = SimpleNamespace(
        Event=Event,
        synchronize=lambda: trace.append(("synchronize", None)),
    )
    active = SimpleNamespace(
        get_device_interface=lambda: interface,
        get_empty_cache_for_benchmark=lambda: object(),
        clear_cache=lambda _cache: trace.append(("clear", None)),
    )
    monkeypatch.setattr(driver_module, "driver", SimpleNamespace(active=active))
    monkeypatch.setattr(testing, "_summarize_statistics", lambda times, *_: times[:3])

    measurement = _device_event_benchmarker(
        lambda: trace.append(("launch", None)),
        before_each=lambda: trace.append(("reset", None)),
        quantiles=(0.5, 0.2, 0.8),
        warmup=1,
        rep=1,
    )
    assert measurement == module._TimingResult((1.0,), (1.0,))

    record_positions = {
        event: index for index, (kind, event) in enumerate(trace)
        if kind == "record"
    }
    for start, end in elapsed_pairs:
        start_index = record_positions[start]
        end_index = record_positions[end]
        interval = [kind for kind, _ in trace[start_index:end_index + 1]]
        assert interval == ["record", "launch", "record"]
        assert any(kind == "reset" for kind, _ in trace[:start_index])


def test_required_noalias_checks_real_user_views():
    _check_required_noalias([FakeTensor(0x1000), FakeTensor(0x4000)], [(0, 1)])
    with pytest.raises(LayoutAutotuneError, match="no-alias"):
        _check_required_noalias([FakeTensor(0x1000), FakeTensor(0x1080)], [(0, 1)])


def test_resource_record_uses_shared_tier_and_metax_private_words():
    kernel = FakeKernel("kernel", (1.0, 0.9, 1.1))
    kernel.metadata.shared = 16 * 1024 + 1
    kernel.n_regs = 48
    kernel.n_spills = 12
    assert _kernel_resources(kernel) == {
        "shared": 16 * 1024 + 1,
        "shared_residency_tier_kib": 32,
        "registers": 48,
        "private_words32": 12,
    }


def test_controller_accepts_different_binaries_and_memory_hit_skips_preflight(tmp_path, monkeypatch):
    kernels = _kernels()
    bundle = _make_bundle(kernels)
    _, cache_factory = _cache_factory(tmp_path)
    calls = {"effects": 0, "measure": []}
    original_parse = module.parse_runtime_contract

    def parse_contract(*args):
        calls["effects"] += 1
        return original_parse(*args)

    monkeypatch.setattr(module, "parse_runtime_contract", parse_contract)
    controller = CompilerLayoutAutotuner(
        cache_manager_factory=cache_factory,
        measurer=lambda kernel, *_: calls["measure"].append(kernel.name) or kernel.timing,
    )
    args = {"out": FakeTensor(0x1000), "M": 128}
    assert controller.prepare_for_launch(bundle, (1,), None, args) is kernels[1]
    assert calls == {"effects": 1, "measure": ["fallback", "winner", "other", "slow"]}

    assert len({kernel.kernel for kernel in kernels}) == 4
    assert controller.prepare_for_launch(bundle, (1,), None, args) is kernels[1]
    assert calls == {"effects": 1, "measure": ["fallback", "winner", "other", "slow"]}


def test_exact_process_fast_hit_skips_manifest_identity_digest_and_effect_work(tmp_path, monkeypatch):
    kernels = _kernels()
    bundle = _make_bundle(kernels)
    controller = CompilerLayoutAutotuner(
        cache_manager_factory=lambda key: FakeCache(tmp_path / key),
        measurer=lambda kernel, *_: kernel.timing,
    )
    args = {"out": FakeTensor(0x1000), "M": 128}
    assert controller.prepare_for_launch(bundle, (1,), None, args) is kernels[1]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("exact process-local winner entered the validated miss path")

    monkeypatch.setattr(module, "parse_layout_manifest", forbidden)
    monkeypatch.setattr(module, "_runtime_contract_digest", forbidden)
    monkeypatch.setattr(module, "build_workload_key", forbidden)
    monkeypatch.setattr(module, "parse_runtime_contract", forbidden)
    monkeypatch.setattr(module, "_check_required_noalias", forbidden)
    monkeypatch.setattr(module, "_prepare_scratch_replay", forbidden)
    monkeypatch.setattr(controller, "_validate_bundle", forbidden)
    assert controller.prepare_for_launch(bundle, (1, 1, 1), None, args) is kernels[1]


def test_process_fast_key_separates_grid_scalars_tensor_views_and_bundle(tmp_path, monkeypatch):
    kernels = _kernels()
    bundle = _make_bundle(kernels)
    controller = CompilerLayoutAutotuner(
        cache_manager_factory=lambda key: FakeCache(tmp_path / key),
        measurer=lambda kernel, *_: kernel.timing,
    )
    full_preparations = []
    prepare_bundle = controller._prepare_bundle

    def counted_prepare(*args, **kwargs):
        full_preparations.append((args[0], args[1], args[3]))
        return prepare_bundle(*args, **kwargs)

    monkeypatch.setattr(controller, "_prepare_bundle", counted_prepare)
    output = FakeTensor(0x1000)
    base_args = {"out": output, "M": 128}

    assert controller.prepare_for_launch(bundle, (1,), None, base_args) is kernels[1]
    assert controller.prepare_for_launch(bundle, (1, 1, 1), None, base_args) is kernels[1]
    assert len(full_preparations) == 1

    distinct_launches = [
        ((2,), {"out": output, "M": 128}),
        ((1,), {"out": output, "M": 129}),
        ((1,), {"out": FakeTensor(0x1000, shape=(4, 32), stride=(32, 1)), "M": 128}),
        ((1,), {"out": FakeTensor(0x1000, shape=(8, 16), stride=(1, 8)), "M": 128}),
        # Allocation identity is deliberately stricter than the persistent
        # workload key: a new object at the same address still re-enters the
        # fully validated path before it gains a process-local fast entry.
        ((1,), {"out": FakeTensor(0x1000), "M": 128}),
        ((1,), {"out": FakeTensor(0x2000), "M": 128}),
    ]
    for ordinal, (grid, args) in enumerate(distinct_launches, start=2):
        assert controller.prepare_for_launch(bundle, grid, None, args) is kernels[1]
        assert len(full_preparations) == ordinal

    replacement_kernels = _kernels()
    replacement_bundle = _make_bundle(replacement_kernels)
    assert controller.prepare_for_launch(replacement_bundle, (1,), None, base_args) is replacement_kernels[1]
    assert len(full_preparations) == len(distinct_launches) + 2


def test_process_fast_key_observes_dtype_device_alignment_and_alias_inputs():
    lhs = FakeTensor(0x1000)
    rhs = FakeTensor(0x4000)
    base = module._fast_launch_identity((1,), None, {"lhs": lhs, "rhs": rhs})

    lhs.dtype = "bfloat16"
    assert module._fast_launch_identity((1,), None, {"lhs": lhs, "rhs": rhs}) != base
    lhs.dtype = "float16"
    lhs.device = "maca:1"
    assert module._fast_launch_identity((1,), None, {"lhs": lhs, "rhs": rhs}) != base
    lhs.device = "maca:0"
    lhs._pointer = 0x1002
    assert module._fast_launch_identity((1,), None, {"lhs": lhs, "rhs": rhs}) != base
    lhs._pointer = 0x1000
    rhs._pointer = 0x1080
    assert module._fast_launch_identity((1,), None, {"lhs": lhs, "rhs": rhs}) != base


def test_persistent_hit_skips_effect_preflight_and_timing(tmp_path, monkeypatch):
    kernels = _kernels()
    bundle = _make_bundle(kernels)
    _, cache_factory = _cache_factory(tmp_path)
    args = {"out": FakeTensor(0x1000), "M": 128}
    first = CompilerLayoutAutotuner(cache_manager_factory=cache_factory, measurer=lambda kernel, *_: kernel.timing)
    assert first.prepare_for_launch(bundle, (1,), None, args) is kernels[1]

    monkeypatch.setattr(
        module,
        "parse_runtime_contract",
        lambda *_: (_ for _ in ()).throw(AssertionError("persistent hit parsed the effect contract")),
    )
    replay = CompilerLayoutAutotuner(
        cache_manager_factory=cache_factory,
        measurer=lambda *_: (_ for _ in ()).throw(AssertionError("persistent hit timed a kernel")),
    )
    assert replay.prepare_for_launch(bundle, (1,), None, args) is kernels[1]


def test_malformed_v8_telemetry_invalidates_persistent_winner(tmp_path):
    kernels = _kernels()
    bundle = _make_bundle(kernels)
    caches, cache_factory = _cache_factory(tmp_path)
    args = {"out": FakeTensor(0x1000), "M": 128}
    first = CompilerLayoutAutotuner(
        cache_manager_factory=cache_factory,
        measurer=lambda kernel, *_: kernel.timing,
    )
    assert first.prepare_for_launch(bundle, (1,), None, args) is kernels[1]

    cache_path = (
        next(iter(caches.values())).root / "gluon-layout-autotune.json"
    )
    record = json.loads(cache_path.read_text())
    del record["variants"][0]["measurement_rounds"][0]["setup_wall_us"]
    cache_path.write_text(json.dumps(record))

    measured = []
    replay = CompilerLayoutAutotuner(
        cache_manager_factory=cache_factory,
        measurer=lambda kernel, *_: measured.append(kernel.name) or kernel.timing,
    )
    assert replay.prepare_for_launch(bundle, (1,), None, args) is kernels[1]
    assert measured == ["fallback", "winner", "other", "slow"]


def test_persistent_hit_ignores_recompile_cache_hash_for_identical_binaries(tmp_path):
    kernels = _kernels()
    bundle = _make_bundle(kernels)
    _, cache_factory = _cache_factory(tmp_path)
    args = {"out": FakeTensor(0x1000), "M": 128}
    first = CompilerLayoutAutotuner(
        cache_manager_factory=cache_factory,
        measurer=lambda kernel, *_: kernel.timing,
    )
    assert first.prepare_for_launch(bundle, (1,), None, args) is kernels[1]

    rebuilt_kernels = _kernels()
    for ordinal, kernel in enumerate(rebuilt_kernels):
        # Model TRITON_ALWAYS_COMPILE: a fresh compile-cache identity may be
        # assigned even though the emitted device binary is byte-identical.
        kernel.hash = f"fresh-compile-instance-{ordinal}"
    rebuilt_bundle = _make_bundle(rebuilt_kernels)
    replay = CompilerLayoutAutotuner(
        cache_manager_factory=cache_factory,
        measurer=lambda *_: (_ for _ in ()).throw(
            AssertionError("byte-identical rebuilt bundle was remeasured")
        ),
    )
    assert replay.prepare_for_launch(rebuilt_bundle, (1,), None, args) is rebuilt_kernels[1]


def test_cache_identity_contains_every_file_binary_and_launch_resource(tmp_path):
    kernels = _kernels()
    for ordinal, kernel in enumerate(kernels):
        kernel.metadata.shared = 1024 * (ordinal + 1)
    bundle = _make_bundle(kernels)
    caches, cache_factory = _cache_factory(tmp_path)
    controller = CompilerLayoutAutotuner(cache_manager_factory=cache_factory, measurer=lambda kernel, *_: kernel.timing)
    args = {"out": FakeTensor(0x1000), "M": 128}
    assert controller.prepare_for_launch(bundle, (1,), None, args) is kernels[1]

    record = json.loads((next(iter(caches.values())).root / "gluon-layout-autotune.json").read_text())
    assert record["version"] == 8
    assert record["winner"]["digest"] == _digest("b")
    assert record["winner"]["mlir_sha256"] == _digest("2")
    assert record["winner"]["executable"]["launch_resources"]["shared"] == 2048
    assert [item["digest"] for item in record["bundle_identity"]] == [
        _digest("a"), _digest("b"), _digest("c"), _digest("d"),
    ]
    assert [item["mlir_sha256"] for item in record["bundle_identity"]] == [
        _digest("1"), _digest("2"), _digest("3"), _digest("4"),
    ]
    assert [item["stage"] for item in record["bundle_identity"]] == [
        "final-ttgir", "final-ttgir",
        "final-ttgir", "final-ttgir",
    ]
    assert [item["executable"]["launch_resources"]["shared"] for item in record["variants"]] == [
        1024, 2048, 3072, 4096,
    ]
    assert record["selection"] == {
        "initial_winner": _digest("b"),
        "close": [_digest("b")],
        "stable": [_digest("a"), _digest("b"), _digest("c")],
        "selected": _digest("b"),
        "fallback_forced": False,
    }
    for order_index, variant in enumerate(record["variants"]):
        assert variant["measurement_rounds"] == [{
            "round_index": 0,
            "order_index": order_index,
            "timing_ms": variant["timing_ms"],
            "device_samples_ms": None,
            "setup_wall_us": None,
            "execution_wall_us": None,
        }]


@pytest.mark.parametrize(
    ("candidate_remeasurements", "selected", "stable", "fallback_forced"),
    (
        (
            [(0.94, 0.89, 0.99), (0.93, 0.88, 0.98)],
            _digest("b"),
            [_digest("a"), _digest("b")],
            False,
        ),
        (
            [(1.05, 1.00, 1.10), (0.94, 0.89, 0.99)],
            _digest("a"),
            [_digest("a")],
            True,
        ),
    ),
)
def test_cache_records_interleaved_rounds_and_selection_trace(
    tmp_path, candidate_remeasurements, selected, stable, fallback_forced,
):
    manifest = _manifest([
        (_digest("a"), f'{_digest("a")}.ttgir', _digest("1")),
        (_digest("b"), f'{_digest("b")}.ttgir', _digest("2")),
    ])
    kernels = [
        FakeKernel("fallback", (1.00, 0.90, 1.10)),
        FakeKernel("candidate", (0.95, 0.90, 1.00)),
    ]
    bundle = _make_bundle(kernels, manifest=manifest)
    samples = {
        "fallback": [
            (1.00, 0.90, 1.10),
            (1.00, 0.90, 1.10),
            (1.00, 0.90, 1.10),
        ],
        "candidate": [
            (0.95, 0.90, 1.00),
            *candidate_remeasurements,
        ],
    }

    def measure(kernel, *_args):
        return samples[kernel.name].pop(0)

    controller = CompilerLayoutAutotuner(
        cache_manager_factory=lambda key: FakeCache(tmp_path / key),
        measurer=measure,
    )
    assert controller.prepare_for_launch(
        bundle, (1,), None, {"out": FakeTensor(0x1000), "M": 128},
    ) is kernels[0 if fallback_forced else 1]
    record = json.loads(
        next(tmp_path.rglob("gluon-layout-autotune.json")).read_text()
    )
    assert record["selection"] == {
        "initial_winner": _digest("b"),
        "close": [_digest("a"), _digest("b")],
        "stable": stable,
        "selected": selected,
        "fallback_forced": fallback_forced,
    }
    by_digest = {variant["digest"]: variant for variant in record["variants"]}
    assert [
        (round_record["round_index"], round_record["order_index"])
        for round_record in by_digest[_digest("a")]["measurement_rounds"]
    ] == [(0, 0), (1, 0), (2, 1)]
    assert [
        (round_record["round_index"], round_record["order_index"])
        for round_record in by_digest[_digest("b")]["measurement_rounds"]
    ] == [(0, 1), (1, 1), (2, 0)]
    assert all(
        round_record["device_samples_ms"] is None
        and round_record["setup_wall_us"] is None
        and round_record["execution_wall_us"] is None
        for variant in record["variants"]
        for round_record in variant["measurement_rounds"]
    )


def test_cache_persists_production_measurement_telemetry(
    tmp_path, monkeypatch,
):
    import importlib

    driver_module = importlib.import_module("triton.runtime.driver")
    monkeypatch.setattr(
        driver_module,
        "driver",
        SimpleNamespace(
            active=SimpleNamespace(
                get_device_interface=lambda: SimpleNamespace(
                    synchronize=lambda: None,
                ),
            ),
        ),
    )
    kernels = _kernels()
    bundle = _make_bundle(kernels)
    controller = CompilerLayoutAutotuner(
        cache_manager_factory=lambda key: FakeCache(tmp_path / key),
    )

    def measure(kernel, *_args):
        median, lower, upper = kernel.timing
        return module._MeasurementResult(
            kernel.timing,
            _kernel_resources(kernel),
            (lower, lower, median, upper, upper),
            100,
            200,
        )

    monkeypatch.setattr(controller, "_measure", measure)
    assert controller.prepare_for_launch(
        bundle, (1,), None, {"out": FakeTensor(0x1000), "M": 128},
    ) is kernels[1]
    record = json.loads(
        next(tmp_path.rglob("gluon-layout-autotune.json")).read_text()
    )
    for variant in record["variants"]:
        measurement = variant["measurement_rounds"][0]
        median, lower, upper = measurement["timing_ms"]
        assert measurement["device_samples_ms"] == [
            lower, lower, median, upper, upper,
        ]
        assert measurement["setup_wall_us"] == 100
        assert measurement["execution_wall_us"] == 200


@pytest.mark.parametrize("field", ["binary", "resource"])
def test_changed_candidate_identity_forces_full_retune(tmp_path, field):
    kernels = _kernels()
    bundle = _make_bundle(kernels)
    _, cache_factory = _cache_factory(tmp_path)
    args = {"out": FakeTensor(0x1000), "M": 128}
    controller = CompilerLayoutAutotuner(cache_manager_factory=cache_factory, measurer=lambda kernel, *_: kernel.timing)
    assert controller.prepare_for_launch(bundle, (1,), None, args) is kernels[1]

    replacement = _kernels()
    changed = _make_bundle(replacement)
    if field == "binary":
        assert replacement[1].hash == kernels[1].hash
        replacement[1].kernel = b"changed-candidate-binary"
    else:
        replacement[1].metadata.shared = 8192
    measured = []
    replay = CompilerLayoutAutotuner(
        cache_manager_factory=cache_factory,
        measurer=lambda kernel, *_: measured.append(kernel.name) or kernel.timing,
    )
    assert replay.prepare_for_launch(changed, (1,), None, args) is replacement[1]
    assert measured == ["fallback", "winner", "other", "slow"]


def test_atomic_read_write_contract_replays_on_restored_scratch(tmp_path):
    kernels = _kernels()
    atomic = _contract()
    atomic["read_args"] = [0]
    atomic["write_only_args"] = []
    atomic["atomic_args"] = [0]
    atomic["deterministic"] = False
    bundle = _make_bundle(kernels, contract=atomic)
    user_output = FakeTensor(0x1000)
    measured_args = []
    controller = CompilerLayoutAutotuner(
        cache_manager_factory=lambda key: FakeCache(tmp_path / key),
        measurer=lambda kernel, _grid, _stream, args: measured_args.append(args) or kernel.timing,
    )
    assert controller.prepare_for_launch(
        bundle, (1,), None, {"out": user_output, "M": 128},
    ) is kernels[1]
    assert len(measured_args) == 4
    scratch_outputs = [args[0] for args in measured_args]
    assert len({id(value) for value in scratch_outputs}) == 1
    assert scratch_outputs[0] is not user_output
    assert scratch_outputs[0].shape == user_output.shape
    assert scratch_outputs[0].stride() == user_output.stride()
    assert scratch_outputs[0].copy_sources == [user_output] * 4
    assert user_output.copy_sources == []


def test_scratch_replay_rejects_allocator_alias_with_live_input():
    output = FakeTensor(0x1000)
    live_input = FakeTensor(0x4000)
    output.new_empty_strided = lambda shape, stride: FakeTensor(
        live_input.data_ptr(), shape=tuple(shape), stride=tuple(stride),
    )
    contract = _contract(noalias=[[0, 1]])
    contract["read_args"] = [1]
    parsed = parse_runtime_contract(contract, 2)
    with pytest.raises(LayoutAutotuneError, match="aliases a live runtime tensor"):
        module._prepare_scratch_replay([output, live_input], parsed)


def test_scratch_reset_failure_abandons_tuning_but_keeps_real_fallback(tmp_path):
    kernels = _kernels()
    contract = _contract()
    contract["read_args"] = [0]
    contract["write_only_args"] = []
    output = FakeTensor(0x1000)
    scratch = FakeTensor(0x100000)

    def fail_copy(_source):
        raise RuntimeError("copy failed")

    scratch.copy_ = fail_copy
    output.new_empty_strided = lambda _shape, _stride: scratch
    measured = []
    controller = CompilerLayoutAutotuner(
        cache_manager_factory=lambda key: FakeCache(tmp_path / key),
        measurer=lambda kernel, *_: measured.append(kernel.name) or kernel.timing,
    )
    with pytest.warns(RuntimeWarning, match="fell back"):
        selected = controller.prepare_for_launch(
            _make_bundle(kernels, contract=contract),
            (1,),
            None,
            {"out": output, "M": 128},
        )
    assert selected is kernels[0]
    assert measured == []
    assert output.copy_sources == []


def test_alias_violation_falls_back_without_timing(tmp_path):
    kernels = _kernels()
    measured = []
    controller = CompilerLayoutAutotuner(
        cache_manager_factory=lambda key: FakeCache(tmp_path / key),
        measurer=lambda kernel, *_: measured.append(kernel.name) or kernel.timing,
    )

    source = FakeSource(
        arg_names=("out", "input"), signature={"out": "*fp16", "input": "*fp16"},
    )
    alias_kernels = [FakeKernel(kernel.name, kernel.timing, source=source) for kernel in _kernels()]
    alias_contract = _contract(noalias=[[0, 1]])
    alias_contract["read_args"] = [1]
    alias_bundle = _make_bundle(alias_kernels, contract=alias_contract)
    with pytest.warns(RuntimeWarning, match="no-alias"):
        assert controller.prepare_for_launch(
            alias_bundle, (1,), None, {"out": FakeTensor(0x1000), "input": FakeTensor(0x1080)},
        ) is alias_kernels[0]
    assert measured == []


def test_known_experimental_failure_is_eliminated_but_fallback_failure_is_fatal(tmp_path):
    from triton.runtime.errors import OutOfResources

    kernels = _kernels()
    bundle = _make_bundle(kernels)

    def measure(kernel, *_):
        if kernel is kernels[1]:
            raise OutOfResources(96 << 10, 64 << 10, "shared memory")
        return kernel.timing

    controller = CompilerLayoutAutotuner(
        cache_manager_factory=lambda key: FakeCache(tmp_path / "candidate" / key), measurer=measure,
    )
    with pytest.warns(RuntimeWarning, match="rejecting Gluon layout variant"):
        assert controller.prepare_for_launch(
            bundle, (1,), None, {"out": FakeTensor(0x1000), "M": 128},
        ) is kernels[2]
    record = json.loads(
        next((tmp_path / "candidate").rglob("gluon-layout-autotune.json")).read_text()
    )
    failed = next(item for item in record["variants"] if item["digest"] == _digest("b"))
    assert failed["failure"] == "execution-error"
    assert "shared memory" in failed["failure_detail"]

    fatal = CompilerLayoutAutotuner(
        cache_manager_factory=lambda key: FakeCache(tmp_path / "fallback" / key),
        measurer=lambda kernel, *_: (_ for _ in ()).throw(RuntimeError("fallback launch failed")),
    )
    with pytest.raises(FallbackVariantExecutionError, match="mandatory"):
        fatal.prepare_for_launch(bundle, (2,), None, {"out": FakeTensor(0x1000), "M": 128})


def test_full_launch_signature_is_timed_after_effect_ordinal_remap(tmp_path):
    source = FakeSource(
        arg_names=("BLOCK", "out", "M"),
        signature={"BLOCK": "constexpr", "out": "*fp16", "M": "i32"},
        constants={(0,): 256},
    )
    kernels = [FakeKernel(kernel.name, kernel.timing, source=source) for kernel in _kernels()]
    bundle = _make_bundle(kernels)
    out = FakeTensor(0x1000)
    observed = []

    def measure(kernel, grid, stream, args):
        observed.append(args)
        return kernel.timing

    controller = CompilerLayoutAutotuner(
        cache_manager_factory=lambda key: FakeCache(tmp_path / key), measurer=measure,
    )
    assert controller.prepare_for_launch(
        bundle, (1,), None, {"BLOCK": 256, "out": out, "M": 128},
    ) is kernels[1]
    assert len(observed) == 4
    assert all(args[0] == 256 and args[2] == 128 for args in observed)
    assert all(args[1] is observed[0][1] and args[1] is not out for args in observed)
    assert observed[0][1].copy_sources == []


def test_bundle_with_unpublished_variant_falls_back_and_plain_kernel_is_unchanged(tmp_path):
    kernels = _kernels()
    bundle = _make_bundle(kernels)
    bundle.variants[_digest("f")] = bundle.variants.pop(_digest("c"))
    controller = CompilerLayoutAutotuner(cache_manager_factory=lambda key: FakeCache(tmp_path / key))
    with pytest.warns(RuntimeWarning, match="only variants published"):
        assert controller.prepare_for_launch(
            bundle, (1,), None, {"out": FakeTensor(0x1000), "M": 128},
        ) is kernels[0]
    plain = object()
    assert controller.prepare_for_launch(plain, (1,), None, {}) is plain


def _install_variant_files(cache, contents):
    entries = []
    for ordinal, content in enumerate(contents.values()):
        digest = chr(ord("a") + ordinal) * 64
        name = f"{digest}.ttgir"
        sha256 = hashlib.sha256(content).hexdigest()
        entries.append((digest, name, sha256))
        cache.put(content, name, binary=True)
    return _manifest(entries)


def test_frontdoor_validates_all_files_then_incrementally_compiles_exact_sources(tmp_path, monkeypatch):
    runtime_module = _load_runtime_source(monkeypatch)
    events = []
    cache = FakeCache(tmp_path, events)
    manifest = _install_variant_files(cache, {
        "fallback.ttgir": b"fallback module",
        "candidate_b.ttgir": b"candidate b module",
        "candidate_c.ttgir": b"candidate c module",
    })
    contract = _contract()
    outer_source = FakeSource(
        arg_names=("BLOCK", "out", "M"),
        signature={"BLOCK": "constexpr", "out": "*fp16", "M": "i32"},
        constants={(0,): 256},
    )
    fallback = FakeKernel(
        "fallback", (1.0, 0.9, 1.1), binary=b"fallback-binary", source=outer_source,
    )
    fallback.metadata.gluon_layout_manifest = manifest
    fallback.metadata.gluon_layout_runtime_contract = contract
    fallback.metadata.gluon_layout_domain_digest = manifest["digest"]
    fallback.metadata.gluon_layout_variant_digest = manifest["fallback"]

    from triton.runtime import cache as runtime_cache
    monkeypatch.setattr(runtime_cache, "get_cache_manager", lambda key: cache)
    jit = object.__new__(runtime_module.GluonJITFunction)
    by_file = {entry["mlir_file"]: entry for entry in manifest["variants"]}

    def compile_variant(path, *, target, options, **kwargs):
        name = Path(path).name
        events.append(("compile", name))
        entry = by_file[name]
        kernel = FakeKernel(name, (0.8, 0.7, 0.9), binary=f"binary-{name}".encode())
        kernel.src = SimpleNamespace(signature={0: "*fp16", 1: "i32"})
        kernel.metadata.gluon_layout_domain_digest = manifest["digest"]
        kernel.metadata.gluon_layout_variant_digest = entry["digest"]
        kernel.metadata.gluon_layout_runtime_contract = contract
        return kernel

    jit.compile = compile_variant
    monkeypatch.setattr(
        runtime_module.layout_autotune_runtime,
        "_compile_variant_isolated",
        lambda request, _source: (
            compile_variant(request.source_path, target=None, options={}),
            ("*fp16", "i32"),
            _isolated_compile_metrics(),
        ),
    )
    bundle = jit._compile_layout_variant_bundle(fallback, "c500", {"num_warps": 4})
    files = [entry["mlir_file"] for entry in manifest["variants"]]
    assert isinstance(bundle, GluonLayoutVariantBundle)
    assert events[:3] == [("read", name) for name in files]
    assert events[3:] == [("compile", files[0])]
    parsed = parse_layout_manifest(manifest)
    for specification in parsed.variants[1:]:
        bundle.get_or_compile(specification)
    assert events[3:] == [("compile", name) for name in files]
    assert set(bundle.compile_metrics) == {
        specification.digest for specification in parsed.variants
    }
    assert bundle.compile_metrics[parsed.fallback.digest]["mode"] == "parent-fallback"
    assert all(
        bundle.compile_metrics[specification.digest]["isolated_wall_us"] == 7
        for specification in parsed.variants[1:]
    )
    assert len({compiled.kernel.kernel for compiled in bundle.variants.values()}) == 3
    assert bundle.source_kernel is fallback
    assert bundle.fallback_kernel is not fallback
    assert all(compiled.kernel.src is outer_source for compiled in bundle.variants.values())
    out = FakeTensor(0x1000)
    assert module._bind_launch_arguments(
        bundle.specialization_kernel, {"BLOCK": 256, "out": out, "M": 128},
    ) == (("BLOCK", 256), ("out", out), ("M", 128))
    assert bind_runtime_arguments(
        bundle.specialization_kernel, {"BLOCK": 256, "out": out, "M": 128},
    ) == (("out", out), ("M", 128))


def test_metax_launcher_accepts_constexpr_but_excludes_it_from_device_abi(
    monkeypatch,
):
    monkeypatch.syspath_prepend(str(SOURCE_ROOT.parent))
    from third_party.metax.backend.driver import make_launcher

    source = make_launcher(
        {},
        {"BLOCK": "constexpr", "out": "*fp16", "M": "i32"},
        {"ids_of_const_exprs": (0,)},
    )
    # The Python launcher parses all three user arguments (O, O, i), while the
    # device parameter array deliberately omits constexpr arg0.
    assert 'PyArg_ParseTuple(args, "iiiKKOOOOOOOOi"' in source
    assert "void *params[] = { &arg1, &arg2, &global_scratch, &profile_scratch };" in source


def test_frontdoor_file_hash_mismatch_prevents_any_candidate_compilation(tmp_path, monkeypatch):
    runtime_module = _load_runtime_source(monkeypatch)
    events = []
    cache = FakeCache(tmp_path, events)
    manifest = _install_variant_files(cache, {
        "fallback.ttgir": b"fallback module", "candidate.ttgir": b"candidate module",
    })
    candidate_file = manifest["variants"][1]["mlir_file"]
    (tmp_path / candidate_file).write_bytes(b"corrupted after publication")
    contract = _contract()
    fallback = FakeKernel("fallback", (1.0, 0.9, 1.1))
    fallback.metadata.gluon_layout_manifest = manifest
    fallback.metadata.gluon_layout_runtime_contract = contract
    fallback.metadata.gluon_layout_domain_digest = manifest["digest"]
    fallback.metadata.gluon_layout_variant_digest = manifest["fallback"]

    from triton.runtime import cache as runtime_cache
    monkeypatch.setattr(runtime_cache, "get_cache_manager", lambda key: cache)
    jit = object.__new__(runtime_module.GluonJITFunction)
    jit.compile = lambda *_args, **_kwargs: events.append(("compile", None))
    with pytest.raises(RuntimeError, match="MLIR SHA mismatch"):
        jit._compile_layout_variant_bundle(fallback, "c500", {})
    assert not any(event[0] == "compile" for event in events)


def test_frontdoor_without_manifest_is_an_ordinary_kernel_even_with_effect_metadata(monkeypatch):
    runtime_module = _load_runtime_source(monkeypatch)
    fallback = FakeKernel("ordinary", (1.0, 0.9, 1.1))
    fallback.metadata.gluon_layout_runtime_contract = _contract()
    jit = object.__new__(runtime_module.GluonJITFunction)
    jit.compile = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ordinary kernel compiled variants"))
    assert jit._compile_layout_variant_bundle(fallback, "c500", {}) is fallback


def test_frontdoor_filters_experimental_compile_failure_but_fallback_failure_is_fatal(tmp_path, monkeypatch):
    runtime_module = _load_runtime_source(monkeypatch)
    cache = FakeCache(tmp_path)
    manifest = _install_variant_files(cache, {
        "fallback.ttgir": b"fallback module",
        "rejected.ttgir": b"rejected module",
        "survivor.ttgir": b"survivor module",
    })
    contract = _contract()
    outer = FakeKernel("outer-source", (1.0, 0.9, 1.1))
    outer.metadata.gluon_layout_manifest = manifest
    outer.metadata.gluon_layout_runtime_contract = contract
    outer.metadata.gluon_layout_domain_digest = manifest["digest"]
    outer.metadata.gluon_layout_variant_digest = manifest["fallback"]
    by_file = {entry["mlir_file"]: entry for entry in manifest["variants"]}
    rejected_file = manifest["variants"][1]["mlir_file"]

    from triton.runtime import cache as runtime_cache
    monkeypatch.setattr(runtime_cache, "get_cache_manager", lambda key: cache)
    jit = object.__new__(runtime_module.GluonJITFunction)

    def compile_with_rejection(path, **_kwargs):
        name = Path(path).name
        if name == rejected_file:
            raise RuntimeError("candidate lowering is unsupported")
        entry = by_file[name]
        kernel = FakeKernel(name, (0.8, 0.7, 0.9))
        kernel.src = SimpleNamespace(signature={0: "*fp16", 1: "i32"})
        kernel.metadata.gluon_layout_domain_digest = manifest["digest"]
        kernel.metadata.gluon_layout_variant_digest = entry["digest"]
        kernel.metadata.gluon_layout_runtime_contract = contract
        return kernel

    jit.compile = compile_with_rejection
    monkeypatch.setattr(
        runtime_module.layout_autotune_runtime,
        "_compile_variant_isolated",
        lambda request, _source: (
            compile_with_rejection(request.source_path),
            ("*fp16", "i32"),
            _isolated_compile_metrics(),
        ),
    )
    bundle = jit._compile_layout_variant_bundle(outer, "c500", {})
    parsed = parse_layout_manifest(manifest)
    with pytest.raises(Exception, match="candidate lowering is unsupported"):
        bundle.get_or_compile(parsed.variants[1])
    bundle.get_or_compile(parsed.variants[2])
    assert set(bundle.variants) == {
        manifest["fallback"], manifest["variants"][2]["digest"]
    }
    assert bundle.source_kernel is outer

    jit.compile = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fallback lowering failed"))
    with pytest.raises(RuntimeError, match="mandatory Gluon layout fallback"):
        jit._compile_layout_variant_bundle(outer, "c500", {})


@pytest.mark.parametrize(
    ("violation", "message"),
    (
        ("identity", "changed its identity"),
        ("device_signature", "changed the device ABI signature"),
        ("launcher", "initialized its launcher too early"),
    ),
)
def test_frontdoor_post_compile_invariant_violation_rejects_whole_bundle(
    tmp_path, monkeypatch, violation, message,
):
    runtime_module = _load_runtime_source(monkeypatch)
    cache = FakeCache(tmp_path)
    manifest = _install_variant_files(cache, {
        "fallback.ttgir": b"fallback module", "candidate.ttgir": b"candidate module",
    })
    contract = _contract()
    outer = FakeKernel("outer-source", (1.0, 0.9, 1.1))
    outer.metadata.gluon_layout_manifest = manifest
    outer.metadata.gluon_layout_runtime_contract = contract
    by_file = {entry["mlir_file"]: entry for entry in manifest["variants"]}
    candidate_file = manifest["variants"][1]["mlir_file"]

    from triton.runtime import cache as runtime_cache
    monkeypatch.setattr(runtime_cache, "get_cache_manager", lambda key: cache)
    jit = object.__new__(runtime_module.GluonJITFunction)

    def compile_variant(path, **_kwargs):
        name = Path(path).name
        entry = by_file[name]
        kernel = FakeKernel(name, (0.8, 0.7, 0.9))
        kernel.src = SimpleNamespace(signature={0: "*fp16", 1: "i32"})
        kernel.metadata.gluon_layout_domain_digest = manifest["digest"]
        kernel.metadata.gluon_layout_variant_digest = entry["digest"]
        kernel.metadata.gluon_layout_runtime_contract = contract
        if name == candidate_file and violation == "identity":
            kernel.metadata.gluon_layout_variant_digest = _digest("f")
        if name == candidate_file and violation == "device_signature":
            kernel.src.signature[1] = "i64"
        if name == candidate_file and violation == "launcher":
            kernel.module = object()
        return kernel

    jit.compile = compile_variant
    def compile_isolated(request, _source):
        kernel = compile_variant(request.source_path)
        return (
            kernel,
            tuple(str(item) for item in kernel.src.signature.values()),
            _isolated_compile_metrics(),
        )

    monkeypatch.setattr(
        runtime_module.layout_autotune_runtime,
        "_compile_variant_isolated",
        compile_isolated,
    )
    bundle = jit._compile_layout_variant_bundle(outer, "c500", {})
    with pytest.raises(RuntimeError, match=message):
        bundle.get_or_compile(parse_layout_manifest(manifest).variants[1])


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sync_and_async_compile_frontdoors_return_closed_bundle(monkeypatch, asynchronous):
    runtime_module = _load_runtime_source(monkeypatch)
    from triton._C import libtriton
    from triton.runtime import _async_compile
    from triton.runtime import cache as runtime_cache

    jit = object.__new__(runtime_module.GluonJITFunction)
    kernel_cache = {}
    backend = object()
    jit.device_caches = {0: (kernel_cache, None, "c500", backend, None)}
    jit._call_hook = lambda *_: False
    source = object()
    jit.ASTSource = lambda *_: source
    fallback = FakeKernel("fallback", (1.0, 0.9, 1.1))
    bundle = object()
    events = []
    env_snapshots = []

    def get_env_snapshot():
        snapshot = {"generation": len(env_snapshots)}
        env_snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(libtriton, "get_cache_invalidating_env_vars", get_env_snapshot)

    def compile_fallback(compiled_source, **kwargs):
        events.append(("fallback", compiled_source, kwargs))
        return fallback

    def close_bundle(compiled_fallback, target, options, env_vars=None):
        events.append(("bundle", compiled_fallback, target, options, env_vars))
        return bundle

    jit.compile = compile_fallback
    jit._compile_layout_variant_bundle = close_bundle
    options = SimpleNamespace(num_warps=4)

    class Future:

        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class AsyncMode:

        def submit(self, cache_key, compile_fn, finalize_fn):
            events.append(("submit", cache_key))
            value = compile_fn()
            finalize_fn(value)
            return Future(value)

    mode = AsyncMode() if asynchronous else None
    monkeypatch.setattr(_async_compile, "active_mode", SimpleNamespace(get=lambda: mode))
    monkeypatch.setattr(runtime_cache, "get_cache_key", lambda *_: "async-cache-key")
    result = jit._do_compile("jit-key", {}, 0, {}, options, {}, False)
    if asynchronous:
        assert result.result() is bundle
        assert events[0] == ("submit", "async-cache-key")
    else:
        assert result is bundle
        assert events[0][0] == "fallback"
    assert kernel_cache["jit-key"] is bundle
    assert [event[0] for event in events].count("fallback") == 1
    assert [event[0] for event in events].count("bundle") == 1
    assert len(env_snapshots) == 1
    fallback_event = next(event for event in events if event[0] == "fallback")
    bundle_event = next(event for event in events if event[0] == "bundle")
    assert fallback_event[2]["_env_vars"] is env_snapshots[0]
    assert bundle_event[4] is env_snapshots[0]


def test_runtime_launch_path_neither_reads_mlir_nor_compiles_candidates(monkeypatch):
    runtime_module = _load_runtime_source(monkeypatch)
    jit = object.__new__(runtime_module.GluonJITFunction)
    selected = object()
    jit._layout_autotune = SimpleNamespace(
        prepare_for_launch=lambda kernel, grid, stream, args: selected,
    )
    jit.compile = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("runtime compiled a candidate"))
    assert jit._prepare_kernel_for_launch(object(), (1,), None, {"M": 128}) is selected


def test_jit_layout_autotune_defaults_on_and_can_be_disabled(monkeypatch):
    runtime_module = _load_runtime_source(monkeypatch)

    def kernel():
        pass

    enabled = runtime_module.jit(kernel)
    disabled = runtime_module.jit(layout_autotune=False)(kernel)

    assert enabled.layout_autotune_enabled is True
    assert disabled.layout_autotune_enabled is False
    assert enabled._layout_autotune is not None
    assert disabled._layout_autotune is None

    with pytest.raises(TypeError, match="layout_autotune must be a bool"):
        runtime_module.jit(layout_autotune=1)(kernel)


def test_layout_autotune_disabled_uses_compiler_fallback(monkeypatch):
    runtime_module = _load_runtime_source(monkeypatch)
    kernels = _kernels()
    bundle = _make_bundle(kernels)
    jit = object.__new__(runtime_module.GluonJITFunction)
    jit.layout_autotune_enabled = False
    jit._layout_autotune = SimpleNamespace(
        prepare_for_launch=lambda *_: (_ for _ in ()).throw(
            AssertionError("layout autotuner was invoked")
        ),
    )

    assert (
        jit._compile_layout_variant_bundle(kernels[0], None, None)
        is kernels[0]
    )
    assert jit._prepare_kernel_for_launch(bundle, (1,), None, {}) is kernels[0]


def test_runtime_prepare_exception_returns_executable_fallback_not_bundle(monkeypatch):
    runtime_module = _load_runtime_source(monkeypatch)
    kernels = _kernels()
    bundle = _make_bundle(kernels)
    jit = object.__new__(runtime_module.GluonJITFunction)
    jit._layout_autotune = SimpleNamespace(
        prepare_for_launch=lambda *_: (_ for _ in ()).throw(RuntimeError("selection failed")),
    )
    with pytest.warns(RuntimeWarning, match="selection failed"):
        assert jit._prepare_kernel_for_launch(bundle, (1,), None, {}) is kernels[0]

    jit._layout_autotune = SimpleNamespace(
        prepare_for_launch=lambda *_: (_ for _ in ()).throw(FallbackVariantExecutionError("fallback failed")),
    )
    with pytest.raises(FallbackVariantExecutionError, match="fallback failed"):
        jit._prepare_kernel_for_launch(bundle, (1,), None, {})


def test_selection_requires_a_real_speedup():
    manifest = parse_layout_manifest(_manifest([
        (_digest("a"), f'{_digest("a")}.ttgir', _digest("1")),
        (_digest("b"), f'{_digest("b")}.ttgir', _digest("2")),
    ]))
    fallback, candidate = manifest.variants
    assert select_measured_variant(
        {fallback: (1.0, 0.9, 1.1), candidate: (1.001, 0.8, 1.2)}, fallback,
    ) is fallback


def test_controller_uses_fixed_device_event_timer():
    assert CompilerLayoutAutotuner()._get_benchmarker() is _device_event_benchmarker
    assert module._MEASUREMENT_PROTOCOL == {
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


@pytest.mark.parametrize(
    "round_record",
    (
        {
            "round_index": 0,
            "order_index": 0,
            "timing_ms": [1.0, 1.0, 1.0],
            "device_samples_ms": [1.0] * 4,
            "setup_wall_us": 10,
            "execution_wall_us": 20,
        },
        {
            "round_index": 0,
            "order_index": 0,
            "timing_ms": [1.0, 1.0, 1.0],
            "device_samples_ms": [1.0] * 5,
            "setup_wall_us": None,
            "execution_wall_us": 20,
        },
        {
            "round_index": 0,
            "order_index": 0,
            "timing_ms": [2.0, 2.0, 2.0],
            "device_samples_ms": [1.0] * 5,
            "setup_wall_us": 10,
            "execution_wall_us": 20,
        },
        {
            "round_index": 1,
            "order_index": 0,
            "timing_ms": [1.0, 1.0, 1.0],
            "device_samples_ms": None,
            "setup_wall_us": None,
            "execution_wall_us": None,
        },
    ),
)
def test_v8_measurement_round_schema_rejects_inconsistent_telemetry(
    round_record,
):
    with pytest.raises(LayoutAutotuneError, match="measurement round"):
        module._validate_cached_measurement_rounds(
            [round_record],
            round_record["timing_ms"],
        )


def test_v8_selection_trace_requires_fallback_and_selected_to_be_stable():
    manifest = parse_layout_manifest(_manifest([
        (_digest("a"), f'{_digest("a")}.ttgir', _digest("1")),
        (_digest("b"), f'{_digest("b")}.ttgir', _digest("2")),
    ]))
    fallback, candidate = manifest.variants
    with pytest.raises(LayoutAutotuneError, match="selection trace"):
        module._validate_selection_trace(
            {
                "initial_winner": candidate.digest,
                "close": [fallback.digest, candidate.digest],
                "stable": [candidate.digest],
                "selected": candidate.digest,
                "fallback_forced": False,
            },
            manifest,
            candidate,
            {
                fallback: (1.0, 0.9, 1.1),
                candidate: (0.8, 0.7, 0.9),
            },
        )


@pytest.mark.parametrize(
    "metrics",
    (
        {"mode": "parent-fallback", "outcome": "success", "wall_us": True},
        {"mode": "isolated-candidate", "outcome": "failure", "isolated_wall_us": -1},
        {
            **_isolated_compile_metrics(),
            "lowering_stages_us": [["mlir", 1], ["mlir", 2]],
        },
        {**_isolated_compile_metrics(), "total_us": 11},
        {**_isolated_compile_metrics(), "unexpected": 1},
    ),
)
def test_compile_metrics_schema_rejects_ambiguous_or_invalid_records(metrics):
    with pytest.raises(LayoutAutotuneError, match="compile metrics|compile stage"):
        module._validate_compile_metrics(metrics)


def test_candidate_diagnostic_is_byte_bounded_and_content_addressed():
    message = "候选失败 " * 600
    compact = module._compact_candidate_error(message)
    assert "truncated normalized-message sha256=" in compact
    assert len(compact.encode("utf-8")) < 1200
    assert compact == module._compact_candidate_error(message)


def test_candidate_compile_timeout_terminates_and_reaps_worker(monkeypatch):
    class Connection:

        def __init__(self):
            self.closed = False
            self.timeout = None

        def poll(self, timeout):
            self.timeout = timeout
            return False

        def close(self):
            self.closed = True

    class Process:

        pid = 123
        exitcode = None

        def __init__(self):
            self.started = False
            self.alive = True
            self.terminated = False

        def start(self):
            self.started = True

        @staticmethod
        def join(timeout=None):
            assert timeout == 5

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

    parent = Connection()
    child = Connection()
    process = Process()

    class Context:

        @staticmethod
        def Pipe(duplex=False):
            assert not duplex
            return parent, child

        @staticmethod
        def Process(**kwargs):
            assert kwargs["target"] is module._isolated_compile_worker
            assert not kwargs["daemon"]
            return process

    monkeypatch.setattr(module.multiprocessing, "get_context", lambda _: Context())
    with pytest.raises(module._CandidateCompileFailure, match="compile-timeout") as error:
        module._compile_variant_isolated(object(), FakeKernel("outer", (1, 1, 1)))

    assert process.started and process.terminated and not process.alive
    assert parent.timeout == module._AUTOTUNE_CANDIDATE_COMPILE_TIMEOUT_SECONDS
    assert parent.closed and child.closed
    assert error.value.compile_metrics["mode"] == "isolated-candidate"
    assert error.value.compile_metrics["outcome"] == "failure"


def test_candidate_compile_isolation_setup_failure_has_metrics(monkeypatch):
    monkeypatch.setattr(
        module.multiprocessing,
        "get_context",
        lambda _method: (_ for _ in ()).throw(ValueError("no forkserver")),
    )
    with pytest.raises(
        module._CandidateCompileFailure,
        match="compile-isolation-unavailable",
    ) as error:
        module._compile_variant_isolated(object(), FakeKernel("outer", (1, 1, 1)))
    assert error.value.compile_metrics["outcome"] == "failure"


def test_candidate_compile_start_failure_has_metrics_and_closes_pipes(monkeypatch):
    parent = _MeasurementProtocolConnection()
    child = _MeasurementProtocolConnection()

    class Process:
        pid = None

        @staticmethod
        def start():
            raise RuntimeError("spawn failed")

    process = Process()

    class Context:

        @staticmethod
        def Pipe(duplex=False):
            assert not duplex
            return parent, child

        @staticmethod
        def Process(**_kwargs):
            return process

    monkeypatch.setattr(module.multiprocessing, "get_context", lambda _method: Context())
    with pytest.raises(module._CandidateCompileFailure, match="compile-start-error") as error:
        module._compile_variant_isolated(object(), FakeKernel("outer", (1, 1, 1)))
    assert error.value.compile_metrics["outcome"] == "failure"
    assert parent.closed and child.closed


def test_candidate_compile_reload_failure_has_metrics(monkeypatch):
    artifact = module._CompiledArtifact(
        module._KernelReloadDescriptor((), "compile-hash", (), (), (), ()),
        (),
        _isolated_compile_metrics(0),
    )
    parent = _MeasurementProtocolConnection(
        polls=[True], responses=[{"status": "ok", "artifact": artifact}]
    )
    child = _MeasurementProtocolConnection()

    class Process:
        pid = 123
        exitcode = 0

        @staticmethod
        def start():
            return None

        @staticmethod
        def join(timeout=None):
            assert timeout == 5

        @staticmethod
        def is_alive():
            return False

    class Context:

        @staticmethod
        def Pipe(duplex=False):
            assert not duplex
            return parent, child

        @staticmethod
        def Process(**_kwargs):
            return Process()

    monkeypatch.setattr(module.multiprocessing, "get_context", lambda _method: Context())
    compiler_module = importlib.import_module("triton.compiler.compiler")
    monkeypatch.setattr(
        compiler_module,
        "CompiledKernel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("reload failed")),
    )
    with pytest.raises(module._CandidateCompileFailure, match="compile-artifact-error") as error:
        module._compile_variant_isolated(object(), FakeKernel("outer", (1, 1, 1)))
    assert error.value.compile_metrics["outcome"] == "failure"
    assert parent.closed and child.closed


class _MeasurementProtocolConnection:

    def __init__(self, *, polls=(), responses=()):
        self.polls = list(polls)
        self.responses = list(responses)
        self.poll_timeouts = []
        self.sent = []
        self.closed = False

    def poll(self, timeout):
        self.poll_timeouts.append(timeout)
        return self.polls.pop(0)

    def recv(self):
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def send(self, message):
        self.sent.append(message)

    def close(self):
        self.closed = True


class _MeasurementProtocolProcess:

    pid = 456
    exitcode = 17

    def __init__(self):
        self.started = False
        self.alive = True
        self.join_timeouts = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def start(self):
        self.started = True

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1
        self.alive = False


def _install_measurement_protocol(
    monkeypatch, *, polls, responses
):
    parent = _MeasurementProtocolConnection(
        polls=polls, responses=responses
    )
    child = _MeasurementProtocolConnection()
    process = _MeasurementProtocolProcess()

    class Context:

        @staticmethod
        def Pipe(duplex=False):
            assert duplex
            return parent, child

        @staticmethod
        def Process(**kwargs):
            assert kwargs["target"] is module._isolated_measurement_worker
            assert kwargs["args"][0] is child
            assert kwargs["daemon"]
            return process

    monkeypatch.setattr(
        module.multiprocessing,
        "get_all_start_methods",
        lambda: ["forkserver"],
    )
    monkeypatch.setattr(
        module.multiprocessing,
        "get_context",
        lambda method: Context() if method == "forkserver" else None,
    )
    monkeypatch.setattr(
        module,
        "_make_kernel_reload_descriptor",
        lambda _kernel: SimpleNamespace(constant_paths=()),
    )
    driver_module = importlib.import_module("triton.runtime.driver")
    monkeypatch.setattr(
        driver_module,
        "driver",
        SimpleNamespace(
            active=SimpleNamespace(get_current_device=lambda: 7)
        ),
    )
    return parent, child, process


def _assert_isolated_measurement_cleanup(parent, child, process):
    assert process.started
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.join_timeouts == [5, 5, 5]
    assert not process.alive
    assert parent.closed and child.closed


def test_isolated_measurement_returns_raw_samples_and_phase_walls(monkeypatch):
    raw_samples = [1.0] * 5
    parent, child, process = _install_measurement_protocol(
        monkeypatch,
        polls=[True, True],
        responses=[
            {"status": "ready"},
            {
                "status": "ok",
                "timing": [1.0, 1.0, 1.0],
                "device_samples_ms": raw_samples,
                "resources": {"shared": 1024},
            },
        ],
    )
    timestamps = iter((1_000_000, 2_500_000, 3_000_000, 5_250_000))
    monkeypatch.setattr(
        module.time,
        "perf_counter_ns",
        lambda: next(timestamps),
    )

    measurement = module._measure_variant_isolated(
        object(),
        (1, 1, 1),
        module._ScratchReplay((), lambda: None, ()),
    )

    assert measurement == module._MeasurementResult(
        (1.0, 1.0, 1.0),
        {"shared": 1024},
        tuple(raw_samples),
        1500,
        2250,
    )
    assert parent.poll_timeouts == [60, 60]
    assert parent.sent == ["run"]
    _assert_isolated_measurement_cleanup(parent, child, process)


def test_isolated_measurement_setup_timeout_reaps_worker(monkeypatch):
    parent, child, process = _install_measurement_protocol(
        monkeypatch, polls=[False], responses=[]
    )
    replay = module._ScratchReplay((), lambda: None, ())

    with pytest.raises(
        module._CandidateExecutionFailure, match="setup-timeout"
    ):
        module._measure_variant_isolated(object(), (1, 1, 1), replay)

    assert parent.poll_timeouts == [60]
    assert parent.sent == []
    _assert_isolated_measurement_cleanup(parent, child, process)


@pytest.mark.parametrize(
    "polls,responses,reason",
    [
        (
            [True, False],
            [{"status": "ready"}],
            "execution-timeout",
        ),
        (
            [True, True],
            [{"status": "ready"}, EOFError()],
            "execution-crash",
        ),
        (
            [True, True],
            [{"status": "ready"}, {"timing": [1.0, 0.9, 1.1]}],
            "execution-protocol",
        ),
    ],
)
def test_isolated_measurement_post_ready_failures_reap_worker(
    monkeypatch, polls, responses, reason
):
    parent, child, process = _install_measurement_protocol(
        monkeypatch, polls=polls, responses=responses
    )
    replay = module._ScratchReplay((), lambda: None, ())

    with pytest.raises(module._CandidateExecutionFailure, match=reason):
        module._measure_variant_isolated(object(), (1, 1, 1), replay)

    assert parent.poll_timeouts == [60, 60]
    assert parent.sent == ["run"]
    _assert_isolated_measurement_cleanup(parent, child, process)


def test_candidate_compile_diagnostic_is_bounded_and_identifiable():
    message = "layout verifier failure " + "x" * 4096
    compact = module._compact_candidate_error(RuntimeError(message))
    wrapped = module._compact_candidate_error(f"mlir.RuntimeError: {compact}")

    assert len(compact.encode("utf-8")) <= 1024
    assert compact.startswith("layout verifier failure")
    assert "truncated normalized-message sha256=" in compact
    assert "\n" not in compact
    assert compact.rsplit("sha256=", 1)[1] == wrapped.rsplit("sha256=", 1)[1]
    assert len(wrapped.encode("utf-8")) <= 1024


def test_parallel_plugin_is_native_autotuner_created_only_by_factory(monkeypatch):
    factory_module, parallel_module = _load_parallel_source(monkeypatch)
    from triton.runtime.autotuner import Autotuner

    def kernel():
        pass

    controller = factory_module.create_layout_autotuner(kernel)
    assert isinstance(controller, parallel_module.ParallelCompileAutotuner)
    assert isinstance(controller, Autotuner)
    assert controller.fn is kernel
    assert controller.configs
    assert controller._compile_scheduler.batch_size == 16
    assert controller._compile_scheduler.max_workers == 16
    assert controller._compile_scheduler.ready_batches == 1


def test_candidate_domain_preserves_closed_manifest_order_and_batch_limit(
    monkeypatch,
):
    _, parallel_module = _load_parallel_source(monkeypatch)
    entries = []
    for index in range(18):
        digest = hashlib.sha256(f"digest-{index}".encode()).hexdigest()
        entries.append(
            (
                digest,
                f"{digest}.ttgir",
                hashlib.sha256(f"source-{index}".encode()).hexdigest(),
            )
        )
    manifest = parse_layout_manifest(_manifest(entries))
    domain = parallel_module.CandidateDomain.from_manifest(manifest)

    assert domain.fallback.specification == manifest.fallback
    assert [candidate.specification for candidate in domain.nonfallback] == list(
        manifest.variants[1:]
    )
    batches = domain.batches(16)
    assert [len(batch) for batch in batches] == [16, 1]
    assert [
        candidate.order_index for batch in batches for candidate in batch
    ] == list(range(1, 18))
    with pytest.raises(ValueError, match=r"\[1, 16\]"):
        parallel_module.BatchCompileScheduler(batch_size=17)


def test_batch_scheduler_starts_eagerly_and_restores_manifest_order(
    monkeypatch,
):
    _, parallel_module = _load_parallel_source(monkeypatch)
    manifest = parse_layout_manifest(_manifest())
    domain = parallel_module.CandidateDomain.from_manifest(manifest)
    all_started = threading.Event()
    lock = threading.Lock()
    started = []
    releases = {
        candidate.order_index: threading.Event()
        for candidate in domain.nonfallback
    }

    class Backend:

        def compile(self, _bundle, candidate):
            with lock:
                started.append(candidate.order_index)
                if len(started) == len(domain.nonfallback):
                    all_started.set()
            releases[candidate.order_index].wait(timeout=5)
            return SimpleNamespace(digest=candidate.specification.digest)

    scheduler = parallel_module.BatchCompileScheduler(
        batch_size=3,
        max_workers=3,
        ready_batches=1,
        backend=Backend(),
    )
    batches = scheduler.stream(object(), domain)
    assert all_started.wait(timeout=5), "stream() did not eagerly start compilation"
    for order_index in reversed(range(1, 4)):
        releases[order_index].set()
    compiled = list(batches)

    assert len(compiled) == 1
    assert [result.candidate.order_index for result in compiled[0].results] == [
        1,
        2,
        3,
    ]
    assert [result.compiled.digest for result in compiled[0].results] == [
        manifest.variants[index].digest for index in range(1, 4)
    ]


def test_batch_stream_closes_before_first_consume_without_leaking_producer(
    monkeypatch,
):
    _, parallel_module = _load_parallel_source(monkeypatch)
    manifest = parse_layout_manifest(_manifest())
    domain = parallel_module.CandidateDomain.from_manifest(manifest)

    class Backend:

        def compile(self, _bundle, candidate):
            return SimpleNamespace(digest=candidate.specification.digest)

    scheduler = parallel_module.BatchCompileScheduler(
        batch_size=1,
        max_workers=1,
        ready_batches=1,
        backend=Backend(),
    )
    batches = scheduler.stream(object(), domain)
    batches.close()

    assert not batches._producer.is_alive()
    assert batches._closed
    assert list(batches) == []


def test_parallel_consumer_is_serial_while_next_batch_compiles(monkeypatch):
    _, parallel_module = _load_parallel_source(monkeypatch)

    def native_kernel():
        pass

    entries = []
    for index in range(5):
        digest = hashlib.sha256(f"pipeline-{index}".encode()).hexdigest()
        entries.append(
            (
                digest,
                f"{digest}.ttgir",
                hashlib.sha256(f"pipeline-source-{index}".encode()).hexdigest(),
            )
        )
    manifest_raw = _manifest(entries)
    manifest = parse_layout_manifest(manifest_raw)
    kernels = [
        FakeKernel(f"pipeline-{index}", (1.0 - index * 0.1,) * 3)
        for index in range(5)
    ]
    bundle = _make_bundle(kernels, manifest=manifest_raw)
    second_batch_started = threading.Event()
    compile_threads = set()

    class CompileBackend:

        def compile(self, compile_bundle, candidate):
            compile_threads.add(threading.get_ident())
            if candidate.order_index >= 3:
                second_batch_started.set()
            return compile_bundle.variants[candidate.specification.digest]

    measurement_threads = []
    active_measurements = 0
    max_active_measurements = 0

    class MeasureBackend:

        def measure(
            self,
            _controller,
            outcomes,
            _manifest,
            candidate,
            compiled,
            _grid,
            _stream,
            _replay,
        ):
            nonlocal active_measurements, max_active_measurements
            if candidate.order_index == 1:
                assert second_batch_started.wait(timeout=5)
            active_measurements += 1
            max_active_measurements = max(
                max_active_measurements, active_measurements
            )
            measurement_threads.append(threading.get_ident())
            outcomes.timings[candidate.specification] = compiled.kernel.timing
            outcomes.resources[candidate.specification] = {}
            active_measurements -= 1

    controller = parallel_module.ParallelCompileAutotuner(
        native_kernel,
        compile_batch_size=2,
        compile_workers=2,
        compile_backend=CompileBackend(),
        measurement_backend=MeasureBackend(),
    )
    replay = module._ScratchReplay((), lambda: None, ())
    outcomes = controller._evaluate_domain(
        bundle, manifest, (1, 1, 1), None, replay
    )

    assert list(outcomes.timings) == list(manifest.variants)
    assert second_batch_started.is_set()
    assert max_active_measurements == 1
    assert set(measurement_threads) == {threading.get_ident()}
    assert compile_threads.isdisjoint(measurement_threads)


def test_bundle_compile_lock_is_per_digest_and_deduplicates_same_digest():
    manifest = parse_layout_manifest(_manifest())
    fallback = GluonLayoutCompiledVariant(
        manifest.fallback.digest,
        manifest.fallback.mlir_file,
        manifest.fallback.mlir_sha256,
        (),
        object(),
    )
    entered = threading.Barrier(2)
    compile_counts = {}
    count_lock = threading.Lock()

    def compile_variant(specification):
        with count_lock:
            compile_counts[specification.digest] = (
                compile_counts.get(specification.digest, 0) + 1
            )
        entered.wait(timeout=5)
        return GluonLayoutCompiledVariant(
            specification.digest,
            specification.mlir_file,
            specification.mlir_sha256,
            (),
            object(),
        )

    bundle = GluonLayoutVariantBundle(
        _manifest(),
        {manifest.fallback.digest: fallback},
        _contract(),
        compile_variant=compile_variant,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        different = [
            executor.submit(bundle.get_or_compile, specification)
            for specification in manifest.variants[1:3]
        ]
        assert all(future.result() is not None for future in different)
    assert compile_counts == {
        manifest.variants[1].digest: 1,
        manifest.variants[2].digest: 1,
    }

    duplicate_started = threading.Event()
    duplicate_release = threading.Event()
    duplicate_count = 0

    def compile_duplicate(specification):
        nonlocal duplicate_count
        duplicate_count += 1
        duplicate_started.set()
        duplicate_release.wait(timeout=5)
        return GluonLayoutCompiledVariant(
            specification.digest,
            specification.mlir_file,
            specification.mlir_sha256,
            (),
            object(),
        )

    duplicate_bundle = GluonLayoutVariantBundle(
        _manifest(),
        {manifest.fallback.digest: fallback},
        _contract(),
        compile_variant=compile_duplicate,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            duplicate_bundle.get_or_compile, manifest.variants[1]
        )
        assert duplicate_started.wait(timeout=5)
        second = executor.submit(
            duplicate_bundle.get_or_compile, manifest.variants[1]
        )
        duplicate_release.set()
        assert first.result() is second.result()
    assert duplicate_count == 1


def test_parallel_persistent_hit_loads_winner_without_starting_scheduler(
    tmp_path, monkeypatch,
):
    _, parallel_module = _load_parallel_source(monkeypatch)

    def native_kernel():
        pass

    kernels = _kernels()
    bundle = _make_bundle(kernels)
    _, cache_factory = _cache_factory(tmp_path)
    args = {"out": FakeTensor(0x1000), "M": 128}
    measured = []
    first = parallel_module.ParallelCompileAutotuner(
        native_kernel,
        cache_manager_factory=cache_factory,
        measurer=lambda kernel, *_: measured.append(kernel.name)
        or kernel.timing,
    )
    assert first.prepare_for_launch(bundle, (1,), None, args) is kernels[1]
    assert measured == ["fallback", "winner", "other", "slow"]

    replay = parallel_module.ParallelCompileAutotuner(
        native_kernel,
        cache_manager_factory=cache_factory,
        measurer=lambda *_: (_ for _ in ()).throw(
            AssertionError("persistent hit measured a candidate")
        ),
    )
    monkeypatch.setattr(
        replay._compile_scheduler,
        "stream",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("persistent hit started batch compilation")
        ),
    )
    assert replay.prepare_for_launch(bundle, (1,), None, args) is kernels[1]
    assert replay._measurement_backend._session is None


def test_persistent_measurement_backend_reuses_one_worker(monkeypatch):
    _, parallel_module = _load_parallel_source(monkeypatch)
    raw_samples = [1.0] * 5
    parent = _MeasurementProtocolConnection(
        polls=[True] * 5,
        responses=[
            {"status": "session-ready"},
            {"status": "candidate-ready"},
            {
                "status": "ok",
                "timing": [1.0, 1.0, 1.0],
                "device_samples_ms": raw_samples,
                "resources": {"shared": 1024},
            },
            {"status": "candidate-ready"},
            {
                "status": "ok",
                "timing": [1.0, 1.0, 1.0],
                "device_samples_ms": raw_samples,
                "resources": {"shared": 2048},
            },
        ],
    )
    child = _MeasurementProtocolConnection()
    process = _MeasurementProtocolProcess()
    process_creations = []

    class Context:

        @staticmethod
        def Pipe(duplex=False):
            assert duplex
            return parent, child

        @staticmethod
        def Process(**kwargs):
            assert kwargs["target"] is (
                parallel_module._persistent_measurement_worker
            )
            assert kwargs["args"] == (child, 7)
            assert kwargs["daemon"]
            process_creations.append(kwargs)
            return process

    monkeypatch.setattr(
        parallel_module.multiprocessing,
        "get_all_start_methods",
        lambda: ["forkserver"],
    )
    monkeypatch.setattr(
        parallel_module.multiprocessing,
        "get_context",
        lambda method: Context() if method == "forkserver" else None,
    )
    monkeypatch.setattr(
        parallel_module,
        "_make_kernel_reload_descriptor",
        lambda kernel: module._KernelReloadDescriptor(
            (), f"hash-{id(kernel)}", (), (), (), ()
        ),
    )
    driver_module = importlib.import_module("triton.runtime.driver")
    monkeypatch.setattr(
        driver_module,
        "driver",
        SimpleNamespace(
            active=SimpleNamespace(get_current_device=lambda: 7)
        ),
    )
    timestamps = iter(
        (
            1_000_000,
            2_000_000,
            3_000_000,
            4_000_000,
            5_000_000,
            5_100_000,
            6_000_000,
            6_200_000,
        )
    )
    monkeypatch.setattr(
        parallel_module.time,
        "perf_counter_ns",
        lambda: next(timestamps),
    )

    backend = parallel_module.MeasurementBackend()
    replay = module._ScratchReplay((), lambda: None, ())
    with backend.scope():
        first = backend.measure_kernel(object(), (1, 1, 1), replay)
        second = backend.measure_kernel(object(), (1, 1, 1), replay)

    assert len(process_creations) == 1
    assert first.setup_wall_us == 1000
    assert first.execution_wall_us == 1000
    assert second.setup_wall_us == 100
    assert second.execution_wall_us == 200
    assert first.resources == {"shared": 1024}
    assert second.resources == {"shared": 2048}
    assert [
        message if isinstance(message, str) else message["command"]
        for message in parent.sent
    ] == ["prepare", "run", "prepare", "run", "stop"]
    assert backend._session is None
