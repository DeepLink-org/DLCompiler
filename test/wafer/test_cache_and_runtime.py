import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from triton.backends.compiler import GPUTarget


TARGET = GPUTarget("wafer", "wafer", 32)


def test_modes_are_snapshotted_and_separate_cache_keys(
    monkeypatch, wafer_modules, fake_toolchain
):
    _, compiler, _ = wafer_modules
    monkeypatch.setenv("USE_SIM_MODE", "0")
    monkeypatch.setenv("WAFER_ENABLE_RUNTIME", "0")
    offline = compiler.WaferBackend(TARGET)
    monkeypatch.setenv("WAFER_ENABLE_RUNTIME", "1")
    hardware = compiler.WaferBackend(TARGET)
    monkeypatch.setenv("USE_SIM_MODE", "1")
    simulator = compiler.WaferBackend(TARGET)
    monkeypatch.setenv("WAFER_ENABLE_RUNTIME", "0")
    for backend, final in ((offline, "o"), (hardware, "so"), (simulator, "so")):
        stages = {}
        backend.add_stages(stages, backend.parse_options({}))
        assert backend.binary_ext == final
        assert list(stages)[-1] == final
    assert len({backend.hash() for backend in (offline, hardware, simulator)}) == 3


def test_archive_update_invalidates_both_compiler_and_link_cache(
    monkeypatch, wafer_modules, fake_toolchain, tmp_path
):
    _, compiler, _ = wafer_modules
    _, libraries, _ = fake_toolchain
    linker, _ = compiler._runtime_link_inputs()
    monkeypatch.setenv("USE_SIM_MODE", "0")
    monkeypatch.setenv("WAFER_ENABLE_RUNTIME", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("WAFER_DEVICE_LOG_ABI", "wafer")
    commands = []

    def link(command):
        from pathlib import Path

        commands.append(command)
        Path(command[-1]).write_bytes(b"linked " + libraries[3].read_bytes())

    monkeypatch.setattr(compiler, "_run_tool", link)
    before = compiler.WaferBackend(TARGET).hash()
    metadata = {}
    first = compiler.object_to_binary(b"same object", metadata)
    path_before = metadata["kernel_path"]
    assert compiler.object_to_binary(b"same object", {}) == first
    assert sum(str(command[0]) == str(linker) for command in commands) == 1
    # Even a same-size edit with the old mtime restored must invalidate.
    old_stat = libraries[3].stat()
    libraries[3].write_bytes(b"X" * old_stat.st_size)
    os.utime(libraries[3], ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    second = compiler.object_to_binary(b"same object", metadata)
    assert compiler.WaferBackend(TARGET).hash() != before
    assert second != first and metadata["kernel_path"] != path_before
    assert sum(str(command[0]) == str(linker) for command in commands) == 2


def test_launcher_sdk_and_compiler_changes_invalidate_cache(
    wafer_modules, fake_toolchain
):
    _, _, runtime = wafer_modules
    tools, _, sdk = fake_toolchain
    keys = [runtime._launcher_cache_key("same source")]
    for path in (
        sdk / "include/tx_runtime.h",
        sdk / "lib/libhpgr.so",
        tools["clang++"],
    ):
        path.write_bytes(path.read_bytes() + b" changed")
        keys.append(runtime._launcher_cache_key("same source"))
    assert len(set(keys)) == 4


def test_device_log_abi_invalidates_cache(monkeypatch, wafer_modules, fake_toolchain):
    _, compiler, _ = wafer_modules
    monkeypatch.setenv("USE_SIM_MODE", "0")
    monkeypatch.setenv("WAFER_ENABLE_RUNTIME", "1")
    monkeypatch.setenv("WAFER_DEVICE_LOG_ABI", "wafer")
    before = compiler.WaferBackend(TARGET).hash()
    monkeypatch.setenv("WAFER_DEVICE_LOG_ABI", "rcs")
    assert compiler.WaferBackend(TARGET).hash() != before


def test_repeated_compiled_kernel_launch_initializes_once(monkeypatch, wafer_modules):
    from triton.compiler import compiler

    _, _, runtime = wafer_modules
    launch = Mock()
    launcher_cls = Mock(return_value=launch)
    driver = SimpleNamespace(
        utils=runtime.WaferUtils(),
        launcher_cls=launcher_cls,
        get_current_device=lambda: 0,
        get_current_stream=lambda device: None,
        get_current_target=lambda: TARGET,
    )
    monkeypatch.setattr(compiler, "driver", SimpleNamespace(active=driver))
    monkeypatch.setattr(compiler, "max_shared_mem", lambda device: 4096)
    kernel = compiler.CompiledKernel.__new__(compiler.CompiledKernel)
    kernel.module, kernel.function, kernel._run = None, None, None
    kernel.src = object()
    kernel.name, kernel.kernel = "test_kernel", b"ELF lifetime token"
    kernel.metadata = SimpleNamespace(shared=0, num_warps=1)
    kernel.packed_metadata = ()
    kernel.metadata_group, kernel.hash = {}, "unit-test"
    kernel[(1, 1, 1)](123)
    kernel[(1, 1, 1)](456)
    kernel.launch_metadata((1, 1, 1), None)
    assert kernel.module is kernel.kernel
    assert launcher_cls.call_count == 1
    assert launch.call_count == 2


def test_jit_and_compiled_kernel_argument_contracts(monkeypatch, wafer_modules):
    _, _, runtime = wafer_modules
    launch = Mock()
    monkeypatch.setattr(
        runtime, "compile_launcher", lambda source: SimpleNamespace(launch=launch)
    )
    src = SimpleNamespace(
        fn=SimpleNamespace(arg_names=["pointer", "alpha", "BLOCK"]),
        signature={"BLOCK": "constexpr", "alpha": "fp32", "pointer": "*fp32"},
        constants={(2,): 256},
    )
    metadata = object()
    launcher = runtime.WaferLauncher(src, metadata)
    prefix = (1, 1, 1, None, 0, (), None, None, None)
    launcher(*prefix, 123, 1.25, 256)
    launcher(*prefix, 123, 1.25)
    expected = (*prefix[:5], metadata, *prefix[6:], 123, 1.25)
    assert launch.call_args_list[0].args == expected
    assert launch.call_args_list[1].args == expected
    with pytest.raises(TypeError, match="expected 2 runtime arguments"):
        launcher(*prefix, 123)


def test_linker_library_lookup(monkeypatch, wafer_modules, tmp_path):
    _, compiler, _ = wafer_modules
    library = tmp_path / "libc.a"
    library.write_bytes(b"archive")
    query = Mock(return_value=str(library) + "\n")
    monkeypatch.setattr(compiler.subprocess, "check_output", query)
    assert compiler._find_linker_library("gcc", "libc.a") == library
    for output in ("libc.a", str(tmp_path / "missing.a")):
        query.return_value = output
        with pytest.raises(RuntimeError, match="could not locate"):
            compiler._find_linker_library("gcc", "libc.a")
    query.side_effect = subprocess.CalledProcessError(1, "gcc")
    with pytest.raises(subprocess.CalledProcessError):
        compiler._find_linker_library("gcc", "libc.a")


def test_half_scalars_are_not_silently_packed_as_fp32(wafer_modules):
    _, _, runtime = wafer_modules
    for scalar in ("fp16", "bf16"):
        with pytest.raises(NotImplementedError, match="pass an fp32 scalar"):
            runtime.make_launcher({0: scalar})
        assert "get_pointer" in runtime.make_launcher({0: "*" + scalar})


def test_runtime_selection_does_not_hide_native_abi_errors(monkeypatch, wafer_modules):
    import builtins

    _, _, runtime = wafer_modules
    mock_wafer_runtime = object()
    fallback = object()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(txda=mock_wafer_runtime))
    monkeypatch.setitem(sys.modules, "torch_txda", SimpleNamespace())
    monkeypatch.setattr(runtime, "_KuiperRuntime", lambda: fallback)
    assert runtime.get_runtime() is mock_wafer_runtime
    monkeypatch.setitem(sys.modules, "torch_txda", None)
    assert runtime.get_runtime() is fallback
    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "torch_txda":
            raise OSError("undefined symbol: required_runtime_api")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    with pytest.raises(OSError, match="required_runtime_api"):
        runtime.get_runtime()


def test_rcs_adaptation_uses_cached_private_copies(
    monkeypatch, wafer_modules, fake_toolchain, tmp_path
):
    _, compiler, _ = wafer_modules
    linker, libraries = compiler._runtime_link_inputs()
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    originals = [path.read_bytes() for path in libraries]
    commands = []

    def adapt(command):
        from pathlib import Path

        commands.append(command)
        Path(command[-1]).write_bytes(Path(command[-2]).read_bytes() + b" adapted")

    monkeypatch.setattr(compiler, "_run_tool", adapt)
    fingerprint = compiler._link_fingerprint(linker, libraries, "rcs")
    first = compiler._adapt_logging_libraries(libraries, fingerprint)
    second = compiler._adapt_logging_libraries(libraries, fingerprint)
    assert first == second and len(commands) == 4
    assert [path.read_bytes() for path in libraries] == originals
    assert all(a != b for a, b in zip(first[:4], libraries[:4]))
    assert first[4:] == libraries[4:]
    assert all(
        "--redefine-sym=tx8_kernel_printf=rcs_kernel_printf" in command
        for command in commands
    )
