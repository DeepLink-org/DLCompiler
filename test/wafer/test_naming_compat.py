"""Compatibility behavior at the Wafer naming migration boundaries."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_legacy_imports_keep_one_discoverable_backend(wafer_modules):
    from triton.backends import _find_concrete_subclasses
    from triton.backends.compiler import BaseBackend

    _, compiler, runtime = wafer_modules
    assert compiler.TXDABackend is compiler.WaferBackend
    assert compiler.TXDAOptions is compiler.WaferOptions
    assert runtime.TXDALauncher is runtime.WaferLauncher
    assert runtime.TXDAUtils is runtime.WaferUtils
    assert _find_concrete_subclasses(compiler, BaseBackend) is compiler.WaferBackend


def test_runtime_dependencies_require_wafer_configuration(
    tmp_path, monkeypatch, wafer_modules
):
    _, compiler, _ = wafer_modules
    root = tmp_path / "sdk"
    (root / "lib").mkdir(parents=True)
    for name in ("libcommon_util.a", "libinstr_tx81.a", "liblibc_stub.a"):
        (root / "lib" / name).write_bytes(b"sdk")
    toolchain = tmp_path / "toolchain"
    (toolchain / "bin").mkdir(parents=True)
    (toolchain / "bin/riscv64-unknown-elf-gcc").touch()
    archive_dir = tmp_path / "crt"
    archive_dir.mkdir()
    (archive_dir / "libvr.a").touch()
    for name in ("libm.a", "libc.a", "libgcc.a", "libgloss.a"):
        (archive_dir / name).touch()
    monkeypatch.setenv("XUANTIE_NAME", str(toolchain))
    monkeypatch.setenv("WAFER_RUNTIME_LIB_DIR", str(archive_dir))
    monkeypatch.delenv("WAFER_DEPS_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="WAFER_DEPS_ROOT is not set"):
        compiler._runtime_link_inputs()
    monkeypatch.setenv("WAFER_DEPS_ROOT", str(root))
    monkeypatch.setattr(compiler, "_find_linker_library", lambda linker, name: archive_dir / name)
    _, libraries = compiler._runtime_link_inputs()
    assert all(path.parent == root / "lib" for path in libraries[:3])


@pytest.mark.parametrize("mode", ["environment", "canonical", "both"])
@pytest.mark.parametrize("setting", ["WAFER_DEPS_ROOT", "WAFER_BSP_INCLUDE_DIR"])
def test_cmake_configuration_precedence(tmp_path, mode, setting):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required to verify configuration compatibility")
    repo = Path(__file__).resolve().parents[2]
    script = tmp_path / "config.cmake"
    script.write_text(
        f'include("{repo / "third_party/wafer/cmake/WaferConfig.cmake"}")\n'
        f'if(NOT {setting} STREQUAL EXPECT)\n'
        '  message(FATAL_ERROR "configuration precedence mismatch")\n'
        'endif()\n'
    )
    arguments = []
    if mode != "environment":
        arguments.append(f"-D{setting}=/canonical")
    expected = "/environment" if mode == "environment" else "/canonical"
    environment = os.environ.copy()
    environment.pop(setting, None)
    if mode != "canonical":
        environment[setting] = "/environment"
    subprocess.run(
        [cmake, *arguments, f"-DEXPECT={expected}", "-P", str(script)],
        check=True, capture_output=True, text=True, env=environment,
    )
