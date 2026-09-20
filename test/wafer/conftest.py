import importlib
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture
def wafer_modules(monkeypatch):
    package = ModuleType("_wafer_under_test")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "backend")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    names = ("wafer_cache", "wafer", "wafer_runtime")
    modules = []
    for name in names:
        full_name = package.__name__ + "." + name
        monkeypatch.delitem(sys.modules, full_name, raising=False)
        modules.append(importlib.import_module(full_name))
    yield modules
    for name in names:
        sys.modules.pop(package.__name__ + "." + name, None)


@pytest.fixture
def fake_toolchain(tmp_path, monkeypatch, wafer_modules):
    _, compiler, runtime = wafer_modules
    tools = {}
    for name in (
        "wafer-opt",
        "mlir-translate",
        "clang++",
        "llvm-objcopy",
        "riscv64-unknown-elf-gcc",
        "riscv64-unknown-elf-ld",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        tools[name] = path
    libraries = []
    for name in (
        "libcommon_util.a",
        "libinstr_tx81.a",
        "liblibc_stub.a",
        "libvr.a",
        "libm.a",
        "libc.a",
        "libgcc.a",
        "libgloss.a",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        libraries.append(path)
    monkeypatch.setattr(compiler, "_find_wafer_opt", lambda: tools["wafer-opt"])
    monkeypatch.setattr(compiler, "_find_llvm_tool", lambda name: tools[name])
    monkeypatch.setattr(
        compiler,
        "_runtime_link_inputs",
        lambda: (tools["riscv64-unknown-elf-gcc"], libraries),
    )
    monkeypatch.setattr(runtime, "_launcher_compiler", lambda: str(tools["clang++"]))
    sdk = tmp_path / "kuiper"
    (sdk / "include").mkdir(parents=True)
    (sdk / "lib").mkdir()
    (sdk / "include/tx_runtime.h").write_text("sdk v1")
    (sdk / "lib/libhpgr.so").write_bytes(b"runtime v1")
    monkeypatch.setenv("KUIPER_ROOT", str(sdk))
    return tools, libraries, sdk
