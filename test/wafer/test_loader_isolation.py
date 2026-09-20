"""Exercise upstream Triton's loader call without loading a device program."""

from types import SimpleNamespace

import pytest
from triton.compiler import compiler


def make_kernel(monkeypatch, load):
    launcher = object()
    driver = SimpleNamespace(
        utils=SimpleNamespace(load_binary=load),
        get_current_device=lambda: 3,
        get_current_target=lambda: SimpleNamespace(warp_size=32),
        launcher_cls=lambda src, metadata: launcher,
    )
    monkeypatch.setattr(compiler, "driver", SimpleNamespace(active=driver))
    monkeypatch.setattr(compiler, "max_shared_mem", lambda device: 1024)
    kernel = object.__new__(compiler.CompiledKernel)
    kernel.module = None
    kernel.function = None
    kernel.src = object()
    kernel.name = "wafer_entry"
    kernel.kernel = b"ELF"
    kernel.metadata = SimpleNamespace(shared=64, num_warps=1)
    kernel.metadata_group = {}
    kernel.hash = "test-hash"
    return kernel, launcher


def test_upstream_triton_uses_native_five_result_loader(monkeypatch):
    calls = []
    native = (object(), object(), 7, 2, 1024)

    def load(*args):
        calls.append(args)
        return native

    kernel, launcher = make_kernel(monkeypatch, load)
    kernel._init_handles()
    assert calls == [("wafer_entry", b"ELF", 64, 3)]
    assert (kernel.module, kernel.function, kernel.n_regs, kernel.n_spills, kernel.n_max_threads) == native
    assert kernel._run is launcher
    kernel._init_handles()
    assert len(calls) == 1


def test_loader_error_is_not_retried_with_another_signature(monkeypatch):
    calls = []

    def load(*args):
        calls.append(args)
        raise TypeError("invalid binary in vendor loader")

    kernel, _ = make_kernel(monkeypatch, load)
    with pytest.raises(TypeError, match="invalid binary"):
        kernel._init_handles()
    assert len(calls) == 1
