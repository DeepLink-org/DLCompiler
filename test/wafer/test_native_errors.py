"""Exercise generated C++ validation without submitting work to a device."""

import os
from types import SimpleNamespace

import pytest


@pytest.fixture
def native_launcher(monkeypatch, tmp_path, wafer_modules):
    if not os.environ.get("KUIPER_ROOT"):
        pytest.skip("Native launcher validation needs Kuiper headers and libhpgr")
    _, _, runtime = wafer_modules
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    return runtime.compile_launcher(
        runtime.make_launcher({0: "*fp32", 1: "i32"})
    ).launch


def test_native_invalid_arguments(native_launcher, tmp_path):
    empty = tmp_path / "empty.so"
    empty.touch()

    def call(metadata, pointer=123, stream=None, grid=(1, 1, 1), enter=None):
        return native_launcher(
            *grid, stream, 0, metadata, None, enter, None, pointer, 256
        )

    with pytest.raises(AttributeError, match="kernel_path"):
        call(SimpleNamespace())
    with pytest.raises(TypeError):
        call(SimpleNamespace(kernel_path=123, name="test"))
    with pytest.raises(OSError, match="empty"):
        call(SimpleNamespace(kernel_path=str(empty), name="test"))
    with pytest.raises(FileNotFoundError):
        call(SimpleNamespace(kernel_path=str(tmp_path / "missing.so"), name="test"))
    with pytest.raises(TypeError):
        call(SimpleNamespace(), stream=object())
    with pytest.raises(AttributeError, match="data_ptr"):
        call(SimpleNamespace(), pointer=object())
    with pytest.raises(ValueError, match="nonnegative"):
        call(SimpleNamespace(), grid=(-1, 1, 1))
    assert call(SimpleNamespace(), grid=(0, 1, 1)) is None

    def failing_hook(metadata):
        raise ValueError("hook failed")

    with pytest.raises(ValueError, match="hook failed"):
        call(SimpleNamespace(), enter=failing_hook)
