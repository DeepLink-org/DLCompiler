"""Run against the isolated wheel; no vendor runtime or device is required."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ir, wafer
from triton.backends.compiler import GPUTarget
from triton.backends.dicp_triton.wafer import WaferBackend, _find_wafer_opt
from triton.compiler import ASTSource


pytestmark = pytest.mark.skipif(
    getattr(wafer, "build_role", None) != "frontend",
    reason="requires the Wafer-only frontend wheel",
)


@triton.jit
def loop_kernel(out):
    i = tl.arange(0, 16)
    x = i.to(tl.float32)
    for _ in range(2):
        x = x + 1.0
    tl.store(out + i, x)


@triton.jit
def non_power_of_two(out):
    i = tl.arange(0, 3)
    tl.store(out + i, i.to(tl.float32))


@triton.jit
def scalar_copy_kernel(src, out):
    tl.store(out, tl.load(src))


@triton.jit
def member_slice_kernel(out):
    x = tl.full((16,), 1, tl.float32)
    sub = x.extract_slice(offsets=(0,), sizes=(8,), strides=(1,))
    y = x.insert_slice(sub + 1, offsets=(8,))
    tl.store(out + tl.arange(0, 16), y)


@triton.jit
def bounded_slice_kernel(out):
    x = tl.arange(0, 16).to(tl.float32)
    left = x[:8]
    right = x[8:]
    tl.store(out + tl.arange(0, 8)[:, None], (left + right)[:, None])


def make_module(fn, signature=None, constexprs=None):
    backend = WaferBackend(GPUTarget("wafer", "wafer", 32))
    options = backend.parse_options({})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    codegen = backend.get_codegen_implementation(options)
    return ASTSource(fn, signature=signature or {"out": "*fp32"}, constexprs=constexprs).make_ir(
        backend.target, options, codegen, {}, context
    )


def test_package_has_no_original_dicp_or_cann():
    from triton._C import libtriton

    assert not hasattr(libtriton, "dicp_triton")
    root = Path(triton.__file__).parent
    assert not (root / "language/extra/deeplink").exists()
    assert not (root / "backends/dicp_triton/dicp_opt").exists()
    text = str(make_module(scalar_copy_kernel, {"src": "*fp32", "out": "*fp32"}))
    assert "dicp.disable_addptr_fold" not in text
    assert "tt.load" in text and "tt.store" in text


@pytest.mark.parametrize("target", ["wafer", "wafer-cache-before-tle", "wafer-cache-after-tle", "wafer-slices"])
def test_codegen_in_separate_processes(target):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), target],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cache_tracks_vendor_sources_without_importing_them(tmp_path, monkeypatch):
    from triton.runtime import cache
    import sysconfig

    root = tmp_path / "triton"
    # A package initializer must never run just to compute a cache key. Both
    # package and nested source edits must nevertheless invalidate that key.
    source = root / "language/extra/vendor_probe/ops.py"
    files = {
        "runtime/cache.py": "# cache input\n",
        "_C/libtriton." + sysconfig.get_config_var("EXT_SUFFIX").split(".")[-1]: "binary input",
        "language/extra/__init__.py": "raise RuntimeError('unexpected import')\n",
        "language/extra/vendor_probe/__init__.py": "raise RuntimeError('unexpected import')\n",
        "language/extra/vendor_probe/ops.py": "VALUE = 1\n",
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    monkeypatch.setattr(cache, "__file__", str(root / "runtime/cache.py"))
    initial = cache.triton_key.__wrapped__()
    source.write_text("VALUE = 2\n")
    changed_source = cache.triton_key.__wrapped__()
    source.with_name("__init__.py").write_text("raise RuntimeError('still must not run')\n")
    assert len({initial, changed_source, cache.triton_key.__wrapped__()}) == 3


def test_frontend_text_can_enter_wafer_lowering(tmp_path):
    source = tmp_path / "frontend.mlir"
    source.write_text(str(make_module(loop_kernel)))
    result = subprocess.run(
        [os.getenv("WAFER_TEST_OPT") or str(_find_wafer_opt()), str(source),
         "--triton-to-core-dialects", "-o", str(tmp_path / "core.mlir")],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "linalg" in (tmp_path / "core.mlir").read_text()


def test_tle_text_can_enter_separate_tools(tmp_path):
    from test_tle_frontend import local_kernel

    source = tmp_path / "dsa.mlir"
    source.write_text(str(make_module(local_kernel)))
    assert "dsa.alloc" in source.read_text()
    result = subprocess.run(
        [os.getenv("WAFER_TEST_OPT") or str(_find_wafer_opt()), str(source),
         "--triton-to-core-dialects", "--tle-to-mk", "--dsa-memory-to-core",
         "-o", str(tmp_path / "core.mlir")], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "dsa.alloc" not in (tmp_path / "core.mlir").read_text()


if __name__ == "__main__":
    assert wafer.build_role == "frontend"
    if sys.argv[1] == "wafer-slices":
        import triton.language.extra.wafer.slicing

        text = str(make_module(bounded_slice_kernel))
        assert text.count('"dsa.extract_slice"') == 2
        assert "tt.expand_dims" in text
        assert not any(".deeplink.cann" in name for name in sys.modules)
        sys.exit(0)
    if sys.argv[1].startswith("wafer-cache-"):
        from triton.runtime.cache import triton_key

        if sys.argv[1] == "wafer-cache-before-tle":
            triton_key()
        import triton.experimental.tle.language  # registers Wafer tensor members
        members = (tl.tensor.extract_slice, tl.tensor.insert_slice, tl.tensor.__getitem__)
        original_tanh = getattr(tl.math, "tanh", None)
        triton_key()
        assert not any(".deeplink.cann" in name for name in sys.modules)
        assert members == (tl.tensor.extract_slice, tl.tensor.insert_slice, tl.tensor.__getitem__)
        assert getattr(tl.math, "tanh", None) is original_tanh
        text = str(make_module(member_slice_kernel))
        assert '"dsa.extract_slice"' in text and '"dsa.insert_slice"' in text
        sys.exit(0)
    from triton.runtime.cache import triton_key
    triton_key()
    original_tanh = getattr(tl.math, "tanh", None)
    module = str(make_module(loop_kernel))
    assert "scf.for" in module and "tt.store" in module
    assert "dicp.disable_addptr_fold" not in module
    assert not any(".deeplink.cann" in name for name in sys.modules)
    assert getattr(tl.math, "tanh", None) is original_tanh
    with pytest.raises(triton.compiler.errors.CompilationError, match="power of 2"):
        make_module(non_power_of_two)
