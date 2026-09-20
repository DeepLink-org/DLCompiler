"""Precision policy must affect both compilation options and cache identity."""
import pytest
from triton.backends.compiler import GPUTarget


def test_precision_mode_snapshot_and_override(monkeypatch, wafer_modules, fake_toolchain):
    _, compiler, _ = wafer_modules
    monkeypatch.setenv("WAFER_ENABLE_RUNTIME", "0")
    monkeypatch.delenv("PRECISION_MODE", raising=False)
    monkeypatch.setenv("PRECISION_PRIORITY", "1")
    legacy = compiler.WaferBackend(GPUTarget("wafer", "wafer", 32))
    assert legacy.parse_options({}).precision_mode == 2
    before = legacy.hash()
    monkeypatch.setenv("PRECISION_MODE", "1")
    assert legacy.hash() == before
    modern = compiler.WaferBackend(GPUTarget("wafer", "wafer", 32))
    assert modern.parse_options({}).precision_mode == 1
    assert modern.hash() != before
    assert modern.parse_options({"precision_mode": 0}).precision_mode == 0
    assert modern.parse_options({"precision_mode": 2}).hash() != modern.parse_options({}).hash()
    monkeypatch.setenv("PRECISION_MODE", "invalid")
    with pytest.raises(ValueError, match="PRECISION_MODE"):
        compiler.WaferBackend(GPUTarget("wafer", "wafer", 32))


def test_strided_output_copyback(tmp_path):
    import os
    import re
    import subprocess
    from triton.backends.dicp_triton.wafer import _find_wafer_opt
    source = tmp_path / "stride.mlir"
    source.write_text('''module {
      func.func @slice(%input: memref<4xf32>, %output: memref<8xf32>) {
        %one = arith.constant 1.0 : f32
        %view = memref.subview %output[0] [4] [2] : memref<8xf32> to memref<4xf32, strided<[2]>>
        "mk.addvs"(%input, %one, %view) : (memref<4xf32>, f32, memref<4xf32, strided<[2]>>) -> ()
        return
      }
    }''')
    result = subprocess.run([os.getenv("WAFER_TEST_OPT") or _find_wafer_opt(), str(source),
                             "--materialize-strided-linalg-inputs"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    copies = re.findall(r"memref.copy (%[\w]+), (%[\w]+)", result.stdout)
    assert len(copies) == 2, result.stdout
    assert copies[0] == copies[1][::-1], result.stdout
    assert copies[0][0] != copies[0][1], result.stdout


def test_pipeline_option_cache_isolation(monkeypatch, wafer_modules, fake_toolchain):
    _, compiler, _ = wafer_modules
    monkeypatch.setenv("WAFER_ENABLE_RUNTIME", "0")
    monkeypatch.setenv("TRITON_PIPELINE", "0")
    plain = compiler.WaferBackend(GPUTarget("wafer", "wafer", 32))
    before = plain.hash()
    monkeypatch.setenv("TRITON_PIPELINE", "1")
    pipelined = compiler.WaferBackend(GPUTarget("wafer", "wafer", 32))
    assert not plain.parse_options({}).enable_pipeline
    assert plain.hash() == before
    assert pipelined.parse_options({}).enable_pipeline
    assert pipelined.hash() != before
    override = pipelined.parse_options({"enable_pipeline": False})
    assert not override.enable_pipeline
    assert override.hash() != pipelined.parse_options({}).hash()
