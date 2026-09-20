"""Exercise promised SCF/Tensor interfaces through the actual CoreIR pipeline."""
import os
from pathlib import Path
import subprocess

import pytest

from triton.backends.dicp_triton.wafer import _find_wafer_opt


@pytest.mark.parametrize("name", ["interfaces-argmax2d", "interfaces-flip", "interfaces-sort",
                                  "pointer-state-nested_loops", "pointer-state-modulo",
                                  "pointer-state-scalar_store", "pointer-state-tensor_index_iterargs"])
def test_external_interfaces(name, tmp_path):
    source = Path(__file__).with_name("ir") / f"{name}.mlir"
    output = tmp_path / "coreir.mlir"
    compiler = os.getenv("WAFER_TEST_OPT") or str(_find_wafer_opt())
    result = subprocess.run([
        compiler, str(source), "--triton-to-core-dialects", "--tle-to-mk",
        "--dsa-memory-to-core", "--linalg-tiling", "--core-dialects-to-mk",
        "--linalg-fusion", "--legalize-tensor-form-loops", "--one-shot-bufferize",
        "--convert-bufferization-to-memref", "--cse", "--canonicalize",
        "-o", str(output),
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert output.stat().st_size > 0
