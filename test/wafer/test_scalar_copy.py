"""Scalar copies used by debug reductions must survive the real SPM lowering."""
import os
import re
import subprocess

import pytest


@pytest.mark.parametrize("dtype", ["i1", "i8", "i32", "f32"])
def test_scalar_spm_copy(dtype, tmp_path):
    from triton.backends.dicp_triton.wafer import _find_wafer_opt

    source = tmp_path / "scalar-copy.mlir"
    source.write_text(f"""module {{
      func.func @copy(%value: {dtype}) -> {dtype} {{
        %src = memref.alloc() : memref<{dtype}>
        %dst = memref.alloc() : memref<{dtype}>
        memref.store %value, %src[] : memref<{dtype}>
        memref.copy %src, %dst : memref<{dtype}> to memref<{dtype}>
        %result = memref.load %dst[] : memref<{dtype}>
        return %result : {dtype}
      }}
    }}""")
    result = subprocess.run([
        os.getenv("WAFER_TEST_OPT") or str(_find_wafer_opt()), str(source),
        "--spmd-allocate-shared-memory", "--expand-strided-metadata",
        "--lower-affine", "--mk-to-wafer",
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "memref.copy" not in result.stdout
    assert "linalg.transpose" not in result.stdout
    # Both the original access and the copy must retain scalar memory ops;
    # WaferToLLVM needs the isSpm marker to apply the device SPM address map.
    loads = re.findall(r"memref.load[^\n]+", result.stdout)
    stores = re.findall(r"memref.store[^\n]+", result.stdout)
    assert len(loads) == len(stores) == 2, result.stdout
    assert all("isSpm = 1" in op for op in loads + stores), result.stdout
