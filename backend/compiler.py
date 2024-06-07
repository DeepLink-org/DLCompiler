from triton.backends.compiler import BaseBackend
from triton._C.libtriton import ir, passes
from dataclasses import dataclass
from typing import Any
import hashlib
import tempfile
import os
import re
import subprocess
import functools
from pathlib import Path

def _get_dicp_triton_opt_path() -> str:
    path = os.getenv("DICP_TRITON_OPT_PATH", "")
    if path == "":
        raise Exception("DICP_TRITON_OPT_PATH is not set.")
    return path

def _ttir_to_dicp_linalgdir(mod):
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "dicp_linalg.mlir")
        Path(src_path).write_text(ttir_code)
        triton_shared_opt_path = _get_dicp_triton_opt_path()
        subprocess.check_call([triton_shared_opt_path, src_path, "--triton-to-structured", "--canonicalize", "--triton-arith-to-linalg", "--cse", "--structured-to-memref", "-o", dst_path])
        return Path(dst_path).read_text()
    
def _optimize_dicp_linalgdir(linalgdir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return linalgdir

def _generate_execute_bin(linalgdir: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttshared_path = os.path.join(tmpdir, "dicp_linalg.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")


class DSABackend(BaseBackend):
    pass

