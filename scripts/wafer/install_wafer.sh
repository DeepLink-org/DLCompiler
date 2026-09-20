#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR=${WAFER_BUILD_DIR:-$SCRIPT_DIR/third_party/wafer/build_manual}
WHEEL_DIR=${WAFER_WHEEL_DIR:-$BUILD_DIR/wheel}
PYTHON=${PYTHON:-python3}
SKIP_BUILD=0

if [[ ${1:-} == "--skip-build" ]]; then
    SKIP_BUILD=1
elif [[ -n ${1:-} ]]; then
    echo "Usage: $0 [--skip-build]" >&2
    exit 2
fi

if [[ -f "$SCRIPT_DIR/wafer_env.sh" ]]; then
    source "$SCRIPT_DIR/wafer_env.sh"
fi

: "${LLVM_SYSPATH:?Run 'bash scripts/wafer/setup_wafer_env.sh' before installing Wafer}"
: "${LLVM_BINARY_DIR:?LLVM_BINARY_DIR is not set}"

if [[ $SKIP_BUILD == 0 ]]; then
    bash "$SCRIPT_DIR/scripts/wafer/compile_wafer.sh"
fi

for artifact in "$BUILD_DIR/wafer-build.json"; do
    if [[ ! -f "$artifact" ]]; then
        echo "ERROR: required build artifact not found: $artifact" >&2
        exit 1
    fi
done

rm -rf "$WHEEL_DIR"
"$PYTHON" "$SCRIPT_DIR/setup_on_wafer.py" \
    --build-dir "$BUILD_DIR" \
    --wheel-dir "$WHEEL_DIR"

wheel=$(find "$WHEEL_DIR" -maxdepth 1 -name 'triton-*.whl' -type f -print -quit)
if [[ -z "$wheel" ]]; then
    echo "ERROR: Wafer wheel was not produced under $WHEEL_DIR" >&2
    exit 1
fi
"$PYTHON" -m pip install --no-index --no-deps --force-reinstall "$wheel"

(
    cd /tmp
    DICP_BACKEND=wafer USE_SIM_MODE=1 LLVM_BINARY_DIR="$LLVM_BINARY_DIR" \
        "$PYTHON" - <<'PY'
import tempfile
from pathlib import Path

import triton
from triton._C import libtriton
from triton.backends import backends
from triton.backends.compiler import GPUTarget

if list(backends) != ["dicp_triton"]:
    raise RuntimeError(f"Unexpected Triton backends: {list(backends)}")

target = GPUTarget("wafer", "wafer", 32)
backend = backends["dicp_triton"].compiler(target)
backend.load_dialects(libtriton.ir.context())

import triton.language.extra.wafer  # noqa: F401, E402
import triton.experimental.tle.language  # noqa: F401, E402

if hasattr(libtriton, "dicp_triton"):
    raise RuntimeError("The Wafer-only package unexpectedly contains the original DICP C++ binding")

with tempfile.TemporaryDirectory() as tmpdir:
    source = Path(tmpdir) / "wafer_install_check.ttir"
    source.write_text(
        'module { tt.func public @wafer_install_check() '
        'attributes {tt.kernel = 1 : i1} { tt.return } }'
    )
    kernel = triton.compile(str(source), target=target)
    if not kernel.asm["o"].startswith(b"\x7fELF"):
        raise RuntimeError("Wafer compiler did not produce an ELF object")
    print("Wafer compiler installation verified:")
    print("  triton:", triton.__file__)
    print("  libtriton:", libtriton.__file__)
    print("  stages:", list(kernel.asm))
    print("  object bytes:", len(kernel.asm["o"]))
PY
)
