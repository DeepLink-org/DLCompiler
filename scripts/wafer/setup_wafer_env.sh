#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LLVM_COMMIT=7d5de3033187c8a3bb4d2e322f5462cdaf49808f
: "${LLVM_SYSPATH:?Provide the local LLVM directory}"
: "${WAFER_DEPS_ROOT:?Provide the local Wafer SDK directory}"
WAFER_SDK_ROOT=$WAFER_DEPS_ROOT
ENV_FILE=${WAFER_ENV_FILE:-$SCRIPT_DIR/wafer_env.sh}

if [[ ! -x "$LLVM_SYSPATH/bin/llvm-config" ]]; then
    echo "ERROR: LLVM 22 package not found at $LLVM_SYSPATH" >&2
    exit 1
fi
if [[ $("$LLVM_SYSPATH/bin/llvm-config" --version) != "22.0.0git" ]]; then
    echo "ERROR: $LLVM_SYSPATH is not the required LLVM 22 package" >&2
    exit 1
fi
if [[ ! -f "$WAFER_SDK_ROOT/include/instr_def.h" ]]; then
    echo "ERROR: Wafer compiler header not found under $WAFER_SDK_ROOT/include" >&2
    exit 1
fi

cat >"$ENV_FILE" <<EOF
#!/usr/bin/env bash

export LLVM_COMMIT="$LLVM_COMMIT"
export LLVM_SYSPATH="$LLVM_SYSPATH"
export LLVM_BINARY_DIR="\$LLVM_SYSPATH/bin"
export LLVM_DIR="\$LLVM_SYSPATH/lib/cmake/llvm"
export MLIR_DIR="\$LLVM_SYSPATH/lib/cmake/mlir"
export WAFER_SDK_INCLUDE_DIR="$WAFER_SDK_ROOT/include"
export WAFER_DEPS_ROOT="$WAFER_SDK_ROOT"
export PATH="\$LLVM_BINARY_DIR:\${PATH:-}"
export DICP_BACKEND=wafer
export USE_SIM_MODE=1
EOF
chmod +x "$ENV_FILE"

echo "Wafer compiler environment is ready:"
echo "  LLVM_SYSPATH:  $LLVM_SYSPATH"
echo "  SDK headers:   $WAFER_SDK_ROOT/include"
echo "  activation:    source $ENV_FILE"
