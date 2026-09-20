#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
DLC_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

if [[ -z ${LLVM_BINARY_DIR:-} ]]; then
    : "${LLVM_SYSPATH:?Set LLVM_SYSPATH or LLVM_BINARY_DIR before running this script}"
    LLVM_BINARY_DIR="$LLVM_SYSPATH/bin"
fi
TRITON_DUMP_PATH=${TRITON_DUMP_PATH:-/tmp/tsm_dump}
DUMP_INDEX=${DUMP_INDEX:-1}
DUMP_DIR="$TRITON_DUMP_PATH/dump$DUMP_INDEX"

rm -rf "$TRITON_DUMP_PATH"
mkdir -p "$TRITON_DUMP_PATH"

export DICP_BACKEND=${DICP_BACKEND:-wafer}
export USE_SIM_MODE=${USE_SIM_MODE:-1}
export LLVM_BINARY_DIR
export TRITON_DUMP_PATH
export TRITON_ALWAYS_COMPILE=1
export MLIR_ENABLE_DUMP=1

cd "$SCRIPT_DIR"

python3 - <<'PY'
import torch
import test_vec_add as v

x = torch.rand(1024, device="cpu")
y = torch.rand(1024, device="cpu")
v.add(x, y)
PY

echo ""
echo "Dump directory: $DUMP_DIR"
echo ""
ls -lah "$DUMP_DIR"
echo ""
echo "Common files:"
for f in tt_0.mlir core_0.mlir wafer_0.mlir ll_0.mlir ll_0.ir kernel_0.ll kernel_0.o cmds.txt; do
    if [ -e "$DUMP_DIR/$f" ]; then
        echo "  $DUMP_DIR/$f"
    fi
done
