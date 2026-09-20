#!/bin/bash

set -euo pipefail

DUMP_ROOT=${TRITON_DUMP_PATH:-/tmp/tsm_dump}
DUMP_INDEX=${DUMP_INDEX:-1}
DUMP_DIR="$DUMP_ROOT/dump$DUMP_INDEX"
LINES=${LINES:-120}

if [ ! -d "$DUMP_DIR" ]; then
    echo "ERROR: dump directory not found: $DUMP_DIR" >&2
    echo "Run ./dump_vec_add_ir.sh first, or set TRITON_DUMP_PATH/DUMP_INDEX." >&2
    exit 1
fi

show_file() {
    local title="$1"
    local path="$2"
    if [ -f "$path" ]; then
        echo "========================================"
        echo " $title"
        echo "========================================"
        echo "FILE: $path"
        echo ""
        sed -n "1,${LINES}p" "$path"
        echo ""
    fi
}

echo "Dump directory: $DUMP_DIR"
echo ""
ls -lah "$DUMP_DIR"
echo ""

show_file "Commands" "$DUMP_DIR/cmds.txt"
show_file "TTIR" "$DUMP_DIR/tt_0.mlir"
show_file "CoreIR" "$DUMP_DIR/core_0.mlir"
show_file "Wafer IR" "$DUMP_DIR/wafer_0.mlir"
show_file "LLVM Dialect MLIR" "$DUMP_DIR/ll_0.mlir"
show_file "LLVM IR" "$DUMP_DIR/ll_0.ir"
show_file "Kernel LLVM IR" "$DUMP_DIR/kernel_0.ll"
