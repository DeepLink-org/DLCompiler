#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
COMPILER_ONLY=0
case "${1:-}" in
    --compiler-only) COMPILER_ONLY=1 ;;
    -h|--help)
        printf '%s\n' 'Usage: bash scripts/wafer/migrate_wafer_env.sh [--compiler-only]' \
            'Prepare LLVM_SYSPATH and WAFER_DEPS_ROOT locally before running.' \
            'Full mode also requires an existing matching Torch and Kuiper environment.' \
            'This command checks dependencies and writes wafer_env.sh; it does not install dependencies.'
        exit 0 ;;
    '') ;;
    *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
    printf '%s\n' 'ERROR: too many arguments' >&2
    exit 2
fi
: "${LLVM_SYSPATH:?Provide the local LLVM directory}"
: "${WAFER_DEPS_ROOT:?Provide the local Wafer SDK directory}"
if [[ $COMPILER_ONLY == 0 ]]; then
    "${PYTHON:-python3}" "$SCRIPT_DIR/test/wafer/verify_wafer_torch_stack.py"
fi
bash "$SCRIPT_DIR/scripts/wafer/setup_wafer_env.sh"