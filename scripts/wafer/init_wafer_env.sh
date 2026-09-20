#!/usr/bin/env bash

WAFER_WORKSPACE_ROOT=${WAFER_WORKSPACE_ROOT:-}
WAFER_DEPS_ROOT=${WAFER_DEPS_ROOT:-}
WAFER_SDK_INCLUDE_DIR=${WAFER_SDK_INCLUDE_DIR:-}
WAFER_RT_THREAD_SMP_ROOT=${WAFER_RT_THREAD_SMP_ROOT:-}


if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "ERROR: source this script instead of executing it:" >&2
    echo "  source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

WAFER_WORKSPACE_ROOT=${WAFER_WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
CONDA_ROOT=${CONDA_ROOT:-$WAFER_WORKSPACE_ROOT/miniconda3}
ENV_NAME=${ENV_NAME:-wafer310}

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: Conda not found at $CONDA_ROOT" >&2
    return 1
fi
if [[ ! -d "$CONDA_ROOT/envs/$ENV_NAME" ]]; then
    echo "ERROR: Conda environment not found: $ENV_NAME" >&2
    return 1
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export WAFER_WORKSPACE_ROOT
export CONDA_ROOT
export ENV_NAME
export REPO_DIR=${REPO_DIR:-$WAFER_WORKSPACE_ROOT/DLCompiler}
export DEPS_ROOT=${DEPS_ROOT:-$WAFER_WORKSPACE_ROOT/deps}
export PACKAGE_ROOT=${PACKAGE_ROOT:-$WAFER_WORKSPACE_ROOT/packages}
export PYTHON="$CONDA_PREFIX/bin/python"
export LLVM_COMMIT=${LLVM_COMMIT:-7d5de3033187c8a3bb4d2e322f5462cdaf49808f}
export LLVM_SYSPATH=${LLVM_SYSPATH:-$DEPS_ROOT/llvm-7d5de303-ubuntu-x64}
export LLVM_BINARY_DIR="$LLVM_SYSPATH/bin"
export LLVM_DIR="$LLVM_SYSPATH/lib/cmake/llvm"
export MLIR_DIR="$LLVM_SYSPATH/lib/cmake/mlir"
export WAFER_DEPS_ROOT=${WAFER_DEPS_ROOT:-$DEPS_ROOT/wafer_deps}
export KUIPER_ROOT=${KUIPER_ROOT:-/usr/local/kuiper}
export WAFER_SDK_INCLUDE_DIR=${WAFER_SDK_INCLUDE_DIR:-$WAFER_DEPS_ROOT/include}
export WAFER_RT_THREAD_SMP_ROOT=${WAFER_RT_THREAD_SMP_ROOT:-$WAFER_DEPS_ROOT/tx8-yoc-rt-thread-smp}
export XUANTIE_NAME=${XUANTIE_NAME:-$WAFER_DEPS_ROOT/Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2}
export WAFER_BUILD_DIR=${WAFER_BUILD_DIR:-$WAFER_WORKSPACE_ROOT/build/wafer}
# Installed wheels include libvr.a; only override that default when this
# workspace also has a freshly built hardware CRT.
if [[ -z ${WAFER_RUNTIME_LIB_DIR:-} ]]; then
    for runtime_dir in "$WAFER_BUILD_DIR/tools/third_party/wafer/crt/lib" "$WAFER_BUILD_DIR/third_party/wafer/crt/lib"; do
        if [[ -f "$runtime_dir/libvr.a" ]]; then
            export WAFER_RUNTIME_LIB_DIR="$runtime_dir"
            break
        fi
    done
fi
export DICP_BACKEND=wafer
export USE_SIM_MODE=${USE_SIM_MODE:-1}
# This workspace uses Kuiper 1.4 firmware with the RCS device logging API.
export WAFER_DEVICE_LOG_ABI=${WAFER_DEVICE_LOG_ABI:-rcs}
export PATH="$LLVM_BINARY_DIR:$PATH"
export LD_LIBRARY_PATH="$KUIPER_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

hash -r 2>/dev/null || true

echo "Wafer build environment activated"
echo "  Conda: $CONDA_DEFAULT_ENV"
echo "  Python: $PYTHON"
echo "  LLVM: $LLVM_SYSPATH"
echo "  SDK: $WAFER_DEPS_ROOT"
echo "  Kuiper: $KUIPER_ROOT"
